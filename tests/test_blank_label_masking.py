"""How a blank CheXpert cell reaches the loss, under each blank-label policy.

79.4% of the CheXpert label matrix is blank, and blank means the labeler found
no mention of the finding. `model.mhcac.blank_label_policy` picks the reading:

* ``ignore`` -- blank is masked per cell and must NEVER train as a negative.
  The policy from 2026-08-13 to 2026-09-24, and the default when the key is
  absent. The failure it guards against is silent: training a blank as a
  negative lowers the loss just as smoothly as training a real one.
* ``negative`` -- blank is class 0, as in the original META-CXR paper
  ("missing (NaN) values were treated as the negative class"). Shipped from
  2026-09-24.

Under both, a study with no CheXpert information must stay IGNORE_LABEL on
every cell, and the mention-gate targets must not depend on the policy.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from mhcac.explanation import logit_difference_squared
from mhcac.loss import ClassificationLoss
from model.lavis.data.chexpert_labels import (
    BLANK_AS_NEGATIVE,
    BLANK_IGNORED,
    BLANK_LABEL_POLICIES,
    DEFAULT_BLANK_LABEL_POLICY,
    IGNORE_LABEL,
    attach_chexpert_labels,
    map_chexpert_labels,
    mention_column,
    prepare_chexpert_labels,
    resolve_blank_label_policy,
)

COLS = ["No Finding", "Cardiomegaly", "Edema"]


def _export():
    """A tiny CheXpert export: one study per interesting pattern."""
    return pd.DataFrame(
        {
            "subject_id": [1, 2, 3, 4],
            "study_id": [10, 20, 30, 40],
            "No Finding": [np.nan, 1.0, np.nan, np.nan],
            "Cardiomegaly": [1.0, np.nan, -1.0, np.nan],
            "Edema": [0.0, np.nan, np.nan, np.nan],  # study 40: all blank
        }
    )


def _annotation():
    """Split rows: studies 10..40 have a record, study 50 matched none."""
    return pd.DataFrame(
        {
            "subject_id": [1, 2, 3, 4, 5],
            "study_id": [10, 20, 30, 40, 50],
            "dicom_id": ["a", "b", "c", "d", "e"],
        }
    )


# --------------------------------------------------------------------------
# The policy switch itself
# --------------------------------------------------------------------------


def test_missing_key_keeps_the_historical_ignore_policy():
    assert DEFAULT_BLANK_LABEL_POLICY == BLANK_IGNORED
    assert resolve_blank_label_policy(None) == BLANK_IGNORED


@pytest.mark.parametrize("bad", ["Negative", "zero", "", "mask", 0])
def test_an_unknown_policy_raises_rather_than_falling_back(bad):
    with pytest.raises(ValueError, match="blank_label_policy"):
        resolve_blank_label_policy(bad)
    with pytest.raises(ValueError, match="blank_label_policy"):
        map_chexpert_labels(_export(), COLS, bad)


def test_shipped_config_selects_negative():
    import yaml

    cfg_path = Path(__file__).resolve().parents[1] / "pretraining/configs/mimic_cxr_full.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    assert cfg["model"]["mhcac"]["blank_label_policy"] == BLANK_AS_NEGATIVE


# --------------------------------------------------------------------------
# The mapping, per policy
# --------------------------------------------------------------------------


def test_ignore_maps_blank_to_the_sentinel_not_negative():
    frame = pd.DataFrame({"Cardiomegaly": [1.0, 0.0, -1.0, np.nan], "Edema": 1.0})

    mapped = map_chexpert_labels(frame, ["Cardiomegaly", "Edema"], BLANK_IGNORED)
    column = mapped["Cardiomegaly"].tolist()

    assert column == [1, 0, 2, IGNORE_LABEL]
    assert column[3] != 0, "a blank must not be indistinguishable from a negative"


def test_negative_maps_blank_to_class_zero():
    frame = pd.DataFrame({"Cardiomegaly": [1.0, 0.0, -1.0, np.nan], "Edema": 1.0})

    mapped = map_chexpert_labels(frame, ["Cardiomegaly", "Edema"], BLANK_AS_NEGATIVE)

    assert mapped["Cardiomegaly"].tolist() == [1, 0, 2, 0]


@pytest.mark.parametrize("policy", BLANK_LABEL_POLICIES)
def test_explicit_values_do_not_depend_on_the_policy(policy):
    frame = pd.DataFrame({"Cardiomegaly": [1.0, 0.0, -1.0]})

    mapped = map_chexpert_labels(frame, ["Cardiomegaly"], policy)

    assert mapped["Cardiomegaly"].tolist() == [1, 0, 2]


@pytest.mark.parametrize("policy", BLANK_LABEL_POLICIES)
def test_an_all_blank_record_stays_ignored_under_both_policies(policy):
    """Fourteen blanks carry no information; they must not become a normal study."""
    mapped = map_chexpert_labels(_export(), COLS, policy)

    assert mapped.iloc[3].tolist() == [IGNORE_LABEL] * len(COLS)


@pytest.mark.parametrize("policy", BLANK_LABEL_POLICIES)
def test_ignore_sentinel_survives_int8_storage(policy):
    """int8 spans -128..127, so the sentinel must not wrap into a real class."""
    mapped = map_chexpert_labels(_export(), COLS, policy)
    assert mapped.dtypes.eq("int8").all()
    stored = pd.Series([IGNORE_LABEL], dtype="int8").iloc[0]
    assert int(stored) == IGNORE_LABEL < 0


def test_preprocessing_applies_the_identical_mapping(monkeypatch):
    """preprocess_mimic_cxr.py keeps its own copy; the two must never diverge."""
    path = Path(__file__).resolve().parents[1] / "preporcessing/preprocess_mimic_cxr.py"
    pytest.importorskip("tqdm")
    monkeypatch.syspath_prepend(str(path.parent))
    spec = importlib.util.spec_from_file_location("_preprocess_mimic_cxr", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.IGNORE_LABEL == IGNORE_LABEL
    assert set(module.BLANK_LABEL_POLICIES) == set(BLANK_LABEL_POLICIES)
    for policy in BLANK_LABEL_POLICIES:
        cleaned, cols = module.clean_chexpert(_export(), policy)
        expected = map_chexpert_labels(_export(), cols, policy)
        pd.testing.assert_frame_equal(
            cleaned[cols].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_dtype=True,
        )
    with pytest.raises(ValueError, match="blank_label_policy"):
        module.clean_chexpert(_export(), "zero")


# --------------------------------------------------------------------------
# The full dataset path: prepare -> merge onto split rows
# --------------------------------------------------------------------------


def _attached(policy, excluded=()):
    prepared = prepare_chexpert_labels(
        _export(), COLS, blank_policy=policy, excluded_labels=list(excluded)
    )
    return attach_chexpert_labels(_annotation(), prepared, COLS)


@pytest.mark.parametrize("policy", BLANK_LABEL_POLICIES)
def test_a_study_without_a_chexpert_record_stays_ignored(policy):
    merged = _attached(policy)
    unmatched = merged[merged["study_id"] == 50].iloc[0]

    assert [int(unmatched[c]) for c in COLS] == [IGNORE_LABEL] * len(COLS)
    assert not bool(unmatched["classification_valid"])
    assert not bool(unmatched["mention_valid"])


def test_mention_targets_are_identical_under_both_policies():
    mention_cols = [mention_column(c) for c in COLS]
    ignore = _attached(BLANK_IGNORED)
    negative = _attached(BLANK_AS_NEGATIVE)

    pd.testing.assert_frame_equal(ignore[mention_cols], negative[mention_cols])
    pd.testing.assert_series_equal(ignore["mention_valid"], negative["mention_valid"])
    # And they are the raw notna pattern, not something derived after the fill.
    assert ignore[mention_cols].iloc[0].tolist() == [0, 1, 1]
    assert ignore[mention_cols].iloc[4].tolist() == [0, 0, 0]


def test_ignore_policy_masks_blanks_of_a_matched_study():
    merged = _attached(BLANK_IGNORED)
    first = merged[merged["study_id"] == 10].iloc[0]

    assert [int(first[c]) for c in COLS] == [IGNORE_LABEL, 1, 0]


def test_negative_policy_fills_blanks_of_a_matched_study():
    merged = _attached(BLANK_AS_NEGATIVE)
    by_study = merged.set_index("study_id")

    assert by_study.loc[10, COLS].tolist() == [0, 1, 0]
    assert by_study.loc[20, COLS].tolist() == [1, 0, 0]
    assert by_study.loc[30, COLS].tolist() == [0, 2, 0]
    assert by_study.loc[40, COLS].tolist() == [IGNORE_LABEL] * len(COLS)


@pytest.mark.parametrize("policy", BLANK_LABEL_POLICIES)
def test_excluded_labels_are_applied_after_the_fill(policy):
    merged = _attached(policy, excluded=["No Finding"])

    assert (merged["No Finding"] == IGNORE_LABEL).all()
    # The gate still sees No Finding's raw pattern.
    assert merged[mention_column("No Finding")].tolist() == [0, 1, 0, 0, 0]


def test_a_study_whose_only_label_is_excluded_regains_a_cell_under_negative():
    """Study 20 carries only `No Finding`. Masked, it has nothing to train on
    under `ignore`; under `negative` its other blanks are real negatives."""
    ignore = _attached(BLANK_IGNORED, excluded=["No Finding"]).set_index("study_id")
    negative = _attached(BLANK_AS_NEGATIVE, excluded=["No Finding"]).set_index("study_id")

    assert not bool(ignore.loc[20, "classification_valid"])
    assert bool(negative.loc[20, "classification_valid"])
    assert not bool(negative.loc[40, "classification_valid"]), "all-blank stays out"
    assert not bool(negative.loc[50, "classification_valid"]), "unmatched stays out"


def test_processed_flag_still_guards_the_join():
    prepared = prepare_chexpert_labels(_export(), COLS, blank_policy=BLANK_AS_NEGATIVE)
    claims = np.array([True, True, True, True, True])  # 40 and 50 have nothing

    with pytest.raises(ValueError, match="claim CheXpert labels"):
        attach_chexpert_labels(_annotation(), prepared, COLS, processed_has_label=claims)


# --------------------------------------------------------------------------
# What the loss does with the result
# --------------------------------------------------------------------------


def test_masked_cells_contribute_no_gradient():
    torch.manual_seed(0)
    logits = torch.randn(4, 2, 3, requires_grad=True)
    labels = torch.tensor([[1, IGNORE_LABEL], [0, IGNORE_LABEL],
                           [1, IGNORE_LABEL], [0, IGNORE_LABEL]])

    ClassificationLoss(num_abnormalities=2)(logits, labels).backward()

    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad[:, 0]) > 0, "label 0 should train"
    assert torch.equal(logits.grad[:, 1], torch.zeros_like(logits.grad[:, 1])), (
        "every cell of label 1 is blank, so it must receive no gradient"
    )


def test_blank_label_loss_matches_dropping_the_label_entirely():
    torch.manual_seed(0)
    both = torch.randn(4, 2, 3)
    labels_masked = torch.tensor([[1, IGNORE_LABEL], [0, IGNORE_LABEL],
                                  [1, IGNORE_LABEL], [0, IGNORE_LABEL]])
    only = both[:, :1].clone()
    labels_only = labels_masked[:, :1]

    loss_masked = ClassificationLoss(num_abnormalities=2)(both, labels_masked)
    loss_only = ClassificationLoss(num_abnormalities=1)(only, labels_only)

    assert loss_masked == pytest.approx(float(loss_only))


def test_negative_policy_blanks_do_train_as_class_zero():
    torch.manual_seed(0)
    logits = torch.randn(3, 1, 3, requires_grad=True)
    frame = pd.DataFrame({"Edema": [np.nan, np.nan, 1.0], "Cardiomegaly": 1.0})
    mapped = map_chexpert_labels(frame, ["Edema", "Cardiomegaly"], BLANK_AS_NEGATIVE)
    labels = torch.tensor(mapped[["Edema"]].to_numpy(dtype=np.int64))

    ClassificationLoss(num_abnormalities=1)(logits, labels).backward()

    assert labels[:, 0].tolist() == [0, 0, 1]
    assert torch.count_nonzero(logits.grad[:2]) > 0, "a filled blank must train"


def test_blank_cells_are_not_read_as_positive_by_the_explanation_score():
    logits = torch.randn(1, 3, 3)
    labels = torch.tensor([[IGNORE_LABEL, IGNORE_LABEL, IGNORE_LABEL]])

    score, valid = logit_difference_squared(logits, labels)

    assert float(score) == 0.0
    assert not bool(valid[0]), "an all-blank study has no positive finding to explain"
