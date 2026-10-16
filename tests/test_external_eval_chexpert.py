"""CheXpert validation set served through MIMIC_CXR_Dataset (paper Table 4).

`model/lavis/data/external_eval.py` turns CheXpert's ``valid.csv`` into the
two tables the MIMIC-CXR dataset reads. The failures pinned here are the silent
ones: scoring a labeler-derived ``train.csv`` (with -1 uncertain) as if it were
the radiologist-labelled validation set, a frontal view mapped to the wrong
projection, and a label table that no longer joins one study to one row.
"""

import pandas as pd
import pytest

from model.lavis.data.chexpert_labels import (
    BLANK_AS_NEGATIVE,
    attach_chexpert_labels,
    prepare_chexpert_labels,
)
from model.lavis.data.external_eval import (
    CHEXPERT_VAL,
    ExternalEval,
    chexpert_val_tables,
    load_chexpert_val,
    resolve_external_eval,
)
from model.lavis.data.mimic_cxr_utils import build_study_index

COLS = ["No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
        "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
        "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture",
        "Support Devices"]
ROOT = "CheXpert-v1.0-small/valid"


def _row(path, fl, appa, positives=()):
    row = {"Path": path, "Sex": "Female", "Age": 70, "Frontal/Lateral": fl, "AP/PA": appa}
    row.update({c: (1.0 if c in positives else 0.0) for c in COLS})
    return row


def _valid():
    return pd.DataFrame([
        _row(f"{ROOT}/patient64541/study1/view1_frontal.jpg", "Frontal", "AP", ["Edema"]),
        _row(f"{ROOT}/patient64542/study1/view1_frontal.jpg", "Frontal", "PA", ["Cardiomegaly"]),
        _row(f"{ROOT}/patient64542/study1/view2_lateral.jpg", "Lateral", None, ["Cardiomegaly"]),
        _row(f"{ROOT}/patient64543/study2/view1_frontal.jpg", "Frontal", None, ["No Finding"]),
    ])


def test_tables_have_the_mimic_schema_and_one_label_row_per_study():
    reports, labels = chexpert_val_tables(_valid(), COLS)
    for col in ("subject_id", "study_id", "dicom_id", "image_path", "findings_clean",
                "has_chexpert_label", "target_valid", "ViewPosition"):
        assert col in reports
    assert len(reports) == 4 and len(labels) == 3
    assert not labels.duplicated(["subject_id", "study_id"]).any()
    assert reports["image_path"].iloc[0] == f"{ROOT}/patient64541/study1/view1_frontal.jpg"
    assert not reports["target_valid"].any()
    assert reports["findings_clean"].eq("").all()
    assert reports["dicom_id"].is_unique


def test_view_positions_follow_frontal_lateral_and_ap_pa():
    reports, _ = chexpert_val_tables(_valid(), COLS)
    assert reports["ViewPosition"].tolist() == ["AP", "PA", "LATERAL", "UNKNOWN"]


def test_frontal_is_the_anchor_and_lateral_the_auxiliary_view():
    reports, _ = chexpert_val_tables(_valid(), COLS)
    studies = build_study_index(reports.to_dict("records"), max_aux_views=1)
    two_view = [s for s in studies if s["study_key"] == (64542, 1)][0]
    assert reports.loc[two_view["anchor"], "ViewPosition"] == "PA"
    assert [reports.loc[p, "ViewPosition"] for p in two_view["aux"]] == ["LATERAL"]


def test_labels_join_as_zero_one_through_the_production_mapping():
    reports, labels = chexpert_val_tables(_valid(), COLS)
    prepared = prepare_chexpert_labels(labels, COLS, blank_policy=BLANK_AS_NEGATIVE)
    merged = attach_chexpert_labels(
        reports, prepared, COLS, processed_has_label=reports["has_chexpert_label"].to_numpy()
    )
    assert merged["classification_valid"].all()
    assert set(merged[COLS].to_numpy().ravel().tolist()) <= {0, 1}
    first = merged.iloc[0]
    assert first["Edema"] == 1 and first["Cardiomegaly"] == 0


def test_uncertain_or_blank_labels_are_refused():
    for value in (-1.0, float("nan")):
        bad = _valid()
        bad.loc[0, "Edema"] = value
        with pytest.raises(ValueError, match="must be 0/1"):
            chexpert_val_tables(bad, COLS)


def test_a_train_csv_path_is_refused():
    bad = _valid()
    bad.loc[0, "Path"] = "CheXpert-v1.0-small/train/patient00001/study1/view1_frontal.jpg"
    with pytest.raises(ValueError, match="not CheXpert validation images"):
        chexpert_val_tables(bad, COLS)


def test_views_of_one_study_with_different_labels_are_refused():
    bad = _valid()
    bad.loc[2, "Edema"] = 1.0
    with pytest.raises(ValueError, match="study-level"):
        chexpert_val_tables(bad, COLS)


def test_missing_columns_are_named():
    with pytest.raises(ValueError, match="Fracture"):
        chexpert_val_tables(_valid().drop(columns=["Fracture"]), COLS)


def _cfg(**over):
    cfg = {"kind": CHEXPERT_VAL, "valid_csv": "/x/valid.csv", "vis_root": "/x", "splits": ["test"]}
    cfg.update(over)
    return cfg


def test_resolve_replaces_only_the_listed_eval_splits():
    assert resolve_external_eval(None, "test") is None
    assert resolve_external_eval(_cfg(), "val") is None
    assert resolve_external_eval(_cfg(), "train") is None
    ext = resolve_external_eval(_cfg(), "test")
    assert ext.kind == CHEXPERT_VAL and ext.splits == ("test",)


@pytest.mark.parametrize("over, match", [
    ({"splits": ["train"]}, "never replaces train"),
    ({"splits": []}, "non-empty"),
    ({"kind": "vindr"}, "kind"),
    ({"vis_root": None}, "missing keys"),
    ({"extra": 1}, "unknown keys"),
])
def test_resolve_refuses_bad_configs(over, match):
    with pytest.raises((ValueError, TypeError), match=match):
        resolve_external_eval(_cfg(**over), "test")


def test_load_requires_every_image_under_vis_root(tmp_path):
    csv = tmp_path / "CheXpert-v1.0-small" / "valid.csv"
    csv.parent.mkdir(parents=True)
    _valid().to_csv(csv, index=False)
    ext = ExternalEval(CHEXPERT_VAL, csv, tmp_path, ("test",))
    with pytest.raises(FileNotFoundError, match="4 of 4"):
        load_chexpert_val(ext, COLS)
    for path in _valid()["Path"]:
        image = tmp_path / path
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"")
    reports, labels = load_chexpert_val(ext, COLS)
    assert len(reports) == 4 and len(labels) == 3
