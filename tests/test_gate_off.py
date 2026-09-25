"""The mention gate is OFF in production, and off must mean OFF (2026-09-24, D-019).

With `blank_label_policy: negative` the P/N/U head learns "not mentioned" as
class 0 directly, so the gate's reason to exist -- somewhere to put the blanks
that the masked policy threw away -- is gone. `lambda_gate` is 0 and the code is
kept for ablation. Three ways "off" could silently not be off:

1. a gate weight table reaching the P/N/U cross entropy, so the class weights
   recomputed for the new policy are not the ones actually used;
2. the gate BCE still entering the total loss;
3. an evaluator forming sigmoid(mention) x q_pos from heads that never trained,
   i.e. multiplying the real score by noise.

Also pins that No Finding is back in the head with real 0/1 labels, and that the
evaluator reports 13- and 14-label macros beside the historical 12-label one.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from mhcac.loss import (
    ClassificationLoss,
    MentionGateLoss,
    build_classification_losses,
    mention_gate_is_trained,
)

_REPO = Path(__file__).resolve().parents[1]
_CONFIG = _REPO / "pretraining" / "configs" / "mimic_cxr_full.yaml"


@pytest.fixture(scope="module")
def cfg():
    yaml = pytest.importorskip("yaml")
    return yaml.safe_load(_CONFIG.read_text())


def _shipped_builder_args(cfg, **overrides):
    loss, mhcac = cfg["model"]["loss"], cfg["model"]["mhcac"]
    args = dict(
        class_weights=[list(row) for row in mhcac["class_weights"]],
        label_smoothing=float(mhcac.get("label_smoothing", 0.0)),
        uncertain_policy=mhcac["uncertain_policy"],
        lambda_cls=loss["lambda_cls"],
        lambda_gate=loss["lambda_gate"],
        lambda_mention_conditioned_cls=loss["lambda_mention_conditioned_cls"],
        gate_class_weights=mhcac.get("gate_class_weights"),
        mention_conditioned_pos_weights=mhcac.get("mention_conditioned_pos_weights"),
    )
    args.update(overrides)
    return args


# --------------------------------------------------------------------------
# The shipped configuration
# --------------------------------------------------------------------------


def test_shipped_recipe_trains_no_gate_objective(cfg):
    loss = cfg["model"]["loss"]
    assert loss["lambda_gate"] == 0.0
    assert loss["lambda_mention_conditioned_cls"] == 0.0
    assert not mention_gate_is_trained(loss["lambda_gate"], loss["lambda_mention_conditioned_cls"])


def test_shipped_recipe_keeps_the_decisions_the_user_fixed(cfg):
    mhcac = cfg["model"]["mhcac"]
    assert mhcac["blank_label_policy"] == "negative"
    assert mhcac["excluded_labels"] == []
    # D-022 (2026-09-25): Uncertain is a trained third class, as in the paper,
    # and the eval side scores under the same policy.
    assert mhcac["uncertain_policy"] == "three_class"
    assert cfg["run"]["uncertain_policy"] == "three_class"


def test_shipped_loss_trains_uncertain_as_its_own_class(cfg):
    """D-022: an Uncertain cell moves the shipped P/N/U loss; under the old
    ignore_uncertain it was dropped and w_uncertain was inert."""
    torch.manual_seed(0)
    logits = torch.randn(4, 14, 3)
    labels = torch.zeros(4, 14, dtype=torch.long)
    labels[0, 3] = 2  # one Uncertain cell, Lung Opacity
    shipped, *_ = build_classification_losses(**_shipped_builder_args(cfg))
    ignoring, *_ = build_classification_losses(
        **_shipped_builder_args(cfg, uncertain_policy="ignore_uncertain")
    )
    as_negative = labels.clone()
    as_negative[0, 3] = 0

    assert float(shipped(logits, labels)) != pytest.approx(float(shipped(logits, as_negative)))
    # And the shipped policy differs from the one it replaced on the same batch.
    assert float(ignoring(logits, labels)) != pytest.approx(float(shipped(logits, labels)))


def test_no_finding_uncertain_weight_is_neutral_not_a_division_by_zero(cfg):
    no_finding = cfg["model"]["mhcac"]["class_weights"][0]
    assert no_finding[2] == 1.0
    assert all(np.isfinite(no_finding))


@pytest.mark.parametrize(
    ("lambda_gate", "lambda_mc", "trained"),
    [(0.0, 0.0, False), (0.5, 0.0, True), (0.0, 1.0, True)],
)
def test_gate_is_trained_only_when_an_objective_reads_it(lambda_gate, lambda_mc, trained):
    assert mention_gate_is_trained(lambda_gate, lambda_mc) is trained


# --------------------------------------------------------------------------
# (b) nothing gate-related reaches the P/N/U head
# --------------------------------------------------------------------------


def test_pnu_loss_uses_exactly_the_configured_class_weights(cfg):
    cls_fn, gate_fn, mc_fn = build_classification_losses(**_shipped_builder_args(cfg))

    assert isinstance(cls_fn, ClassificationLoss)
    assert isinstance(gate_fn, MentionGateLoss)
    assert mc_fn is None, "mention_conditioned_pos_weights must not even be built"
    for row, ce in zip(cfg["model"]["mhcac"]["class_weights"], cls_fn.cross_entropy_loss_list,
                       strict=True):
        assert torch.allclose(ce.weight, torch.tensor(row, dtype=torch.float))


def test_gate_weight_tables_cannot_change_the_pnu_loss(cfg):
    torch.manual_seed(0)
    logits = torch.randn(8, 14, 3)
    labels = torch.randint(0, 2, (8, 14))

    shipped, *_ = build_classification_losses(**_shipped_builder_args(cfg))
    perturbed, *_ = build_classification_losses(
        **_shipped_builder_args(
            cfg,
            gate_class_weights=[9.0] * 14,
            mention_conditioned_pos_weights=[7.0] * 14,
        )
    )

    assert float(shipped(logits, labels)) == pytest.approx(float(perturbed(logits, labels)))


def test_the_forward_only_adds_the_gate_bce_while_lambda_gate_is_positive():
    """blip2_qformer cannot be imported on the CPU box; read its source instead."""
    source = (_REPO / "model/lavis/models/blip2_models/blip2_qformer.py").read_text()
    guarded = source.split("if self.lambda_gate > 0:", 1)
    assert len(guarded) == 2, "the gate BCE must stay behind `if self.lambda_gate > 0`"
    assert "total_loss = total_loss + self.lambda_gate * loss_gate" in guarded[1][:900]
    assert source.count("self.lambda_gate * loss_gate") == 1


def test_hierarchical_conflicts_still_raise(cfg):
    with pytest.raises(ValueError, match="subsumes lambda_gate"):
        build_classification_losses(
            **_shipped_builder_args(cfg, lambda_gate=0.5, lambda_mention_conditioned_cls=1.0,
                                    lambda_cls=0.0)
        )
    with pytest.raises(ValueError, match="subsumes lambda_cls"):
        build_classification_losses(
            **_shipped_builder_args(cfg, lambda_mention_conditioned_cls=1.0)
        )


# --------------------------------------------------------------------------
# (c) eval / calibrate never read an untrained gate
# --------------------------------------------------------------------------


def _predictions(metadata, with_mention=True):
    from training.evaluation.schemas import ClassificationPredictions

    rng = np.random.default_rng(0)
    probs = rng.dirichlet(np.ones(3), size=(20, 14))
    return ClassificationPredictions(
        labels=rng.integers(0, 2, size=(20, 14)),
        probabilities=probs,
        pathology_names=tuple(f"p{i}" for i in range(14)),
        sample_keys=np.asarray([f"k{i}" for i in range(20)]),
        mention_probabilities=rng.random((20, 14)) if with_mention else None,
        metadata=metadata,
    )


def test_marginal_presence_raises_on_a_gate_off_prediction_file():
    from training.evaluation.label_framing import (
        MENTION_GATE_TRAINED_KEY,
        ScoreUnavailableError,
        apply_framing,
    )

    # Even if some array were present, a file marked gate-off must refuse.
    predictions = _predictions({MENTION_GATE_TRAINED_KEY: False}, with_mention=True)
    with pytest.raises(ScoreUnavailableError, match="never trained"):
        apply_framing(predictions, "study_presence", "marginal_presence")


def test_default_score_is_q_pos_and_works_without_a_gate():
    from training.evaluation.label_framing import (
        DEFAULT_SCORE,
        MENTION_GATE_TRAINED_KEY,
        POSITIVE,
        apply_framing,
    )

    assert DEFAULT_SCORE == "conditional_positive"
    predictions = _predictions({MENTION_GATE_TRAINED_KEY: False}, with_mention=False)
    framed = apply_framing(predictions, "study_presence", DEFAULT_SCORE)
    assert np.allclose(framed.probabilities[..., POSITIVE],
                       predictions.probabilities[..., POSITIVE])


def test_older_gate_trained_files_still_score_marginal():
    from training.evaluation.label_framing import apply_framing

    apply_framing(_predictions({}, with_mention=True), "study_presence", "marginal_presence")


def test_eval_hook_writes_the_key_label_framing_reads():
    from training.evaluation.label_framing import MENTION_GATE_TRAINED_KEY

    source = (_REPO / "model/lavis/tasks/image_text_pretrain.py").read_text()
    assert f'predictions.metadata["{MENTION_GATE_TRAINED_KEY}"]' in source
    assert "mention_gate_trained" in source


def test_both_clis_default_to_the_conditional_score():
    for script in ("scripts/evaluate_stage1.py", "scripts/calibrate_thresholds.py"):
        source = (_REPO / script).read_text()
        assert "default=DEFAULT_SCORE" in source or 'default="conditional_positive"' in source, script


# --------------------------------------------------------------------------
# No Finding back in the head, and the 13/14-label macros
# --------------------------------------------------------------------------


def test_no_finding_has_real_zero_one_labels_under_the_shipped_config(cfg):
    from model.lavis.data.chexpert_labels import IGNORE_LABEL, prepare_chexpert_labels

    cols = ["No Finding", "Cardiomegaly", "Edema"]
    export = pd.DataFrame(
        {
            "subject_id": [1, 2, 3],
            "study_id": [10, 20, 30],
            "No Finding": [1.0, np.nan, np.nan],
            "Cardiomegaly": [np.nan, 1.0, 0.0],
            "Edema": [np.nan, np.nan, -1.0],
        }
    )
    mhcac = cfg["model"]["mhcac"]
    prepared = prepare_chexpert_labels(
        export, cols,
        blank_policy=mhcac["blank_label_policy"],
        excluded_labels=mhcac["excluded_labels"],
    )

    no_finding = prepared["No Finding"].tolist()
    assert no_finding == [1, 0, 0]
    assert IGNORE_LABEL not in no_finding


def test_evaluator_reports_12_13_and_14_label_macros():
    from training.evaluation.classification_metrics import evaluate_classification
    from training.evaluation.schemas import ClassificationPredictions

    names = (
        "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
        "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
        "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture",
        "Support Devices",
    )
    rng = np.random.default_rng(1)
    predictions = ClassificationPredictions(
        labels=rng.integers(0, 2, size=(200, 14)),
        probabilities=rng.dirichlet(np.ones(3), size=(200, 14)),
        pathology_names=names,
        sample_keys=np.asarray([f"k{i}" for i in range(200)]),
    )
    report = evaluate_classification(predictions, uncertain_policy="ignore_uncertain")
    auroc = {m.name: m.auroc for m in report.per_pathology}

    assert len(report.macro_pathologies) == 12
    assert report.aggregates["macro_auroc"] == pytest.approx(
        np.mean([v for k, v in auroc.items() if k not in {"No Finding", "Support Devices"}]))
    assert report.aggregates["macro_auroc_13labels"] == pytest.approx(
        np.mean([v for k, v in auroc.items() if k != "No Finding"]))
    assert report.aggregates["macro_auroc_14labels"] == pytest.approx(np.mean(list(auroc.values())))
    for key in ("positive_macro_f1", "macro_auprc", "macro_specificity"):
        assert f"{key}_13labels" in report.aggregates
        assert f"{key}_14labels" in report.aggregates
