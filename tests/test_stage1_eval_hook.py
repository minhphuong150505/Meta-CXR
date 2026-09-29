"""The Stage-1 evaluation hook reports the paper's three-class metrics only.

``model/lavis/tasks/image_text_pretrain.py`` scores every validation pass with
``training/evaluation/classification_metrics.evaluate_classification``: argmax
over Negative / Positive / Uncertain, per-finding weighted P/R/F1, one-vs-rest
AUROC per class. The positive-only / binary keys it used to report were removed
on 2026-09-29 (D-023) and must not come back.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

image_text_pretrain = pytest.importorskip(
    "model.lavis.tasks.image_text_pretrain",
    reason="LAVIS task module needs the full model stack",
)
ImageTextPretrainTask = image_text_pretrain.ImageTextPretrainTask


class FakeModel(torch.nn.Module):
    """Returns pre-baked logits, so the metric maths is the only variable."""

    def __init__(self, outputs):
        super().__init__()
        self.outputs = outputs
        self.calls = 0
        self._parameter = torch.nn.Parameter(torch.zeros(1))

    def forward(self, batch):
        output = self.outputs[self.calls]
        self.calls += 1
        return output


def make_batch(labels, mask=None):
    labels = torch.tensor(labels, dtype=torch.long)
    batch = {"classification_labels": labels, "text_output": [""] * labels.shape[0]}
    if mask is not None:
        batch["classification_mask"] = torch.tensor(mask, dtype=torch.bool)
    return batch


def logits_for(predictions, confidence=5.0):
    predictions = torch.tensor(predictions, dtype=torch.long)
    logits = torch.zeros(*predictions.shape, 3)
    return logits.scatter_(-1, predictions.unsqueeze(-1), confidence)


def run_evaluation(labels, predictions, mask=None, cfg=None):
    batch = make_batch(labels, mask)
    output = {"loss": torch.tensor(1.0), "classification_logits": logits_for(predictions)}
    task = ImageTextPretrainTask(cfg=cfg)
    return task.evaluation(FakeModel([output]), [batch], cuda_enabled=False)


def test_paper_metrics_are_reported_every_scored_epoch():
    stats = run_evaluation(
        labels=[[1, 0], [0, 2], [1, 0], [0, 0]],
        predictions=[[1, 0], [0, 2], [1, 0], [0, 0]],
    )
    for key in ("weighted_precision", "weighted_recall", "weighted_f1", "accuracy",
                "auroc_positive_mean", "auroc_negative_mean", "loss"):
        assert key in stats, key
    assert stats["weighted_f1"] == pytest.approx(1.0)


def test_no_binary_or_positive_only_key_is_reported():
    stats = run_evaluation(labels=[[1, 0], [0, 1]], predictions=[[1, 0], [0, 1]])
    for key in stats:
        assert "positive_macro" not in key and "f1_positive" not in key
        assert not key.startswith("sp_") and "presence" not in key


def test_predicting_positive_for_an_uncertain_cell_is_a_miss():
    right = run_evaluation(labels=[[2], [0]], predictions=[[2], [0]])
    folded = run_evaluation(labels=[[2], [0]], predictions=[[1], [0]])
    assert right["weighted_f1"] == pytest.approx(1.0)
    assert folded["weighted_f1"] < 1.0


def test_classification_mask_excludes_unlabelled_rows():
    masked = run_evaluation(labels=[[1], [0]], predictions=[[1], [1]], mask=[True, False])
    unmasked = run_evaluation(labels=[[1], [0]], predictions=[[1], [1]], mask=[True, True])
    assert masked["accuracy"] == pytest.approx(1.0)
    assert unmasked["accuracy"] == pytest.approx(0.5)


def test_shape_mismatch_raises():
    batch = make_batch([[1, 0], [0, 1]])
    output = {"loss": torch.tensor(1.0), "classification_logits": torch.zeros(2, 3, 3)}
    with pytest.raises(ValueError, match=r"\[B, abnormalities, 3\]"):
        ImageTextPretrainTask().evaluation(FakeModel([output]), [batch], cuda_enabled=False)


def test_two_class_logits_are_refused():
    batch = make_batch([[1], [0]])
    output = {"loss": torch.tensor(1.0), "classification_logits": torch.zeros(2, 1, 2)}
    with pytest.raises(ValueError):
        ImageTextPretrainTask().evaluation(FakeModel([output]), [batch], cuda_enabled=False)


@pytest.mark.parametrize("retired", ["uncertain_policy", "report_study_presence", "include_meta_labels"])
def test_a_run_config_with_a_retired_binary_key_is_refused(retired):
    from pretraining.retired_keys import RetiredConfigKey

    with pytest.raises(RetiredConfigKey):
        ImageTextPretrainTask(cfg={retired: "x"})


def test_predictions_are_not_saved_unless_requested(tmp_path):
    task = ImageTextPretrainTask()
    assert not getattr(task, "cfg", None)
    run_evaluation(labels=[[1], [0]], predictions=[[1], [0]])
    assert list(tmp_path.iterdir()) == []
