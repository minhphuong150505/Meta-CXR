"""Stage-1 classification metrics, exactly as the META-CXR paper reports them.

Every finding is a THREE-class problem -- Negative (0), Positive (1),
Uncertain (2) -- and it is evaluated as one. There is no binary "present /
absent" framing anywhere in this module, no positive-only F1 headline, no
uncertain-folding policy and no mention gate. Those were removed on 2026-09-29
(D-023) because they are not what the paper measures; see
``pretraining/retired_keys.py`` and CLAUDE.md, "Three classes, never binary".

What the paper reports, and where each number comes from here
--------------------------------------------------------------
* **Precision / Recall / F1 (0.87 / 0.78 / 0.73, Sec. IV-B-2a, Fig. 10).**
  The reference code (``META-CXR/mhcac/utils.py:compute_metrics_for_tasks``)
  takes the ARGMAX of the three-class softmax per finding, computes sklearn's
  ``average='weighted'`` precision / recall / F1 with ``zero_division=1`` per
  finding, and averages the 14 findings with equal weight. That is
  :attr:`ClassificationReport.aggregates` ``weighted_precision`` /
  ``weighted_recall`` / ``weighted_f1`` / ``accuracy``.
  ⚠ The reference averaged per-BATCH scores and then over batches; this module
  computes each finding over the whole split, which is the exact quantity the
  batch mean approximates. Batch composition therefore explains small gaps.
* **Mean F1 over 5 common findings (Tables 5 and 7).** The same per-finding
  weighted F1, averaged over Atelectasis, Cardiomegaly, Consolidation, Edema
  and Pleural Effusion: ``mean_weighted_f1_5``.
* **AUC-ROC per class (Fig. 5).** For every finding and every class, a
  one-vs-rest ROC of that class's softmax probability ("Positive class vs
  Rest", "Negative class vs Rest", "Uncertain class vs Rest"), all 14 findings
  including No Finding: per-finding ``auroc`` dict, summarised as
  ``auroc_negative_mean`` / ``auroc_positive_mean`` / ``auroc_uncertain_mean``.

Not in the paper
----------------
* **Macro recall over the three classes (``macro_recall``).** Per finding, the
  unweighted mean of the Negative / Positive / Uncertain recalls over the
  classes present in the ground truth -- sklearn's ``balanced_accuracy_score``
  -- averaged over the 14 findings. Added 2026-10-01 as the Stage-1
  checkpoint-selection metric at the user's request. ``weighted_recall`` cannot
  serve: support-weighted recall is identically ``accuracy``, so it rewards the
  majority Negative class; here a missed Positive or Uncertain cell costs as
  much as a missed Negative one. Still three classes, still argmax.

Numpy only, so the core runs on the CPU development box. The weighted P/R/F1
follow sklearn's semantics exactly and are pinned against sklearn on the host
(``tests/test_classification_metrics.py``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from training.evaluation.schemas import (
    CLASS_NAMES,
    MISSING,
    ClassificationPredictions,
    SchemaError,
)

logger = logging.getLogger(__name__)

NUM_CLASSES = 3

#: The five findings the paper's Tables 5 and 7 average over.
PAPER_FIVE_FINDINGS = (
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
)

#: ``zero_division`` the reference code passes to sklearn.
PAPER_ZERO_DIVISION = 1.0


def roc_auc(scores: np.ndarray, y_true: np.ndarray) -> float:
    """AUROC via the rank (Mann-Whitney U) identity, with tie correction.

    Returns ``nan`` when either side is absent -- AUROC is undefined then, and
    returning 0.5 would look like a real measurement of a coin flip.
    """
    scores = np.asarray(scores, dtype=np.float64)
    y_true = np.asarray(y_true).astype(bool)
    n_pos = int(np.sum(y_true))
    n_neg = int(y_true.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.size, dtype=np.float64)
    sorted_scores = scores[order]

    start = 0
    while start < sorted_scores.size:
        stop = start + 1
        while stop < sorted_scores.size and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop

    rank_sum = float(np.sum(ranks[y_true]))
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def average_precision(scores: np.ndarray, y_true: np.ndarray) -> float:
    """Area under the PR curve as step-wise average precision (sklearn's AP)."""
    scores = np.asarray(scores, dtype=np.float64)
    y_true = np.asarray(y_true).astype(bool)
    n_pos = int(np.sum(y_true))
    if n_pos == 0:
        return float("nan")

    order = np.argsort(-scores, kind="mergesort")
    sorted_true = y_true[order]
    sorted_scores = scores[order]
    tp = np.cumsum(sorted_true)
    fp = np.cumsum(~sorted_true)
    distinct = np.where(np.diff(sorted_scores))[0]
    keep = np.r_[distinct, sorted_scores.size - 1]
    tp = tp[keep]
    fp = fp[keep]
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / n_pos
    previous_recall = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - previous_recall) * precision))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """``[3, 3]`` counts, rows = true class, columns = predicted class."""
    matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    np.add.at(matrix, (np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)), 1)
    return matrix


def per_class_prf(
    matrix: np.ndarray, zero_division: float = PAPER_ZERO_DIVISION
) -> dict[str, np.ndarray]:
    """Per-class precision, recall, F1 and support, with sklearn's semantics.

    Precision of a class never predicted, or recall of a class with no support,
    takes ``zero_division`` -- that is what sklearn does and what the reference
    code relied on. F1 is ``2 tp / (support + predicted)`` (sklearn >= 1.3),
    ``zero_division`` only when both are zero.
    """
    tp = np.diag(matrix).astype(np.float64)
    support = matrix.sum(axis=1).astype(np.float64)
    predicted = matrix.sum(axis=0).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(predicted > 0, tp / predicted, zero_division)
        recall = np.where(support > 0, tp / support, zero_division)
        denom = support + predicted
        f1 = np.where(denom > 0, 2.0 * tp / denom, zero_division)
    return {"precision": precision, "recall": recall, "f1": f1, "support": support}


def weighted_prf(
    y_true: np.ndarray, y_pred: np.ndarray, zero_division: float = PAPER_ZERO_DIVISION
) -> dict[str, float]:
    """sklearn ``average='weighted'`` precision / recall / F1, plus accuracy.

    The average runs over the classes present in ``y_true`` OR ``y_pred`` (as
    sklearn's ``unique_labels`` does), weighted by true support, so a class that
    is only ever predicted contributes weight zero.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if y_true.size == 0:
        nan = float("nan")
        return {"precision": nan, "recall": nan, "f1": nan, "accuracy": nan}
    matrix = confusion_matrix(y_true, y_pred)
    stats = per_class_prf(matrix, zero_division)
    present = np.union1d(np.unique(y_true), np.unique(y_pred))
    weights = stats["support"][present]
    total = weights.sum()
    out = {
        name: float(np.sum(stats[name][present] * weights) / total)
        for name in ("precision", "recall", "f1")
    }
    out["accuracy"] = float(np.mean(y_true == y_pred))
    return out


def macro_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Unweighted mean recall over the classes present in ``y_true``.

    sklearn's ``balanced_accuracy_score``: a class with no true support is left
    out rather than scored by ``zero_division``, so predicting a class that
    never occurs cannot raise it.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if y_true.size == 0:
        return float("nan")
    present = np.unique(y_true)
    return float(np.mean([np.mean(y_pred[y_true == c] == c) for c in present]))


@dataclass
class PathologyMetrics:
    """Three-class metrics for one finding."""

    name: str
    support: dict[str, int]
    weighted_precision: float
    weighted_recall: float
    weighted_f1: float
    accuracy: float
    macro_recall: float
    per_class: dict[str, dict[str, float]]
    auroc: dict[str, float]
    auprc: dict[str, float]
    confusion: list[list[int]]

    def to_dict(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "pathology": self.name,
            "weighted_precision": self.weighted_precision,
            "weighted_recall": self.weighted_recall,
            "weighted_f1": self.weighted_f1,
            "accuracy": self.accuracy,
            "macro_recall": self.macro_recall,
        }
        for cls in CLASS_NAMES:
            row[f"n_{cls}"] = self.support.get(cls, 0)
            row[f"auroc_{cls}"] = self.auroc.get(cls, float("nan"))
            row[f"auprc_{cls}"] = self.auprc.get(cls, float("nan"))
            for metric in ("precision", "recall", "f1"):
                row[f"{metric}_{cls}"] = self.per_class.get(cls, {}).get(
                    metric, float("nan")
                )
        return row


@dataclass
class ClassificationReport:
    per_pathology: list[PathologyMetrics]
    aggregates: dict[str, float]
    settings: dict[str, Any] = field(default_factory=dict)

    def pathology(self, name: str) -> PathologyMetrics:
        for metrics in self.per_pathology:
            if metrics.name == name:
                return metrics
        raise KeyError(name)


def _nanmean(values) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0 or np.all(np.isnan(array)):
        return float("nan")
    return float(np.nanmean(array))


def decide(probabilities: np.ndarray) -> np.ndarray:
    """The paper's decision: argmax over the three class probabilities."""
    return np.asarray(probabilities).argmax(axis=-1)


def evaluate_classification(
    predictions: ClassificationPredictions,
    *,
    zero_division: float = PAPER_ZERO_DIVISION,
    thresholds: dict[str, dict[str, float]] | None = None,
) -> ClassificationReport:
    """Score one split with the paper's three-class protocol.

    Cells labelled ``MISSING`` (-1: a study with no CheXpert information) are
    skipped per finding; nothing else is dropped or merged.

    ``thresholds`` (per finding and class, fitted on VALIDATION by
    ``scripts/calibrate_thresholds.py``) replaces argmax with
    ``threshold_calibration.decide_with_thresholds``. That is NOT the paper's
    protocol -- the paper uses these thresholds only for the Stage-2 prompt --
    and is reported beside the argmax numbers, never instead of them. AUROC and
    AUPRC do not depend on the decision rule and are identical either way.
    """
    if predictions.num_classes != NUM_CLASSES:
        raise SchemaError(
            f"the paper's protocol is three-class; got {predictions.num_classes} "
            "classes"
        )
    labels = predictions.labels
    probabilities = predictions.probabilities
    if thresholds is None:
        predicted = decide(probabilities)
    else:
        from training.evaluation.threshold_calibration import decide_with_thresholds

        predicted = decide_with_thresholds(
            probabilities, thresholds, tuple(predictions.pathology_names)
        )

    per_pathology: list[PathologyMetrics] = []
    for index, name in enumerate(predictions.pathology_names):
        valid = labels[:, index] != MISSING
        y_true = labels[valid, index].astype(int)
        y_pred = predicted[valid, index].astype(int)
        probs = probabilities[valid, index, :]

        weighted = weighted_prf(y_true, y_pred, zero_division)
        matrix = confusion_matrix(y_true, y_pred) if y_true.size else np.zeros(
            (NUM_CLASSES, NUM_CLASSES), dtype=np.int64
        )
        stats = per_class_prf(matrix, zero_division)
        per_class = {
            cls: {
                "precision": float(stats["precision"][c]),
                "recall": float(stats["recall"][c]),
                "f1": float(stats["f1"][c]),
            }
            for c, cls in enumerate(CLASS_NAMES)
        }
        auroc = {
            cls: roc_auc(probs[:, c], y_true == c) for c, cls in enumerate(CLASS_NAMES)
        }
        auprc = {
            cls: average_precision(probs[:, c], y_true == c)
            for c, cls in enumerate(CLASS_NAMES)
        }
        per_pathology.append(
            PathologyMetrics(
                name=name,
                support={cls: int(stats["support"][c]) for c, cls in enumerate(CLASS_NAMES)},
                weighted_precision=weighted["precision"],
                weighted_recall=weighted["recall"],
                weighted_f1=weighted["f1"],
                accuracy=weighted["accuracy"],
                macro_recall=macro_recall(y_true, y_pred),
                per_class=per_class,
                auroc=auroc,
                auprc=auprc,
                confusion=matrix.tolist(),
            )
        )

    aggregates = {
        # Paper Sec. IV-B-2a / Fig. 10: equal-weight mean over the 14 findings.
        "weighted_precision": _nanmean(m.weighted_precision for m in per_pathology),
        "weighted_recall": _nanmean(m.weighted_recall for m in per_pathology),
        "weighted_f1": _nanmean(m.weighted_f1 for m in per_pathology),
        "accuracy": _nanmean(m.accuracy for m in per_pathology),
        # Not in the paper: three-class balanced recall, the selection metric.
        "macro_recall": _nanmean(m.macro_recall for m in per_pathology),
        # Paper Tables 5 and 7.
        "mean_weighted_f1_5": _nanmean(
            m.weighted_f1 for m in per_pathology if m.name in PAPER_FIVE_FINDINGS
        ),
    }
    # Paper Fig. 5: one-vs-rest AUC per class, summarised over findings.
    for cls in CLASS_NAMES:
        aggregates[f"auroc_{cls}_mean"] = _nanmean(m.auroc[cls] for m in per_pathology)
        aggregates[f"auprc_{cls}_mean"] = _nanmean(m.auprc[cls] for m in per_pathology)
    aggregates["auroc_mean"] = _nanmean(
        m.auroc[cls] for m in per_pathology for cls in CLASS_NAMES
    )

    missing_five = [n for n in PAPER_FIVE_FINDINGS if n not in predictions.pathology_names]
    if missing_five:
        logger.warning("mean_weighted_f1_5 is missing %s", ", ".join(missing_five))

    return ClassificationReport(
        per_pathology=per_pathology,
        aggregates=aggregates,
        settings={
            "protocol": "META-CXR paper: three-class argmax, per-finding "
            "sklearn-weighted P/R/F1 (zero_division=1), mean over findings; "
            "one-vs-rest AUROC per class",
            "decision": "argmax" if thresholds is None else (
                "per-class Eq. 22 thresholds (largest margin above threshold; "
                "argmax when no class clears) -- NOT the paper's protocol"
            ),
            "zero_division": zero_division,
            "five_findings": list(PAPER_FIVE_FINDINGS),
            "num_findings": len(per_pathology),
        },
    )


#: Aggregate names the evaluator, the training hook and the configs may name.
AGGREGATE_METRICS = (
    "weighted_precision",
    "weighted_recall",
    "weighted_f1",
    "accuracy",
    "macro_recall",
    "mean_weighted_f1_5",
    "auroc_negative_mean",
    "auroc_positive_mean",
    "auroc_uncertain_mean",
    "auprc_negative_mean",
    "auprc_positive_mean",
    "auprc_uncertain_mean",
    "auroc_mean",
)
