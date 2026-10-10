"""Per-class decision thresholds, as the META-CXR paper fits them (Sec. V-C).

For every finding and every one of its THREE classes the paper draws the
one-vs-rest ROC curve of that class's softmax probability and takes the
threshold closest to the top-left corner (Eq. 22):

    distance = sqrt((1 - TPR)^2 + FPR^2)

Fig. 11 shows the result: one threshold per (finding, class), for Positive,
Negative and Uncertain alike. The thresholds are used when building the LLM
prompt -- "we only consider abnormalities with a confidence score higher than
the relevant threshold for each abnormality" -- not to turn the classifier
into a binary one. The classification metrics themselves use argmax
(``classification_metrics.decide``), as the reference code does.

Fit on VALIDATION only. Nothing here reads the test split.

This replaces the binary positive-threshold calibration that lived here until
2026-09-29 (D-023).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from training.evaluation.schemas import CLASS_NAMES, MISSING, ClassificationPredictions

#: Written into every threshold file; a reader refuses any other value, so a
#: file from the retired binary calibration cannot be mistaken for this one.
THRESHOLD_FORMAT = "meta_cxr_per_class_roc_distance_v1"

#: ``abstain`` in :func:`apply_thresholds`: no class cleared its threshold.
ABSTAIN = -1


class CalibrationError(ValueError):
    """A threshold file or fit request is invalid."""


def roc_distance_threshold(scores: np.ndarray, y_true: np.ndarray) -> float:
    """Eq. 22: the ROC threshold minimising ``sqrt((1-TPR)^2 + FPR^2)``.

    Candidate thresholds are the distinct scores (predict the class when
    ``score >= threshold``). Returns ``nan`` when either side is absent -- a
    class that never occurs has no ROC curve and no threshold.
    """
    scores = np.asarray(scores, dtype=np.float64)
    y_true = np.asarray(y_true).astype(bool)
    n_pos = int(y_true.sum())
    n_neg = int(y_true.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_true = y_true[order]
    tp = np.cumsum(sorted_true)
    fp = np.cumsum(~sorted_true)
    last_of_tie = np.r_[np.where(np.diff(sorted_scores))[0], sorted_scores.size - 1]
    tpr = tp[last_of_tie] / n_pos
    fpr = fp[last_of_tie] / n_neg
    distance = np.sqrt((1.0 - tpr) ** 2 + fpr**2)
    return float(sorted_scores[last_of_tie][int(np.argmin(distance))])


def fit_class_thresholds(
    predictions: ClassificationPredictions,
) -> dict[str, dict[str, float]]:
    """``{finding: {class: threshold}}`` for all findings and all three classes."""
    out: dict[str, dict[str, float]] = {}
    for index, name in enumerate(predictions.pathology_names):
        valid = predictions.labels[:, index] != MISSING
        labels = predictions.labels[valid, index]
        probs = predictions.probabilities[valid, index, :]
        out[name] = {
            cls: roc_distance_threshold(probs[:, c], labels == c)
            for c, cls in enumerate(CLASS_NAMES)
        }
    return out


def apply_thresholds(
    probabilities: np.ndarray,
    thresholds: dict[str, dict[str, float]],
    pathology_names: tuple[str, ...],
) -> np.ndarray:
    """``[N, P]`` class decisions under per-class thresholds, or ``ABSTAIN``.

    A class is a candidate when its probability reaches its threshold. With
    several candidates the one furthest above its own threshold wins; with none
    the finding is left out (``ABSTAIN``), which is how the paper keeps
    low-confidence findings out of the prompt. A class whose threshold is
    undefined (it never occurred on validation) is never a candidate.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    table = np.full((len(pathology_names), len(CLASS_NAMES)), np.inf)
    for p, name in enumerate(pathology_names):
        if name not in thresholds:
            raise CalibrationError(f"no thresholds for finding {name!r}")
        for c, cls in enumerate(CLASS_NAMES):
            value = thresholds[name].get(cls)
            if value is not None and not math.isnan(float(value)):
                table[p, c] = float(value)
    margin = probabilities - table[None, :, :]
    margin[~np.isfinite(margin)] = -np.inf
    decision = margin.argmax(axis=-1)
    none = ~(margin.max(axis=-1) >= 0)
    decision[none] = ABSTAIN
    return decision


def decide_with_thresholds(
    probabilities: np.ndarray,
    thresholds: dict[str, dict[str, float]],
    pathology_names: tuple[str, ...],
) -> np.ndarray:
    """``[N, P]`` class decisions for EVALUATION under per-class thresholds.

    :func:`apply_thresholds`, except that a cell where no class clears its
    threshold falls back to argmax instead of ``ABSTAIN``: a metric needs a
    class for every cell. Supplementary to the paper's argmax protocol, never a
    replacement for it (``scripts/evaluate_stage1.py --thresholds``).
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    decision = apply_thresholds(probabilities, thresholds, pathology_names)
    abstained = decision == ABSTAIN
    decision[abstained] = probabilities.argmax(axis=-1)[abstained]
    return decision


@dataclass
class ThresholdFile:
    thresholds: dict[str, dict[str, float]]
    metadata: dict

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": THRESHOLD_FORMAT,
            "thresholds": {
                name: {
                    cls: (None if math.isnan(v) else float(v)) for cls, v in by_class.items()
                }
                for name, by_class in self.thresholds.items()
            },
            "metadata": self.metadata,
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return path


def load_thresholds(path: str | Path) -> dict[str, dict[str, float]]:
    """Read a per-class threshold file. Refuses any other format."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"threshold file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != THRESHOLD_FORMAT:
        raise CalibrationError(
            f"{path} is not a per-class threshold file (format "
            f"{payload.get('format')!r}, expected {THRESHOLD_FORMAT!r}). Binary "
            "positive-only threshold files were retired on 2026-09-29 (D-023); "
            "refit with scripts/calibrate_thresholds.py on the validation split."
        )
    out: dict[str, dict[str, float]] = {}
    for name, by_class in payload["thresholds"].items():
        missing = set(CLASS_NAMES) - set(by_class)
        if missing:
            raise CalibrationError(f"{path}: {name} lacks classes {sorted(missing)}")
        out[name] = {
            cls: (float("nan") if by_class[cls] is None else float(by_class[cls]))
            for cls in CLASS_NAMES
        }
    return out


# ---------------------------------------------------------------------------
# Ordinal cutpoints on the positive-vs-negative axis (2026-10-01; HEADLINE
# decision rule since 2026-10-10, user decision)
# ---------------------------------------------------------------------------
#
# NOT the paper's protocol (the paper reports argmax). Uncertain cases sit
# BETWEEN negatives and positives on s = p_pos / (p_pos + p_neg)
# (docs/handoff/PLAN-2026-09-30-...), so a finding can be decided by two
# cutpoints on s: Negative below t1, Uncertain in [t1, t2), Positive at or above
# t2. One cutpoint (t1 == t2) never calls Uncertain. Fit on VALIDATION only.
# Since 2026-10-10 this project's headline Stage-1 numbers use the two-cutpoint
# rule (scripts/evaluate_stage1.py --cutpoints); argmax is still reported
# beside them as the paper-protocol reference.

CUTPOINT_OBJECTIVES = ("weighted_f1", "macro_recall")


def severity_scores(probabilities: np.ndarray) -> np.ndarray:
    """``p_pos / (p_pos + p_neg)`` per cell -- the paper's Eq. 21 score."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    pos, neg = probabilities[..., 1], probabilities[..., 0]
    return pos / np.maximum(pos + neg, 1e-12)


def decide_with_cutpoints(severity: np.ndarray, t1: float, t2: float) -> np.ndarray:
    """0 below ``t1``, 2 (Uncertain) in ``[t1, t2)``, 1 at or above ``t2``."""
    if t2 < t1:
        raise CalibrationError(f"cutpoints must satisfy t1 <= t2, got {t1} > {t2}")
    severity = np.asarray(severity, dtype=np.float64)
    return np.where(severity >= t2, 1, np.where(severity >= t1, 2, 0))


def fit_severity_cutpoints(
    predictions: ClassificationPredictions,
    *,
    objective: str = "weighted_f1",
    allow_uncertain: bool = True,
    grid_size: int = 121,
    min_uncertain: int = 5,
) -> dict[str, tuple[float, float]]:
    """``{finding: (t1, t2)}`` maximising ``objective`` per finding.

    Candidates are ``grid_size`` quantiles of the finding's own severity scores.
    A finding with fewer than ``min_uncertain`` Uncertain cells, or
    ``allow_uncertain=False``, gets a single cutpoint (``t1 == t2``).
    Ties keep the first (lowest) candidate, so the result is deterministic.
    """
    from training.evaluation.classification_metrics import macro_recall, weighted_prf

    if objective not in CUTPOINT_OBJECTIVES:
        raise CalibrationError(f"objective must be one of {CUTPOINT_OBJECTIVES}")
    severity = severity_scores(predictions.probabilities)
    out: dict[str, tuple[float, float]] = {}
    for index, name in enumerate(predictions.pathology_names):
        valid = predictions.labels[:, index] != MISSING
        y = predictions.labels[valid, index].astype(int)
        s = severity[valid, index]
        if y.size == 0:
            out[name] = (0.5, 0.5)
            continue
        grid = np.unique(np.quantile(s, np.linspace(0.0, 1.0, grid_size)))
        two = allow_uncertain and int((y == 2).sum()) >= min_uncertain
        best, arg = -np.inf, (float(grid[0]), float(grid[0]))
        for i, t1 in enumerate(grid):
            for t2 in (grid[i:] if two else (t1,)):
                d = decide_with_cutpoints(s, t1, t2)
                value = (weighted_prf(y, d)["f1"] if objective == "weighted_f1"
                         else macro_recall(y, d))
                if value > best + 1e-12:
                    best, arg = value, (float(t1), float(t2))
        out[name] = arg
    return out


def apply_cutpoints(
    probabilities: np.ndarray,
    cutpoints: dict[str, tuple[float, float]],
    pathology_names: tuple[str, ...],
) -> np.ndarray:
    """``[N, P]`` decisions from per-finding cutpoints on the severity score."""
    severity = severity_scores(probabilities)
    decisions = np.empty(severity.shape, dtype=int)
    for index, name in enumerate(pathology_names):
        if name not in cutpoints:
            raise CalibrationError(f"no cutpoints for finding {name!r}")
        t1, t2 = cutpoints[name]
        decisions[:, index] = decide_with_cutpoints(severity[:, index], t1, t2)
    return decisions


#: Written into every cutpoint file; :func:`load_cutpoints` refuses any other
#: value, so an Eq. 22 threshold file can never be read as cutpoints.
CUTPOINT_FORMAT = "meta_cxr_severity_cutpoints_v1"


@dataclass
class CutpointFile:
    """Per-finding ``(t1, t2)`` on ``p_pos / (p_pos + p_neg)``, fitted on validation."""

    cutpoints: dict[str, tuple[float, float]]
    metadata: dict

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": CUTPOINT_FORMAT,
            "cutpoints": {name: [float(t1), float(t2)]
                          for name, (t1, t2) in self.cutpoints.items()},
            "metadata": self.metadata,
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
        return path


def load_cutpoint_file(path: str | Path) -> tuple[dict[str, tuple[float, float]], dict]:
    """``(cutpoints, metadata)`` from a cutpoint file. Refuses any other format."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"cutpoint file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != CUTPOINT_FORMAT:
        raise CalibrationError(
            f"{path} is not a cutpoint file (format {payload.get('format')!r}, "
            f"expected {CUTPOINT_FORMAT!r}); fit one with "
            "scripts/calibrate_thresholds.py --rule cutpoints"
        )
    out: dict[str, tuple[float, float]] = {}
    for name, pair in payload["cutpoints"].items():
        if len(pair) != 2:
            raise CalibrationError(f"{path}: {name} needs [t1, t2], got {pair!r}")
        t1, t2 = float(pair[0]), float(pair[1])
        if t2 < t1:
            raise CalibrationError(f"{path}: {name} has t1 {t1} > t2 {t2}")
        out[name] = (t1, t2)
    return out, dict(payload.get("metadata", {}))


def load_cutpoints(path: str | Path) -> dict[str, tuple[float, float]]:
    return load_cutpoint_file(path)[0]
