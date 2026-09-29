"""The two places where the META-CXR paper itself reduces to binary.

Everything else in this project is three-class (``classification_metrics``).
The paper binarises in exactly two evaluations, and only because the
comparison forces it; they are implemented here, named after the table they
reproduce, and nowhere else:

* **Table 4 -- cross-domain classification on the CheXpert validation set.**
  CheXpert's 200 radiologist-labelled studies are 0/1, and the zero-shot
  baselines it is compared against output binary scores, so the paper converts
  its three-class output with Eq. (21), ``p_final = p1 / (p0 + p1)`` (p0 =
  Negative, p1 = Positive; Uncertain mass is dropped), and reports AUC and F1
  on Atelectasis, Cardiomegaly, Consolidation, Edema and Pleural Effusion.
  ⚠ The paper does not state the F1 threshold; this module uses 0.5 unless told
  otherwise and records the value it used.
* **Table 3 -- Clinical Efficacy (CE) of generated reports.** A CheXpert-style
  labeler is run on each generated report and its reference; a finding counts
  as present when the labeler says Positive (1). Precision, recall and macro F1
  over the 14 findings. The labeler is NOT part of this repository
  (``training/evaluation/clinical.py``); this module scores labels someone
  else extracted.

Numpy only.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from training.evaluation.classification_metrics import PAPER_FIVE_FINDINGS, roc_auc

CHEXPERT_14 = (
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
)

NEGATIVE, POSITIVE = 0, 1


def _prf(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    tp = float(np.sum(y_true & y_pred))
    fp = float(np.sum(~y_true & y_pred))
    fn = float(np.sum(y_true & ~y_pred))
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float("nan")
    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn}


def chexpert_crossdomain(
    probabilities: np.ndarray,
    labels01: np.ndarray,
    pathology_names: Sequence[str],
    *,
    threshold: float = 0.5,
    findings: Sequence[str] = PAPER_FIVE_FINDINGS,
) -> dict:
    """Paper Table 4 from three-class probabilities and CheXpert 0/1 labels.

    ``probabilities`` is ``[N, P, 3]`` (negative, positive, uncertain);
    ``labels01`` is ``[N, P]`` with 0/1 and ``-1`` for missing.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels01 = np.asarray(labels01)
    names = list(pathology_names)
    per_finding = {}
    for finding in findings:
        if finding not in names:
            raise ValueError(f"{finding!r} is not among the prediction columns")
        j = names.index(finding)
        valid = labels01[:, j] >= 0
        p0 = probabilities[valid, j, NEGATIVE]
        p1 = probabilities[valid, j, POSITIVE]
        # Eq. (21). Guard the degenerate all-uncertain row instead of dividing by 0.
        p_final = np.divide(p1, p0 + p1, out=np.full_like(p1, 0.5), where=(p0 + p1) > 0)
        y = labels01[valid, j] == 1
        stats = _prf(y, p_final >= threshold)
        per_finding[finding] = {
            "auc": roc_auc(p_final, y),
            "f1": stats["f1"],
            "n": int(valid.sum()),
            "n_positive": int(y.sum()),
        }
    return {
        "per_finding": per_finding,
        "mean_auc": float(np.nanmean([v["auc"] for v in per_finding.values()])),
        "mean_f1": float(np.nanmean([v["f1"] for v in per_finding.values()])),
        "threshold": threshold,
        "conversion": "p_final = p1 / (p0 + p1)  (paper Eq. 21)",
    }


def clinical_efficacy(
    generated_labels: np.ndarray,
    reference_labels: np.ndarray,
    names: Sequence[str] = CHEXPERT_14,
) -> dict:
    """Paper Table 3 CE metrics from labeler output on generated vs reference.

    Both arrays are ``[N, 14]`` of CheXpert labeler values (1 positive, 0
    negative, -1 uncertain, NaN blank). A finding is "present" when the value
    is 1, as in the CE convention the paper follows.
    """
    generated = np.asarray(generated_labels, dtype=np.float64)
    reference = np.asarray(reference_labels, dtype=np.float64)
    if generated.shape != reference.shape or generated.shape[1] != len(names):
        raise ValueError(
            f"label arrays must both be [N, {len(names)}]; got {generated.shape} "
            f"and {reference.shape}"
        )
    pred = generated == 1
    true = reference == 1
    per_finding = {name: _prf(true[:, j], pred[:, j]) for j, name in enumerate(names)}
    micro = _prf(true.reshape(-1), pred.reshape(-1))
    return {
        "per_finding": per_finding,
        "macro_precision": float(np.nanmean([v["precision"] for v in per_finding.values()])),
        "macro_recall": float(np.nanmean([v["recall"] for v in per_finding.values()])),
        "macro_f1": float(np.nanmean([v["f1"] for v in per_finding.values()])),
        "micro_precision": micro["precision"],
        "micro_recall": micro["recall"],
        "micro_f1": micro["f1"],
        "n_reports": int(generated.shape[0]),
    }
