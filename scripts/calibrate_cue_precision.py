"""Fit selective positive cues on validation predictions, without a model/GPU.

The precision floor is an empirical fitting constraint, not an out-of-sample
guarantee. Infeasible labels are explicitly disabled rather than assigned a
fallback threshold that might emit noisy cues. Outputs contain aggregates only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def fit_selective_thresholds(scores, labels, names, *, precision_floor=0.70, min_predicted=20):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    if scores.ndim != 2 or labels.shape != scores.shape or scores.shape[1] != len(names):
        raise ValueError("scores/labels must be [N, findings], matching names")
    if scores.shape[0] == 0 or len(set(names)) != len(names):
        raise ValueError("calibration needs nonempty samples and unique names")
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("scores must be finite and within [0, 1]")
    # Exported ClassificationPredictions uses -1; ReportDataset uses -100.
    # Neither is a positive report label under study_presence.
    if not np.isin(labels, [-100, -1, 0, 1, 2]).all():
        raise ValueError("expected masked P/N/U labels: -100/-1, 0, 1, 2")
    if not 0 < precision_floor <= 1 or min_predicted < 1:
        raise ValueError("precision_floor must be in (0, 1], min_predicted >= 1")
    thresholds, details = {}, {}
    truth = labels == 1
    for index, name in enumerate(names):
        if name == "No Finding":
            continue
        best = None
        for threshold in np.unique(scores[:, index]):
            predicted = scores[:, index] >= threshold
            count = int(predicted.sum())
            tp = int((predicted & truth[:, index]).sum())
            if count >= min_predicted and tp / count >= precision_floor:
                # Maximum recall; ties prefer fewer false positives, then a
                # higher threshold. Tied scores are never split across groups.
                candidate = (tp, -count, float(threshold))
                if best is None or candidate > best:
                    best = candidate
        thresholds[name] = {
            "positive_enabled": int(best is not None),
            "marginal_positive": best[2] if best else 1.0,
        }
        details[name] = {
            "enabled": best is not None,
            "n_positive": int(truth[:, index].sum()),
            "n_predicted": -best[1] if best else 0,
            "precision": best[0] / -best[1] if best else None,
            "recall": best[0] / int(truth[:, index].sum()) if best else 0.0,
            "reason": "fitted_on_validation" if best else "no_feasible_threshold",
        }
    return thresholds, details


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True, help="Validation NPZ only")
    parser.add_argument("--split", choices=["val", "validation"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--precision-floor", type=float, default=0.70)
    parser.add_argument("--min-predicted", type=int, default=20)
    args = parser.parse_args(argv)
    metadata_path = args.output.with_suffix(".metadata.json")
    if args.output.exists() or metadata_path.exists():
        parser.error("output already exists; choose a fresh path")
    # Most stored arrays are numeric; the existing exporter stores names as an
    # object array. Read only locally trusted project prediction artifacts.
    with np.load(args.predictions, allow_pickle=True) as data:
        if "split" in data:
            stored_splits = {str(x) for x in np.asarray(data["split"]).reshape(-1)}
            if not stored_splits <= {"val", "validation"}:
                parser.error("prediction artifact is not validation data")
        probabilities = data["probabilities"]
        mention = data["mention_probabilities"]
        if probabilities.ndim != 3 or probabilities.shape[-1] != 3 or mention.shape != probabilities.shape[:2]:
            parser.error("expected probabilities [N, C, 3] and mention_probabilities [N, C]")
        thresholds, details = fit_selective_thresholds(
            mention * probabilities[:, :, 1], data["labels"],
            [str(x) for x in data["pathology_names"]],
            precision_floor=args.precision_floor, min_predicted=args.min_predicted,
        )
        n_samples = int(probabilities.shape[0])
    metadata = {
        "fit_split": args.split, "n_samples": n_samples,
        "precision_floor": args.precision_floor, "min_predicted": args.min_predicted,
        "cue_rule": "marginal_positive", "score": "mention_probability * q_positive",
        "truth": "study_presence; only report label 1 is positive",
        "limitation": "Validation fitting constraint, not a precision guarantee on new data or proof of image pathology.",
        "details": details,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(thresholds, handle, indent=2)
        handle.write("\n")
    with metadata_path.open("x", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"n_samples": n_samples, "enabled_labels": sum(v["positive_enabled"] for v in thresholds.values()), "total_labels": len(thresholds)}))


if __name__ == "__main__":
    main()
