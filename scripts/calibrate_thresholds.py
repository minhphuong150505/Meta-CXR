#!/usr/bin/env python3
"""Fit the paper's per-class thresholds on the VALIDATION predictions.

META-CXR Sec. V-C / Fig. 11: for every finding and each of its three classes,
take the one-vs-rest ROC of that class's probability and keep the threshold
closest to the top-left corner, ``sqrt((1-TPR)^2 + FPR^2)`` (Eq. 22). The
result is one threshold per (finding, class). They decide which findings the
Stage-2 prompt lists as Positive / Negative / Uncertain; the classification
metrics themselves use argmax.

Fit on validation, never on test.

``--rule cutpoints`` instead fits this project's HEADLINE decision rule (since
2026-10-10): per finding, two cutpoints ``t1 <= t2`` on
``s = p_pos / (p_pos + p_neg)`` maximising that finding's weighted F1 --
Negative below t1, Uncertain in [t1, t2), Positive at or above t2.
``--one-cutpoint`` forces t1 == t2 (never Uncertain). Score with
``scripts/evaluate_stage1.py --cutpoints <file>``.

    python scripts/calibrate_thresholds.py --rule cutpoints \\
        --predictions <run>/result/val_predictions_epoch_best.npz \\
        --output configs/stage1_cutpoints/<run>.json

    python scripts/calibrate_thresholds.py \\
        --predictions <run>/result/val_predictions_epoch_best.npz \\
        --output <private dir>/class_thresholds.json
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from training.evaluation.schemas import CLASS_NAMES, ClassificationPredictions  # noqa: E402
from training.evaluation.threshold_calibration import (  # noqa: E402
    CutpointFile,
    ThresholdFile,
    fit_class_thresholds,
    fit_severity_cutpoints,
)

logger = logging.getLogger("calibrate_thresholds")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", required=True, type=Path,
                        help="VALIDATION prediction .npz.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rule", choices=("roc_distance", "cutpoints"),
                        default="roc_distance",
                        help="roc_distance: the paper's Eq. 22 per-class thresholds "
                        "(default); cutpoints: the project's headline rule")
    parser.add_argument("--one-cutpoint", action="store_true",
                        help="with --rule cutpoints: t1 == t2, never Uncertain")
    parser.add_argument("--checkpoint", default="unknown",
                        help="recorded in the file's metadata")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    predictions = ClassificationPredictions.load(args.predictions)
    split = str(predictions.metadata.get("split", "unknown"))
    if split == "test":
        logger.error("refusing to fit thresholds on the test split")
        return 2
    if args.one_cutpoint and args.rule != "cutpoints":
        parser.error("--one-cutpoint needs --rule cutpoints")
    if args.rule == "cutpoints":
        cut = fit_severity_cutpoints(predictions, objective="weighted_f1",
                                     allow_uncertain=not args.one_cutpoint)
        CutpointFile(
            cutpoints=cut,
            metadata={
                "split": split,
                "num_samples": predictions.num_samples,
                "checkpoint": args.checkpoint,
                "objective": "per-finding sklearn-weighted F1 (three classes)",
                "rule": ("one cutpoint (t1 == t2, never Uncertain)" if args.one_cutpoint
                         else "two cutpoints: N < t1 <= U < t2 <= P"),
                "score": "p_pos / (p_pos + p_neg)",
            },
        ).save(args.output)
        print(f"{'finding':28s} {'t1':>8s} {'t2':>8s}")
        for name, (t1, t2) in cut.items():
            print(f"{name:28s} {t1:8.4f} {t2:8.4f}")
        print(f"\nwrote {args.output}")
        return 0
    thresholds = fit_class_thresholds(predictions)
    ThresholdFile(
        thresholds=thresholds,
        metadata={
            "source": str(args.predictions),
            "split": split,
            "num_samples": predictions.num_samples,
            "rule": "per (finding, class): argmin sqrt((1-TPR)^2 + FPR^2) on the "
                    "one-vs-rest ROC (META-CXR Eq. 22)",
        },
    ).save(args.output)

    print(f"{'finding':28s} " + " ".join(f"{c:>10s}" for c in CLASS_NAMES))
    for name, by_class in thresholds.items():
        print(f"{name:28s} " + " ".join(f"{by_class[c]:10.4f}" for c in CLASS_NAMES))
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
