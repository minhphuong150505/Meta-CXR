#!/usr/bin/env python3
"""Fit the paper's per-class thresholds on the VALIDATION predictions.

META-CXR Sec. V-C / Fig. 11: for every finding and each of its three classes,
take the one-vs-rest ROC of that class's probability and keep the threshold
closest to the top-left corner, ``sqrt((1-TPR)^2 + FPR^2)`` (Eq. 22). The
result is one threshold per (finding, class). They decide which findings the
Stage-2 prompt lists as Positive / Negative / Uncertain; the classification
metrics themselves use argmax.

Fit on validation, never on test.

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
    ThresholdFile,
    fit_class_thresholds,
)

logger = logging.getLogger("calibrate_thresholds")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", required=True, type=Path,
                        help="VALIDATION prediction .npz.")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    predictions = ClassificationPredictions.load(args.predictions)
    split = str(predictions.metadata.get("split", "unknown"))
    if split == "test":
        logger.error("refusing to fit thresholds on the test split")
        return 2
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
