#!/usr/bin/env python3
"""META-CXR Table 4: cross-domain classification on the CheXpert validation set.

Scores a prediction ``.npz`` whose labels are CheXpert's 0/1 radiologist
labels (``-1`` = missing) with the paper's Eq. (21) conversion
``p_final = p1 / (p0 + p1)``, and reports AUC and F1 on the paper's five
findings. See ``training/evaluation/paper_protocol.py``.

⚠ Producing that ``.npz`` needs Stage-1 inference on the CheXpert validation
images, and this repository has no CheXpert loader wired into the Stage-1
runner (the old checkout's ``CheXpertDataset`` was not carried over). Only the
scoring is implemented and tested here.

    python scripts/evaluate_chexpert_crossdomain.py \\
        --predictions <private>/chexpert_val_predictions.npz --output <private>/table4.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from training.evaluation.paper_protocol import chexpert_crossdomain  # noqa: E402
from training.evaluation.schemas import ClassificationPredictions  # noqa: E402

#: Paper Table 4, META-CXR row: (AUC, F1) per finding and the means.
PAPER_TABLE4 = {
    "Atelectasis": (0.792, 0.623),
    "Cardiomegaly": (0.842, 0.674),
    "Consolidation": (0.790, 0.792),
    "Edema": (0.820, 0.721),
    "Pleural Effusion": (0.878, 0.689),
    "mean": (0.824, 0.699),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="F1 threshold on p_final (the paper does not state one).")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    predictions = ClassificationPredictions.load(args.predictions)
    bad = set(predictions.labels.reshape(-1).tolist()) - {-1, 0, 1}
    if bad:
        print(f"labels must be CheXpert 0/1 (-1 missing); found {sorted(bad)}", file=sys.stderr)
        return 2
    result = chexpert_crossdomain(
        predictions.probabilities, predictions.labels, predictions.pathology_names,
        threshold=args.threshold,
    )
    print(f"{'finding':20s} {'AUC':>7s} {'paper':>7s} {'F1':>7s} {'paper':>7s}   n")
    for name, row in result["per_finding"].items():
        pa, pf = PAPER_TABLE4[name]
        print(f"{name:20s} {row['auc']:7.3f} {pa:7.3f} {row['f1']:7.3f} {pf:7.3f}   {row['n']}")
    pa, pf = PAPER_TABLE4["mean"]
    print(f"{'mean':20s} {result['mean_auc']:7.3f} {pa:7.3f} {result['mean_f1']:7.3f} {pf:7.3f}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({**result, "paper_table4": PAPER_TABLE4},
                                          indent=2, default=float), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
