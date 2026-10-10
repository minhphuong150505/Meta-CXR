#!/usr/bin/env python3
"""Stage-1 analysis: decide by cutpoints on p_pos / (p_pos + p_neg), vs argmax.

NOT the paper's protocol -- the paper decides by argmax. Since 2026-10-10 the
two-cutpoint rule is this project's HEADLINE (user decision; the fitted file is
committed under configs/stage1_cutpoints/ and scored by
``scripts/evaluate_stage1.py --cutpoints``); this script remains the paired
comparison of both rules against argmax. Uncertain cases sit between negatives and positives on the score
s = p_pos / (p_pos + p_neg), so this script fits, on VALIDATION, per finding:

* one cutpoint (Negative / Positive, never Uncertain), and
* two cutpoints (Negative / Uncertain band / Positive),

each maximising per-finding weighted F1, then scores the TEST predictions with
argmax and both rules and reports paired-bootstrap differences against argmax.
AUROC does not depend on the decision rule and is not reported here.

    python scripts/evaluate_stage1_cutpoints.py \\
        --val <run>/result/val_predictions_epoch_best.npz \\
        --test <run>/result/test_predictions_epoch_best.npz \\
        --output-dir <private dir>/stage1_cutpoints

Exit codes: 0 success, 2 bad input (missing file, test used for fitting,
mismatched findings).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from training.evaluation.classification_metrics import evaluate_classification  # noqa: E402
from training.evaluation.schemas import MISSING, ClassificationPredictions  # noqa: E402
from training.evaluation.threshold_calibration import (  # noqa: E402
    apply_cutpoints,
    fit_severity_cutpoints,
)

logger = logging.getLogger("evaluate_stage1_cutpoints")

METRICS = (
    "weighted_precision",
    "weighted_recall",
    "weighted_f1",
    "mean_weighted_f1_5",
    "macro_recall",
)


def _aggregates(predictions: ClassificationPredictions, decisions: np.ndarray, idx=None):
    if idx is None:
        idx = np.arange(predictions.num_samples)
    one_hot = np.eye(3)[decisions[idx]]
    sub = ClassificationPredictions(
        labels=predictions.labels[idx],
        probabilities=one_hot,
        pathology_names=predictions.pathology_names,
        sample_keys=predictions.sample_keys[idx],
    )
    return evaluate_classification(sub).aggregates


def _uncertain_counts(labels: np.ndarray, decisions: np.ndarray) -> dict[str, float]:
    valid = labels != MISSING
    y, d = labels[valid], decisions[valid]
    predicted, true = int((d == 2).sum()), int((y == 2).sum())
    hits = int(((d == 2) & (y == 2)).sum())
    return {
        "uncertain_predicted": predicted,
        "uncertain_true": true,
        "uncertain_recall": hits / true if true else float("nan"),
        "uncertain_precision": hits / predicted if predicted else float("nan"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--val", required=True, type=Path, help="validation .npz (fitting)")
    parser.add_argument("--test", required=True, type=Path, help="test .npz (scoring)")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=16)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    for path in (args.val, args.test):
        if not path.is_file():
            logger.error("prediction file not found: %s", path)
            return 2
    val = ClassificationPredictions.load(args.val)
    test = ClassificationPredictions.load(args.test)
    if str(val.metadata.get("split", "")) == "test":
        logger.error("refusing to fit cutpoints on the test split")
        return 2
    if tuple(val.pathology_names) != tuple(test.pathology_names):
        logger.error("val and test name different findings")
        return 2

    rules: dict[str, dict] = {"argmax (paper)": {"decisions": test.probabilities.argmax(-1)}}
    for tag, allow in (("one cutpoint", False), ("two cutpoints", True)):
        cut = fit_severity_cutpoints(val, objective="weighted_f1", allow_uncertain=allow)
        rules[tag] = {
            "decisions": apply_cutpoints(test.probabilities, cut, tuple(test.pathology_names)),
            "cutpoints": {k: list(v) for k, v in cut.items()},
        }

    rng = np.random.default_rng(args.seed)
    n = test.num_samples
    base = rules["argmax (paper)"]["decisions"]
    draws = [rng.integers(0, n, n) for _ in range(args.bootstrap_samples)]
    report: dict = {
        "note": "not the paper's argmax protocol; cutpoints fitted on validation, "
                "maximising per-finding weighted F1 (two cutpoints = project "
                "headline since 2026-10-10)",
        "val": str(args.val), "test": str(args.test), "num_test": n,
        "bootstrap_samples": args.bootstrap_samples, "seed": args.seed, "rules": {},
    }
    base_draws = [_aggregates(test, base, idx) for idx in draws]
    for tag, rule in rules.items():
        decisions = rule["decisions"]
        entry = {"metrics": _aggregates(test, decisions),
                 **_uncertain_counts(test.labels, decisions)}
        entry["metrics"] = {k: entry["metrics"][k] for k in METRICS}
        if "cutpoints" in rule:
            entry["cutpoints"] = rule["cutpoints"]
            deltas = {}
            for k in METRICS:
                diff = [_aggregates(test, decisions, idx)[k] - b[k]
                        for idx, b in zip(draws, base_draws, strict=True)]
                lo, hi = np.percentile(diff, [2.5, 97.5])
                deltas[k] = {"delta": entry["metrics"][k] - report["rules"]["argmax (paper)"]["metrics"][k],
                             "ci95": [float(lo), float(hi)]}
            entry["minus_argmax"] = deltas
        report["rules"][tag] = entry

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "cutpoints_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print("\nCutpoints vs argmax -- argmax is the paper's protocol; two cutpoints is "
          "the project headline since 2026-10-10:")
    for tag, entry in report["rules"].items():
        m = entry["metrics"]
        print(f"  {tag:15s} wF1 {m['weighted_f1']:.4f}  F1_5 {m['mean_weighted_f1_5']:.4f}  "
              f"macro_recall {m['macro_recall']:.4f}  Uncertain predicted "
              f"{entry['uncertain_predicted']} / true {entry['uncertain_true']}")
        for k, d in entry.get("minus_argmax", {}).items():
            print(f"      {k:20s} {d['delta']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]")
    print(f"  report: {args.output_dir / 'cutpoints_report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
