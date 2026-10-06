#!/usr/bin/env python3
"""Paired comparison of two Stage-1 prediction files on the SAME studies.

Paper protocol (three-class argmax, D-023). Both files must hold the same
studies in the same order -- same ``sample_keys``, same ``labels`` -- so every
bootstrap resample scores both arms on identical studies. Built for the
multi-view ablation (docs/handoff/PLAN-2026-10-06-multi-aux-views.md), where
the arms differ only in ``model.data.max_aux_views``.

Besides the whole split it scores subgroups by how many images the model saw,
read from ``num_views`` in ``--subgroups-from`` (default: the candidate, which
is the arm that can see more views). A change confined to ~5% of studies is
invisible in the overall numbers, so the subgroup rows are the ones to read.

    python scripts/compare_stage1_predictions.py \\
        --baseline <dir_aux1>/test_predictions_epoch_provided.npz \\
        --candidate <dir_aux3>/test_predictions_epoch_provided.npz \\
        --output <private dir>/aux3_vs_aux1.json

Exit codes: 0 success, 2 bad input (missing file, misaligned studies, no
``num_views`` in the subgroup source).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from training.evaluation.classification_metrics import evaluate_classification  # noqa: E402
from training.evaluation.schemas import ClassificationPredictions  # noqa: E402

METRICS = (
    "weighted_precision",
    "weighted_recall",
    "weighted_f1",
    "mean_weighted_f1_5",
    "macro_recall",
    "auroc_positive_mean",
    "auroc_negative_mean",
    "auroc_uncertain_mean",
    "auroc_mean",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--subgroups-from", choices=("candidate", "baseline"), default="candidate"
    )
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=16)
    return parser.parse_args(argv)


def subset(p: ClassificationPredictions, idx: np.ndarray) -> ClassificationPredictions:
    return ClassificationPredictions(
        labels=p.labels[idx],
        probabilities=p.probabilities[idx],
        pathology_names=p.pathology_names,
        sample_keys=p.sample_keys[idx],
    )


def aggregates(p: ClassificationPredictions) -> np.ndarray:
    agg = evaluate_classification(p).aggregates
    return np.array([agg[m] for m in METRICS], dtype=float)


def view_groups(num_views: np.ndarray) -> dict[str, np.ndarray]:
    counts = np.asarray(num_views, dtype=int)
    groups = {
        "all": np.arange(counts.size),
        "views_1": np.where(counts == 1)[0],
        "views_2": np.where(counts == 2)[0],
        "views_3plus": np.where(counts >= 3)[0],
        "views_2plus": np.where(counts >= 2)[0],
    }
    return {name: idx for name, idx in groups.items() if idx.size}


def changed_fraction(a: ClassificationPredictions, b: ClassificationPredictions, idx):
    """Share of (study, finding) cells whose argmax differs between the arms."""
    pa = a.probabilities[idx].argmax(-1)
    pb = b.probabilities[idx].argmax(-1)
    return float((pa != pb).mean()) if pa.size else float("nan")


def paired(base, cand, idx, rng, resamples):
    b0, c0 = aggregates(subset(base, idx)), aggregates(subset(cand, idx))
    deltas = np.empty((resamples, len(METRICS)))
    for r in range(resamples):
        draw = idx[rng.integers(0, idx.size, idx.size)]
        deltas[r] = aggregates(subset(cand, draw)) - aggregates(subset(base, draw))
    out = {}
    for i, name in enumerate(METRICS):
        lo, hi = np.nanpercentile(deltas[:, i], [2.5, 97.5])
        out[name] = {
            "baseline": float(b0[i]),
            "candidate": float(c0[i]),
            "delta": float(c0[i] - b0[i]),
            "lo": float(lo),
            "hi": float(hi),
            "excludes_zero": bool(lo > 0 or hi < 0),
        }
    return out


def main(argv=None) -> int:
    args = parse_args(argv)
    for path in (args.baseline, args.candidate):
        if not path.is_file():
            print(f"prediction file not found: {path}", file=sys.stderr)
            return 2
    base = ClassificationPredictions.load(args.baseline)
    cand = ClassificationPredictions.load(args.candidate)
    if not np.array_equal(base.sample_keys, cand.sample_keys):
        print("the two files do not hold the same studies in the same order", file=sys.stderr)
        return 2
    if not np.array_equal(base.labels, cand.labels):
        print("labels differ between the two files", file=sys.stderr)
        return 2
    source = cand if args.subgroups_from == "candidate" else base
    if source.num_views is None:
        print(
            f"{args.subgroups_from} has no num_views; re-run its evaluation with a "
            "build that exports it",
            file=sys.stderr,
        )
        return 2

    rng = np.random.default_rng(args.seed)
    result = {
        "protocol": "paper three-class argmax, paired bootstrap over studies",
        "baseline": str(args.baseline),
        "candidate": str(args.candidate),
        "subgroups_from": args.subgroups_from,
        "resamples": args.resamples,
        "seed": args.seed,
        "groups": {},
    }
    for name, idx in view_groups(source.num_views).items():
        result["groups"][name] = {
            "n": int(idx.size),
            "changed_argmax_fraction": changed_fraction(base, cand, idx),
            "metrics": paired(base, cand, idx, rng, args.resamples),
        }
        g = result["groups"][name]
        print(f"[{name}] n={g['n']} changed argmax cells {g['changed_argmax_fraction']:.4f}")
        for metric, v in g["metrics"].items():
            mark = " *" if v["excludes_zero"] else ""
            print(
                f"  {metric:22s} {v['baseline']:.4f} -> {v['candidate']:.4f} "
                f"{v['delta']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}]{mark}"
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
