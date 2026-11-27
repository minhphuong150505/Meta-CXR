#!/usr/bin/env python
"""Paper-protocol Stage-1 table for OFF + residual lateral R1 vs OFF (2026-11-27).

Reads the npz written by ``scripts/residual_stack.py --save-arms`` (whole scored split): OFF's
probabilities and OFF+L = OFF + R1 per expert seed. Decisions follow residual_stack's own comparison:
on frontal+lateral (PL) studies EACH arm (OFF too) uses cutpoints fitted on val PL under that arm; on
every other study both use OFF's cutpoint file, and R1's rows there are OFF's bit for bit. The plain
OFF (cutpoint file on every study, as reported for run_20261024_nomv) is printed as ``off_reported``,
without an interval. Scores both with ``evaluate_classification`` (three classes, per-finding weighted
P/R/F1 with zero_division=1 averaged over the 14 findings, F1-5, one-vs-rest AUROC per class):

* headline = each arm's own decisions (the project's cutpoint rule, NOT the paper's protocol);
* argmax reference = argmax of each arm's probabilities (the paper's decision rule).

R1 numbers are the mean over the expert seeds (as in Step 3, so ``weighted_f1`` / ``mean_weighted_f1_5``
/ ``macro_recall`` reproduce ``whole_<split>["OFF+L"]`` of its JSON). Intervals: paired study
bootstrap (same resamples for both arms and every seed); for R1 the statistic is the seed mean.

No model selection happens here: the candidate and its cutpoints were frozen before the test was
read (Step 3, 2026-11-05), so this re-scores that one reading. Writes aggregates only.

    python scripts/r1_stage1_table.py --arms <saved.npz> --out <table.json> [--boot 2000]
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
    "weighted_precision", "weighted_recall", "weighted_f1", "mean_weighted_f1_5", "macro_recall",
    "auroc_negative_mean", "auroc_positive_mean", "auroc_uncertain_mean", "auroc_mean",
)
DECISION_METRICS = METRICS[:5]  # AUROC does not depend on the decision rule
BOOT_N, BOOT_SEED = 2000, 16


def _preds(labels, probs, names, keys) -> ClassificationPredictions:
    return ClassificationPredictions(labels=labels, probabilities=probs, pathology_names=names, sample_keys=keys)


def score_arm(labels, probs, decisions, names, keys, idx=None) -> dict[str, dict[str, float]]:
    """{'headline': aggregates, 'argmax': aggregates} on rows ``idx`` (all rows if None)."""
    if idx is not None:
        labels, probs, decisions, keys = labels[idx], probs[idx], decisions[idx], keys[idx]
    p = _preds(labels, probs, names, keys)
    head = evaluate_classification(p, decisions=decisions).aggregates
    arg = evaluate_classification(p).aggregates
    return {"headline": {m: head[m] for m in METRICS}, "argmax": {m: arg[m] for m in DECISION_METRICS}}


def score_r1(labels, probs_k, dec_k, names, keys, idx=None) -> dict[str, dict[str, float]]:
    rows = [score_arm(labels, p, d, names, keys, idx) for p, d in zip(probs_k, dec_k, strict=True)]
    return {rule: {m: float(np.mean([r[rule][m] for r in rows])) for m in rows[0][rule]} for rule in rows[0]}


def _flat(s: dict[str, dict[str, float]]) -> dict[str, float]:
    return {f"{rule}.{m}": v for rule, d in s.items() for m, v in d.items()}


def build_table(arms: dict, *, boot_n: int = BOOT_N, boot_seed: int = BOOT_SEED) -> dict:
    labels = np.asarray(arms["labels"], dtype=np.int64)
    names = tuple(str(n) for n in arms["pathology_names"])
    keys = np.asarray(arms["sample_keys"])
    p_off, d_off = np.asarray(arms["off_probabilities"]), np.asarray(arms["off_decisions"])
    p_r1, d_r1 = np.asarray(arms["r1_probabilities"]), np.asarray(arms["r1_decisions"])
    is_pl = np.asarray(arms["is_pl"], dtype=bool)
    d_rep = np.asarray(arms["off_headline_decisions"]) if "off_headline_decisions" in arms else None
    if p_r1.ndim != 4 or d_r1.shape != p_r1.shape[:3] or p_r1.shape[1:] != p_off.shape:
        raise ValueError("r1 arrays must be [seeds, N, findings(, 3)] and match OFF")
    non_pl = ~is_pl
    for s in range(len(p_r1)):  # R1 touches frontal+lateral studies only (both arms share non-PL decisions)
        if not (np.array_equal(p_r1[s][non_pl], p_off[non_pl]) and np.array_equal(d_r1[s][non_pl], d_off[non_pl])):
            raise ValueError(f"seed {s}: non-PL rows differ from OFF")

    def both(idx):
        return _flat(score_arm(labels, p_off, d_off, names, keys, idx)), _flat(score_r1(labels, p_r1, d_r1, names, keys, idx))

    out: dict = {"n": int(len(labels)), "n_pl": int(is_pl.sum()), "n_seeds": int(len(p_r1)),
                 "scored_split": str(arms.get("scored_split", "unknown"))}
    if d_rep is not None:
        out["off_reported"] = score_arm(labels, p_off, d_rep, names, keys)
    groups = {"all": None, "pl": np.flatnonzero(is_pl)}
    rng = np.random.default_rng(boot_seed)
    for gname, gidx in groups.items():
        base = np.arange(len(labels)) if gidx is None else gidx
        off_pt, r1_pt = both(base)
        draws = {k: [] for k in off_pt}
        dr_off = {k: [] for k in off_pt}
        dr_r1 = {k: [] for k in off_pt}
        for _ in range(boot_n):
            idx = base[rng.integers(0, len(base), len(base))]
            o, r = both(idx)
            for k in off_pt:
                draws[k].append(r[k] - o[k])
                dr_off[k].append(o[k])
                dr_r1[k].append(r[k])

        def ci(v):
            v = np.asarray(v, dtype=np.float64)
            v = v[~np.isnan(v)]
            return [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))] if v.size else [float("nan")] * 2

        out[gname] = {k: {"off": off_pt[k], "off_ci": ci(dr_off[k]), "r1": r1_pt[k], "r1_ci": ci(dr_r1[k]),
                          "delta": r1_pt[k] - off_pt[k], "delta_ci": ci(draws[k])} for k in off_pt}
    out["note"] = ("headline = val-fitted cutpoints (PL studies: each arm's own; others: OFF's file) -- NOT the paper's "
                   "protocol; argmax = the paper's decision rule. R1 = mean over expert seeds. Paired study "
                   f"bootstrap, {boot_n} resamples, seed {boot_seed}.")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", type=Path, required=True, help="npz from residual_stack.py --save-arms")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--boot", type=int, default=BOOT_N)
    args = ap.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    with np.load(args.arms, allow_pickle=False) as z:
        arms = {k: z[k] for k in z.files}
    table = build_table(arms, boot_n=args.boot)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(table, indent=2, sort_keys=True) + "\n")
    for g in ("all", "pl"):
        print(f"[r1-table] {g} (n={table['n'] if g == 'all' else table['n_pl']})")
        for k, v in table[g].items():
            print(f"  {k:32s} OFF {v['off']:.4f}  R1 {v['r1']:.4f}  delta {v['delta']:+.4f} "
                  f"[{v['delta_ci'][0]:+.4f}, {v['delta_ci'][1]:+.4f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
