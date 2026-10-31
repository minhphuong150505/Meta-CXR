#!/usr/bin/env python3
"""Stage-1 operating points: headline wF1 cutpoints vs recall-leaning F-beta cutpoints vs argmax.

Plan: ``docs/handoff/PLAN-2026-11-01-recall-operating-point.md``. NOT the paper's protocol
(the paper reports argmax). Same model, same scores: only the decision rule moves, so AUROC/AP
do not change and a recall gain is a precision trade, never a better model.

``crossfit`` (validation only): 2 halves x ``--repeats``; each rule is fitted on one half and
scored on the other. Rules: ``headline`` (cutpoints maximising weighted F1), ``f<beta>`` (t1 by
F-beta of "present" = P or U vs N, t2 by weighted F1), ``argmax``.

``paired`` (test, once): cutpoint files fitted on val (``--rule NAME=file``, the first is the
reference) plus argmax, scored on the test npz, with a paired study bootstrap of each rule minus
the reference.

Metrics: micro recall / precision of "present" over labelled cells; mean per-finding weighted
F1 (paper protocol), its five-finding mean (F1-5), macro recall; findings called present per
study (No Finding excluded); per-finding present recall. Aggregates only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from training.evaluation.classification_metrics import (  # noqa: E402
    PAPER_FIVE_FINDINGS,
    decide,
    macro_recall,
    weighted_prf,
)
from training.evaluation.schemas import MISSING  # noqa: E402
from training.evaluation.threshold_calibration import (  # noqa: E402
    apply_cutpoints,
    fit_severity_cutpoints,
)

HEADLINE_METRICS = ("present_recall", "present_precision", "weighted_f1", "f1_5", "macro_recall",
                    "present_calls_per_study")


def summarize(decisions: np.ndarray, labels: np.ndarray, names, idx=None) -> dict:
    """Decision-rule metrics over labelled cells; ``idx`` selects studies (bootstrap)."""
    if idx is not None:
        decisions, labels = decisions[idx], labels[idx]
    valid = labels != MISSING
    present, called = (labels != 0) & valid, (decisions != 0) & valid
    wf1, mrec, rec = {}, [], {}
    for j, name in enumerate(names):
        m = valid[:, j]
        y, d = labels[m, j].astype(int), decisions[m, j].astype(int)
        if y.size == 0:
            continue
        wf1[name] = weighted_prf(y, d)["f1"]
        mrec.append(macro_recall(y, d))
        rec[name] = float(np.mean(d[y != 0] != 0)) if np.any(y != 0) else float("nan")
    findings = [j for j, n in enumerate(names) if n != "No Finding"]
    return {
        "present_recall": float((present & called).sum() / max(present.sum(), 1)),
        "present_precision": float((present & called).sum() / max(called.sum(), 1)),
        "weighted_f1": float(np.mean(list(wf1.values()))),
        "f1_5": float(np.mean([wf1[n] for n in PAPER_FIVE_FINDINGS if n in wf1])),
        "macro_recall": float(np.mean(mrec)),
        "present_calls_per_study": float((decisions[:, findings] != 0).sum(1).mean()),
        "present_recall_per_finding": rec,
    }


def _subset(pred, idx):
    return SimpleNamespace(probabilities=pred.probabilities[idx], labels=pred.labels[idx],
                           pathology_names=tuple(pred.pathology_names))


def rule_decisions(rule: str, fit_on, score_on) -> np.ndarray:
    """Decisions on ``score_on`` for ``rule`` (``argmax``, ``headline``, ``f<beta>``) fitted on ``fit_on``."""
    names = tuple(score_on.pathology_names)
    if rule == "argmax":
        return decide(score_on.probabilities)
    if rule == "headline":
        cut = fit_severity_cutpoints(fit_on, objective="weighted_f1")
    elif rule.startswith("f"):
        cut = fit_severity_cutpoints(fit_on, objective="present_fbeta", beta=float(rule[1:]))
    else:
        raise ValueError(f"unknown rule {rule!r}")
    return apply_cutpoints(score_on.probabilities, cut, names)


def crossfit(pred, rules, repeats=10, seed=16) -> dict:
    """Mean of each rule's held-out metrics over 2 halves x ``repeats`` (never scored where fitted)."""
    rng = np.random.default_rng(seed)
    n = len(pred.labels)
    acc: dict = {r: [] for r in rules}
    for _ in range(repeats):
        perm = rng.permutation(n)
        halves = (np.sort(perm[: n // 2]), np.sort(perm[n // 2:]))
        for a, b in ((0, 1), (1, 0)):
            fit_on, score_on = _subset(pred, halves[a]), _subset(pred, halves[b])
            for r in rules:
                acc[r].append(summarize(rule_decisions(r, fit_on, score_on), score_on.labels,
                                        score_on.pathology_names))
    out = {}
    for r, runs in acc.items():
        out[r] = {k: float(np.mean([x[k] for x in runs])) for k in HEADLINE_METRICS}
        out[r]["present_recall_per_finding"] = {
            f: float(np.nanmean([x["present_recall_per_finding"].get(f, np.nan) for x in runs]))
            for f in runs[0]["present_recall_per_finding"]}
    return out


def paired(test, decisions: dict, n=1000, seed=16) -> dict:
    """Full-test metrics per rule and paired bootstrap deltas of each rule minus the first."""
    names = tuple(test.pathology_names)
    ref = next(iter(decisions))
    out = {r: summarize(d, test.labels, names) for r, d in decisions.items()}
    rng = np.random.default_rng(seed)
    deltas = {r: {k: [] for k in HEADLINE_METRICS} for r in decisions if r != ref}
    for _ in range(n):
        i = rng.integers(0, len(test.labels), len(test.labels))
        base = summarize(decisions[ref], test.labels, names, i)
        for r in deltas:
            other = summarize(decisions[r], test.labels, names, i)
            for k in HEADLINE_METRICS:
                deltas[r][k].append(other[k] - base[k])
    comp = {}
    for r, ks in deltas.items():
        comp[f"{r}-{ref}"] = {k: {"delta": out[r][k] - out[ref][k],
                                  "lo": float(np.percentile(v, 2.5)), "hi": float(np.percentile(v, 97.5))}
                              for k, v in ks.items()}
    return {"rules": out, "comparisons": comp}


def _print_table(results: dict, names) -> None:
    print(f"{'rule':12s} " + " ".join(f"{k:>14s}" for k in HEADLINE_METRICS))
    for r, m in results.items():
        print(f"{r:12s} " + " ".join(f"{m[k]:14.4f}" for k in HEADLINE_METRICS))
    print("\npresent recall per finding: " + " | ".join(results))
    for f in names:
        print(f"  {f:30s} " + " ".join(f"{m['present_recall_per_finding'].get(f, float('nan')):.3f}"
                                       for m in results.values()))


def main(argv=None) -> int:
    from training.evaluation.schemas import ClassificationPredictions
    from training.evaluation.threshold_calibration import load_cutpoint_file

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("crossfit")
    c.add_argument("--val", type=Path, required=True)
    c.add_argument("--betas", default="1,1.5,2,3,4")
    c.add_argument("--repeats", type=int, default=10)
    c.add_argument("--report", type=Path, required=True)
    p = sub.add_parser("paired")
    p.add_argument("--test", type=Path, required=True)
    p.add_argument("--rule", action="append", required=True, help="NAME=cutpoint file fitted on val; first = reference")
    p.add_argument("--report", type=Path, required=True)
    args = ap.parse_args(argv)

    if args.cmd == "crossfit":
        pred = ClassificationPredictions.load(args.val)
        if str(pred.metadata.get("split", "")) == "test":
            raise SystemExit("crossfit reads validation only")
        rules = ["headline"] + [f"f{b}" for b in args.betas.split(",")] + ["argmax"]
        results = crossfit(pred, rules, repeats=args.repeats)
        _print_table(results, pred.pathology_names)
        report = {"mode": "crossfit", "n": int(len(pred.labels)), "repeats": args.repeats, "rules": results}
    else:
        test = ClassificationPredictions.load(args.test)
        decisions = {}
        for spec in args.rule:
            name, path = spec.split("=", 1)
            cut, meta = load_cutpoint_file(path)
            if str(meta.get("split", "")) != "val":
                raise SystemExit(f"{path} was not fitted on val (split={meta.get('split')!r})")
            decisions[name] = apply_cutpoints(test.probabilities, cut, tuple(test.pathology_names))
        decisions["argmax"] = decide(test.probabilities)
        report = {"mode": "paired", "n": int(len(test.labels)), **paired(test, decisions)}
        _print_table(report["rules"], test.pathology_names)
        print()
        for comp, ks in report["comparisons"].items():
            for k, v in ks.items():
                print(f"{comp:20s} {k:24s} {v['delta']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}]")
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
