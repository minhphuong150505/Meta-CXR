#!/usr/bin/env python3
"""Attention probes and reading rule of the BioViL-T previous-image probe.

Plan and reading rule: ``docs/handoff/PLAN-2026-10-31-biovil-prior-probe.md``. Features come
from ``scripts/biovil_prior_features.py``. Probes are the encoder probe's ``fit_attention``
unchanged (3 seeds, holdout = 10% of labelled train studies, permutation seed 16).

Gates: G0 -- R1C's val npz re-scores to 0.4271 and its labels match after key alignment;
G1 -- attn(swin), attn(biovil), attn(swin+biovil) within 0.003 of the encoder probe's 0.4263 /
0.4039 / 0.4315. Deciding: delta = attn(swin+biovilprior) - attn(swin+biovil), val macro AP of
"present", paired study bootstrap: GO_ON (>= +0.02, lower bound > 0) / NOT_ENOUGH / STOP (<= 0).
Prints aggregates only.

    python scripts/biovil_prior_fit.py --train-dir <dir> --val-dir <dir> --mhcac R1C=<val npz> \\
        --report <dir>/report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (_REPO_ROOT, _REPO_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from encoder_probe_fit import (  # noqa: E402
    RARE,
    _load,
    fit_attention,
    macro_ap,
    paired_bootstrap,
    per_finding,
)

GO_MARGIN = 0.02
G1_TOLERANCE = 0.003
RECORDED = {"attn_swin": 0.4263, "attn_biovil": 0.4039, "attn_swin+biovil": 0.4315, "mhcac_R1C": 0.4271}
COMBOS = (("swin",), ("biovil",), ("biovilprior",), ("swin", "biovil"), ("swin", "biovilprior"))


def verdict(delta: float, lo: float) -> str:
    if delta <= 0:
        return "STOP"
    if delta >= GO_MARGIN and lo > 0:
        return "GO_ON"
    return "NOT_ENOUGH"


def gate1(scores: dict) -> dict:
    """Recorded value, re-measured value and pass flag for each G1 probe."""
    out = {}
    for k in ("attn_swin", "attn_biovil", "attn_swin+biovil"):
        out[k] = {"recorded": RECORDED[k], "now": scores[k], "passed": abs(scores[k] - RECORDED[k]) <= G1_TOLERANCE}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-dir", type=Path, required=True)
    ap.add_argument("--val-dir", type=Path, required=True)
    ap.add_argument("--mhcac", required=True, help="R1C=<val npz> (gate G0 + reference)")
    ap.add_argument("--report", type=Path, required=True)
    args = ap.parse_args(argv)

    import torch

    from training.evaluation.schemas import ClassificationPredictions

    names = ["swin", "biovil", "biovilprior"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr, va = _load(args.train_dir, names), _load(args.val_dir, names)
    y_tr, y_val = tr["labels"], va["labels"]
    has_prior = np.load(args.val_dir / "has_prior.npy")
    labelled = np.flatnonzero((y_tr >= 0).any(1))
    perm = np.random.default_rng(16).permutation(labelled)
    hold, fit = np.sort(perm[: len(perm) // 10]), np.sort(perm[len(perm) // 10:])
    report: dict = {"n_fit": int(len(fit)), "n_hold": int(len(hold)), "n_val": int(len(y_val)),
                    "val_with_prior": int(has_prior.sum()),
                    "train_with_prior": int(np.load(args.train_dir / "has_prior.npy").sum())}
    print(f"[prior] fit {len(fit)}, holdout {len(hold)}, val {len(y_val)} ({has_prior.sum()} with prior)", flush=True)

    name, path = args.mhcac.split("=", 1)
    p = ClassificationPredictions.load(path)
    finding_names = list(p.pathology_names)
    value = macro_ap(p.probabilities, p.labels)
    pos = {k: i for i, k in enumerate(va["keys"].tolist())}
    order = np.array([pos[k] for k in p.sample_keys.tolist()])
    if abs(value - RECORDED[f"mhcac_{name}"]) > 5e-5 or not np.array_equal(
            np.where(p.labels < 0, -1, p.labels), np.where(y_val[order] < 0, -1, y_val[order])):
        raise SystemExit(f"G0 FAILED: {name} macro AP {value:.4f} or labels differ")
    ref = np.empty_like(p.probabilities)
    ref[order] = p.probabilities
    print(f"[prior] G0 passed ({name} macro AP {value:.4f})", flush=True)

    probs = {f"mhcac_{name}": ref}
    for c in COMBOS:
        hs, pv = fit_attention(tr, list(c), fit, hold, y_tr, va, y_val, device)
        key = "attn_" + "+".join(c)
        probs[key] = pv
        print(f"[prior] {key}: holdout {hs:.4f}, val macro AP {macro_ap(pv, y_val):.4f}", flush=True)
        if key == "attn_swin+biovil":
            report["gate1"] = gate1({k: macro_ap(probs[k], y_val) for k in
                                     ("attn_swin", "attn_biovil", "attn_swin+biovil")})
            print(f"[prior] G1 {report['gate1']}", flush=True)
            if not all(v["passed"] for v in report["gate1"].values()):
                args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
                raise SystemExit("G1 FAILED: the probe does not reproduce the encoder probe")

    report["table"] = {}
    print("\n[prior] val, macro AP / AUROC present (all | with prior):")
    sub = np.flatnonzero(has_prior)
    for k, pv in probs.items():
        apf, aucf = per_finding(pv, y_val)
        report["table"][k] = {"macro_ap": float(np.nanmean(apf)), "auroc_present": float(np.nanmean(aucf)),
                              "macro_ap_with_prior": macro_ap(pv, y_val, sub), "ap_per_finding": apf.tolist()}
        print(f"  {k:24s} {np.nanmean(apf):.4f} / {np.nanmean(aucf):.4f} | {report['table'][k]['macro_ap_with_prior']:.4f}")
    print("\n[prior] AP per finding (rare marked *): " + " | ".join(report["table"]))
    for j, f in enumerate(finding_names):
        print(f"  {f + ('*' if f in RARE else ''):30s} "
              + " ".join(f"{v['ap_per_finding'][j]:.3f}" for v in report["table"].values()))

    report["comparisons"] = {}
    for label, a, b, idx in (("B-A", "attn_swin+biovil", "attn_swin+biovilprior", None),
                             ("B-A with prior", "attn_swin+biovil", "attn_swin+biovilprior", sub),
                             ("biovilprior-biovil", "attn_biovil", "attn_biovilprior", None),
                             ("B-swin", "attn_swin", "attn_swin+biovilprior", None)):
        pa, pb, yy = (probs[a], probs[b], y_val) if idx is None else (probs[a][idx], probs[b][idx], y_val[idx])
        d, lo, hi = paired_bootstrap(pa, pb, yy)
        report["comparisons"][label] = {"delta": d, "lo": lo, "hi": hi}
        print(f"[prior] {label:20s} {d:+.4f} [{lo:+.4f}, {hi:+.4f}]" + ("" if label == "B-A" else " (descriptive)"))
    c = report["comparisons"]["B-A"]
    report["verdict"] = verdict(c["delta"], c["lo"])
    print(f"VERDICT {report['verdict']}")
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
