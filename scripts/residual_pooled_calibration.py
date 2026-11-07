#!/usr/bin/env python3
"""Why does R1 lower the POOLED (whole-split) ranking metrics, and does per-group calibration fix it? CPU only.

Plan and reading rule: ``docs/handoff/PLAN-2026-11-04-residual-lateral.md``, section "Hiệu chỉnh theo nhóm"
(user-approved 2026-11-07, rule fixed before running). Setup = B2a: R1 (L only, 3 seeds) fitted on train PL with
out-of-fold experts, scored on every val PL study; non-PL studies keep OFF's probabilities.

Score channels, per finding: ``pos`` = p_pos (target Positive), ``unc`` = p_unc (target Uncertain),
``present`` = p_pos / (p_pos + p_neg) (target Positive or Uncertain). Arms on the whole scored split:

* ``OFF``, ``R1`` -- raw probabilities.
* ``scale`` (diagnostic A) -- inside PL, OFF's order with R1's set of values (per finding and channel).
* ``rank`` (diagnostic A) -- inside PL, R1's order with OFF's set of values.
* ``OFF_cal``, ``R1_cal`` (fix B) -- one-dimensional Platt ``sigmoid(a * logit(x) + b)``, ``a > 0``, per (group
  PL / non-PL, finding, channel), fitted on TRAIN (OFF train logits; R1 applied to train PL with the OOF experts)
  and applied to the scored split. Strictly monotone, so every within-group AUROC / AP is unchanged (asserted);
  the calibrated channels are not a probability simplex and feed ranking metrics only -- N/U/P decisions are not
  touched.

Decision (rule): Delta = pooled AUROC Pos mean over 14 findings, ``R1_cal - OFF`` (OFF raw = the reported number),
seed-mean, paired study bootstrap (2,000, seed 16) over the whole scored split: lower CI > 0 -> "TANG"; CI holds 0
-> "HET GIAM"; upper CI < 0 -> "VAN GIAM". Prints and writes aggregates only.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import warnings
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_spec = importlib.util.spec_from_file_location("residual_stack", _HERE / "residual_stack.py")
rs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rs)

from training.evaluation.classification_metrics import average_precision, roc_auc  # noqa: E402

CHANNELS = ("pos", "unc", "present")
EPS, A_MIN, PLATT_LAM, BOOT_N, BOOT_SEED = 1e-7, 1e-3, 1e-6, 2000, 16


# ----------------------------------------------------------------------------- score channels
def channel_scores(p: np.ndarray) -> dict[str, np.ndarray]:
    """``[n, F]`` score per channel from ``[n, F, 3]`` probabilities (class order N, P, U)."""
    p = np.asarray(p, dtype=np.float64)
    return {"pos": p[..., 1].copy(), "unc": p[..., 2].copy(),
            "present": p[..., 1] / np.maximum(p[..., 1] + p[..., 0], 1e-12)}


def channel_target(labels: np.ndarray, channel: str) -> np.ndarray:
    return {"pos": labels == 1, "unc": labels == 2, "present": labels > 0}[channel]


def _logit(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=np.float64), EPS, 1.0 - EPS)
    return np.log(x) - np.log1p(-x)


def platt_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """``(a, b)`` of ``sigmoid(a * logit(x) + b)`` by BCE, ``a >= A_MIN``; identity (1, 0) when y is one-class."""
    from scipy.optimize import minimize

    y = np.asarray(y, dtype=np.float64)
    if len(y) == 0 or y.min() == y.max():
        return 1.0, 0.0
    z = _logit(x)

    def f(theta):
        a, b = theta
        u = a * z + b
        loss = np.mean(np.logaddexp(0.0, u) - y * u) + PLATT_LAM * (a * a + b * b)
        g = 1.0 / (1.0 + np.exp(-u)) - y
        return loss, np.array([np.mean(g * z) + 2 * PLATT_LAM * a, np.mean(g) + 2 * PLATT_LAM * b])

    res = minimize(f, np.array([1.0, 0.0]), jac=True, method="L-BFGS-B", bounds=[(A_MIN, None), (None, None)])
    return float(res.x[0]), float(res.x[1])


def platt_apply(x: np.ndarray, a: float, b: float) -> np.ndarray:
    u = a * _logit(x) + b
    return 1.0 / (1.0 + np.exp(-u))


def fit_group_calibration(scores: dict[str, np.ndarray], labels: np.ndarray, groups: np.ndarray) -> dict:
    """``{(group, channel): (A [F], B [F])}`` fitted on the labelled cells of each group."""
    out = {}
    for g in np.unique(groups):
        rows = groups == g
        for ch in CHANNELS:
            F = labels.shape[1]
            A, B = np.ones(F), np.zeros(F)
            for j in range(F):
                m = rows & (labels[:, j] >= 0)
                A[j], B[j] = platt_fit(scores[ch][m, j], channel_target(labels[m, j], ch))
            out[(str(g), ch)] = (A, B)
    return out


def apply_group_calibration(scores: dict[str, np.ndarray], groups: np.ndarray, params: dict) -> dict[str, np.ndarray]:
    out = {ch: s.copy() for ch, s in scores.items()}
    for (g, ch), (A, B) in params.items():
        rows = groups == g
        for j in range(len(A)):
            out[ch][rows, j] = platt_apply(scores[ch][rows, j], A[j], B[j])
    return out


# ----------------------------------------------------------------------------- diagnostic A
def rank_swap(order_from: np.ndarray, values_from: np.ndarray) -> np.ndarray:
    """``order_from``'s order filled with ``values_from``'s sorted values (column-wise for 2-D input)."""
    order_from, values_from = np.asarray(order_from), np.asarray(values_from)
    if order_from.ndim == 1:
        out = np.empty_like(values_from, dtype=np.float64)
        out[np.argsort(order_from, kind="stable")] = np.sort(values_from)
        return out
    return np.stack([rank_swap(order_from[:, j], values_from[:, j]) for j in range(order_from.shape[1])], 1)


def swap_inside(base: dict, order_src: dict, value_src: dict, rows: np.ndarray, labels: np.ndarray) -> dict:
    """``base`` outside ``rows``; inside, per finding and channel over the labelled cells, ``order_src``'s order with
    ``value_src``'s values."""
    out = {ch: s.copy() for ch, s in base.items()}
    for ch in CHANNELS:
        for j in range(labels.shape[1]):
            m = rows & (labels[:, j] >= 0)
            out[ch][m, j] = rank_swap(order_src[ch][m, j], value_src[ch][m, j])
    return out


# ----------------------------------------------------------------------------- metrics
def pooled_metrics(scores: dict[str, np.ndarray], labels: np.ndarray, idx=None) -> dict[str, np.ndarray]:
    """Per finding: AUROC Pos, AUROC Unc, AP present over the given rows (all if ``idx`` is None)."""
    if idx is not None:
        scores = {ch: s[idx] for ch, s in scores.items()}
        labels = labels[idx]
    F = labels.shape[1]
    au_pos, au_unc, ap_prs = np.full(F, np.nan), np.full(F, np.nan), np.full(F, np.nan)
    for j in range(F):
        m = labels[:, j] >= 0
        y = labels[m, j]
        au_pos[j] = roc_auc(scores["pos"][m, j], y == 1)
        au_unc[j] = roc_auc(scores["unc"][m, j], y == 2)
        if (y > 0).any():
            ap_prs[j] = average_precision(scores["present"][m, j], y > 0)
    return {"auroc_pos": au_pos, "auroc_unc": au_unc, "ap_present": ap_prs}


def _mean(x) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return float(np.nanmean(x))


def summarise(per_find: dict[str, np.ndarray]) -> dict[str, float]:
    return {"auroc_pos_mean": _mean(per_find["auroc_pos"]), "auroc_unc_mean": _mean(per_find["auroc_unc"]),
            "ap_present_macro": _mean(per_find["ap_present"])}


def within_group_max_change(raw: dict, cal: dict, labels: np.ndarray, groups: np.ndarray) -> float:
    """Largest |metric(cal) - metric(raw)| over groups, findings and metrics (0 for a strictly monotone map)."""
    worst = 0.0
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        a, b = pooled_metrics(raw, labels, idx), pooled_metrics(cal, labels, idx)
        for k in a:
            d = np.abs(a[k] - b[k])
            d = d[~np.isnan(d)]
            if len(d):
                worst = max(worst, float(d.max()))
    return worst


def run(off_va: np.ndarray, r1_va: list[np.ndarray], y_va: np.ndarray, groups_va: np.ndarray,
        off_tr: np.ndarray, r1_tr: list[np.ndarray], y_tr: np.ndarray, groups_tr: np.ndarray, *,
        boot_n: int = BOOT_N, boot_seed: int = BOOT_SEED) -> dict:
    """Every arm and the decision. ``off_*`` ``[n, F, 3]`` on the whole split; ``r1_*`` per seed, same shape, equal
    to ``off_*`` outside PL; ``groups_*`` "PL" / "nonPL"."""
    pl = groups_va == "PL"
    off = channel_scores(off_va)
    cal_off = fit_group_calibration(channel_scores(off_tr), y_tr, groups_tr)
    arms: dict[str, list[dict]] = {"OFF": [off], "OFF_cal": [apply_group_calibration(off, groups_va, cal_off)]}
    for name in ("R1", "scale", "rank", "R1_cal"):
        arms[name] = []
    a_params = []
    worst = within_group_max_change(off, arms["OFF_cal"][0], y_va, groups_va)
    for p_va, p_tr in zip(r1_va, r1_tr, strict=True):
        if not np.array_equal(p_va[~pl], off_va[~pl]):
            raise ValueError("R1 must equal OFF outside PL")
        on = channel_scores(p_va)
        arms["R1"].append(on)
        arms["scale"].append(swap_inside(off, off, on, pl, y_va))
        arms["rank"].append(swap_inside(off, on, off, pl, y_va))
        cal_on = fit_group_calibration(channel_scores(p_tr), y_tr, groups_tr)
        a_params.append({f"{g}/{ch}": [float(v) for v in AB[0]] for (g, ch), AB in cal_on.items()})
        arms["R1_cal"].append(apply_group_calibration(on, groups_va, cal_on))
        worst = max(worst, within_group_max_change(on, arms["R1_cal"][-1], y_va, groups_va))
    if worst > 1e-12:
        raise AssertionError(f"calibration changed a within-group metric by {worst:.3g}")

    point = {a: [summarise(pooled_metrics(s, y_va)) for s in ss] for a, ss in arms.items()}
    mean = {a: {m: float(np.mean([x[m] for x in v])) for m in v[0]} for a, v in point.items()}
    per_find = {a: np.mean([pooled_metrics(s, y_va)["auroc_pos"] for s in ss], 0) for a, ss in arms.items()}

    rng = np.random.default_rng(boot_seed)
    n = len(y_va)
    boot = {a: np.zeros(boot_n) for a in arms}
    for r in range(boot_n):
        i = rng.integers(0, n, n)
        for a, ss in arms.items():
            boot[a][r] = np.mean([_mean(pooled_metrics({"pos": s["pos"], "unc": s["unc"], "present": s["present"]},
                                                       y_va, i)["auroc_pos"]) for s in ss])

    def diff(b, a):
        d = boot[b] - boot[a]
        lo, hi = np.nanpercentile(d, [2.5, 97.5])
        return {"delta": mean[b]["auroc_pos_mean"] - mean[a]["auroc_pos_mean"], "lo": float(lo), "hi": float(hi),
                "auroc_unc_delta": mean[b]["auroc_unc_mean"] - mean[a]["auroc_unc_mean"],
                "ap_present_delta": mean[b]["ap_present_macro"] - mean[a]["ap_present_macro"]}

    diffs = {"R1-OFF": diff("R1", "OFF"), "scale-OFF": diff("scale", "OFF"), "rank-OFF": diff("rank", "OFF"),
             "R1_cal-OFF": diff("R1_cal", "OFF"), "R1_cal-OFF_cal": diff("R1_cal", "OFF_cal"),
             "OFF_cal-OFF": diff("OFF_cal", "OFF")}
    d = diffs["R1_cal-OFF"]
    verdict = "TANG" if d["lo"] > 0 else ("VAN GIAM" if d["hi"] < 0 else "HET GIAM")
    return {"n": int(n), "n_pl": int(pl.sum()), "k_seeds": len(r1_va), "arms": mean, "diffs": diffs,
            "verdict": verdict, "within_group_max_change": worst,
            "per_finding_auroc_pos": {a: [float(v) for v in x] for a, x in per_find.items()},
            "platt_a_R1": a_params}


# ----------------------------------------------------------------------------- CLI
def load_split(off_path: Path, view_map: Path, expert_spec: str, expect: str):
    """Whole split ``p [n, F, 3]``, labels, PL row index, groups and the L experts (fold models averaged) on PL."""
    from training.evaluation.schemas import ClassificationPredictions
    from training.evaluation.view_keys import PL, ViewMap, align

    vmap = ViewMap.load(view_map)
    expected = {k: int(v) for k, v in (kv.split("=") for kv in expect.split(","))}
    if vmap.counts() != expected:
        raise SystemExit(f"group counts {vmap.counts()} != expected {expected}")
    off = ClassificationPredictions.load(off_path)
    o = align(off.sample_keys, vmap.frontal_keys)
    p = np.asarray(off.probabilities, dtype=np.float64)[o]
    y = np.asarray(off.labels, dtype=np.int64)[o]
    y[y < 0] = -1
    pl = np.flatnonzero(vmap.groups == PL)
    name, specs = expert_spec.split("=", 1)
    if name != "L":
        raise SystemExit("only the L expert is used (the reported candidate)")
    experts = []
    for spec in specs.split(","):
        keys, probs, labels = rs.load_expert_parts(spec, "mean")
        idx = align(keys, vmap.frontal_keys[pl])
        if not np.array_equal(labels[idx], y[pl]):
            raise SystemExit(f"labels of {spec} differ from OFF's on PL")
        experts.append(probs[idx])
    return p, y, pl, np.where(vmap.groups == PL, "PL", "nonPL"), experts


def main(argv=None) -> int:
    from training.evaluation.schemas import ClassificationPredictions
    from training.evaluation.view_keys import align

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--off", type=Path, required=True, help="OFF predictions npz of the scored split")
    ap.add_argument("--view-map", type=Path, required=True)
    ap.add_argument("--expert", required=True, help="L=seed1(a+b),seed2,... scored-split experts (fold models averaged)")
    ap.add_argument("--expect-counts", default="PL=797,F1=930,other=81")
    ap.add_argument("--train-off", type=Path, required=True, help="OFF train npz (Platt is always fitted on train)")
    ap.add_argument("--train-expert", required=True, help="L=seed1(f0_oof+f1_oof),... out-of-fold train experts")
    ap.add_argument("--r1-fit-off", type=Path, help="Step 3 candidate: fit R1 on ALL PL of this split (val) instead "
                                                    "of train PL; R1 is then applied to train PL and the scored split")
    ap.add_argument("--r1-fit-view-map", type=Path)
    ap.add_argument("--r1-fit-expert")
    ap.add_argument("--r1-fit-expect-counts", default="PL=797,F1=930,other=81")
    ap.add_argument("--scored-split", choices=("val", "test"), default="val")
    ap.add_argument("--allow-test", action="store_true", help="required with --scored-split test (read once)")
    ap.add_argument("--boot", type=int, default=BOOT_N)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    if args.scored_split == "test" and not args.allow_test:
        raise SystemExit("--scored-split test needs --allow-test")
    r1_val = [args.r1_fit_off, args.r1_fit_view_map, args.r1_fit_expert]
    if any(x is not None for x in r1_val) and not all(x is not None for x in r1_val):
        raise SystemExit("--r1-fit-off, --r1-fit-view-map and --r1-fit-expert go together")

    p_va, y_va, pl, groups_va, ex_va = load_split(args.off, args.view_map, args.expert, args.expect_counts)

    toff = ClassificationPredictions.load(args.train_off)
    p_tr_all = np.asarray(toff.probabilities, dtype=np.float64)
    y_tr_all = np.asarray(toff.labels, dtype=np.int64)
    y_tr_all[y_tr_all < 0] = -1
    tname, tspecs = args.train_expert.split("=", 1)
    if tname != "L":
        raise SystemExit("only the L expert is used (the reported candidate)")
    ex_tr, tr_keys = [], None
    for spec in tspecs.split(","):
        keys, probs, labels = rs.load_expert_parts(spec, "concat")
        if tr_keys is None:
            tr_keys = keys
            o_tr = align(toff.sample_keys, tr_keys)
        idx = align(keys, tr_keys)
        if not np.array_equal(labels[idx], y_tr_all[o_tr]):
            raise SystemExit(f"labels of {spec} differ from OFF's train labels")
        ex_tr.append(probs[idx])
    groups_tr = np.full(len(p_tr_all), "nonPL", dtype=object)
    groups_tr[o_tr] = "PL"
    groups_tr = groups_tr.astype(str)

    fit = None
    if args.r1_fit_off is not None:
        p_f, y_f, pl_f, _g, ex_f = load_split(args.r1_fit_off, args.r1_fit_view_map, args.r1_fit_expert,
                                              args.r1_fit_expect_counts)
        if len(ex_f) != len(ex_va):
            raise SystemExit("R1-fit and scored experts need the same seeds")
        fit = (p_f[pl_f], y_f[pl_f], ex_f)

    r1_va, r1_tr = [], []
    for s, (e_va, e_tr) in enumerate(zip(ex_va, ex_tr, strict=True)):
        if fit is None:
            probs_pl, prm = rs.r1_trainfit(p_tr_all[o_tr], [e_tr], y_tr_all[o_tr], p_va[pl], [e_va])
        else:
            probs_pl, prm = rs.r1_trainfit(fit[0], [fit[2][s]], fit[1], p_va[pl], [e_va])
        full_va = p_va.copy()
        full_va[pl] = probs_pl
        r1_va.append(full_va)
        full_tr = p_tr_all.copy()
        full_tr[o_tr] = rs.r1_apply_params(p_tr_all[o_tr], [e_tr], *prm[0])
        r1_tr.append(full_tr)
    print(f"[calib] scored {args.scored_split} n={len(y_va)} (PL {len(pl)}), train n={len(y_tr_all)} "
          f"(PL {len(o_tr)}), R1 fitted on {'train PL' if fit is None else 'r1-fit split PL'}", flush=True)

    res = run(p_va, r1_va, y_va, groups_va, p_tr_all, r1_tr, y_tr_all, groups_tr, boot_n=args.boot)
    res["scored_split"] = args.scored_split
    res["r1_fit_on"] = "train_pl_oof" if fit is None else "r1_fit_split_pl_all"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2, sort_keys=True) + "\n")
    for a, m in res["arms"].items():
        print(f"  {a:8s} AUROC Pos {m['auroc_pos_mean']:.4f}  AUROC Unc {m['auroc_unc_mean']:.4f}  "
              f"AP present {m['ap_present_macro']:.4f}")
    for k, d in res["diffs"].items():
        print(f"  {k:15s} {d['delta']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]")
    print(f"  within-group max change {res['within_group_max_change']:.2e}; verdict: {res['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
