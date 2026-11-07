#!/usr/bin/env python3
"""Output-level residual R1 on a frozen OFF model, for PL studies only; reading rule of Step 0/1/2a.

Plan and reading rule: ``docs/handoff/PLAN-2026-11-04-residual-lateral.md`` (sections 2 and 7).
For every finding f and every PL study (frontal PA/AP anchor + lateral aux):

    z_ON = log p_OFF + sum_E a[f, E] * log p_E + b[f],   a in [0, A_MAX], b[f, neg] = 0,

fitted per finding by masked three-class CE + ``lam * (sum a^2 + |b|^2)`` with L-BFGS-B from
a = b = 0 (= OFF). Fitted on VAL by two-half cross-fitting: PL studies sorted by key,
``fold = default_rng(16).permutation(n) % 2``, each half scored with the parameters fitted on
the other. Every study outside PL, and every PL cell whose fitted parameters are all zero,
keeps OFF's probabilities bit for bit.

Arms, per expert seed s: ``OFF+C``, ``OFF+L``, ``OFF+C+L``, ``OFF+Lperm`` (L's rows shuffled
within each fold). Differences: D1 = (OFF+L) - OFF, D2 = (OFF+C+L) - (OFF+C), D3 = (OFF+C) - OFF.
Deciding metric: mean over T (the five F1-5 findings) of the one-vs-rest AP of the Positive
class (Positive vs Negative or Uncertain, labelled cells only, score p_pos) on val PL. CI:
paired study bootstrap (2,000, seed 16) of the seed-mean difference. NB = 2 * SD across seeds
of the arm (OFF is fixed, so its SD is 0); a mean of k seeds is compared with NB / sqrt(k).

Prints and writes aggregates only.

    python scripts/residual_stack.py --off <OFF val npz> --view-map <val view_map.npz> \\
        --off-cutpoints <cutpoints_nomv.json> --expert L=<s16.npz>,<s17.npz>,<s18.npz> \\
        --expert C=<...> --name swin --out <dir>/residual_swin.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from training.evaluation.classification_metrics import (  # noqa: E402
    PAPER_FIVE_FINDINGS,
    average_precision,
    macro_recall,
    roc_auc,
    weighted_prf,
)

T_FINDINGS = PAPER_FIVE_FINDINGS  # user decision 2026-11-04 (plan, "Quyết định của user")
LAM, A_MAX, EPS, FOLD_SEED, BOOT_N, BOOT_SEED = 1e-2, 3.0, 1e-7, 16, 2000, 16
CUTPOINT_GRID = 121  # fit_severity_cutpoints' default; tests lower it for speed
MISSING = -1


# ----------------------------------------------------------------------------- R1 fit
def crossfit_folds(keys: np.ndarray, seed: int = FOLD_SEED) -> np.ndarray:
    """0/1 fold per row: rows sorted by key, ``default_rng(seed).permutation(n) % 2``."""
    keys = np.asarray(keys)
    fold = np.empty(len(keys), dtype=np.int64)
    fold[np.argsort(keys, kind="stable")] = np.random.default_rng(seed).permutation(len(keys)) % 2
    return fold


def _logp(p: np.ndarray) -> np.ndarray:
    return np.log(np.clip(np.asarray(p, dtype=np.float64), EPS, 1.0))


def _softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def fit_r1_finding(base: np.ndarray, experts: list[np.ndarray], y: np.ndarray,
                   lam: float = LAM, a_max: float = A_MAX) -> tuple[np.ndarray, np.ndarray]:
    """``(a [k], b [3])`` for one finding; ``base``/``experts`` are log-probs ``[n, 3]``, y in 0..2."""
    from scipy.optimize import minimize

    k = len(experts)
    if len(y) == 0:
        return np.zeros(k), np.zeros(3)
    X = np.stack(experts, 0) if k else np.zeros((0,) + base.shape)
    onehot = np.eye(3)[np.asarray(y, dtype=int)]
    n = len(y)

    def f(theta):
        a, b = theta[:k], np.r_[0.0, theta[k:]]
        z = base + np.tensordot(a, X, 1) + b if k else base + b
        logz = z - z.max(1, keepdims=True)
        logz = logz - np.log(np.exp(logz).sum(1, keepdims=True))
        loss = -(onehot * logz).sum() / n + lam * float(theta @ theta)
        g = (np.exp(logz) - onehot) / n
        grad = np.r_[[float((g * X[e]).sum()) for e in range(k)], g.sum(0)[1:]] + 2 * lam * theta
        return loss, grad

    res = minimize(f, np.zeros(k + 2), jac=True, method="L-BFGS-B",
                   bounds=[(0.0, a_max)] * k + [(None, None)] * 2)
    return res.x[:k].copy(), np.r_[0.0, res.x[k:]]


GATES = ("none", "entropy")


def off_entropy(p_off: np.ndarray) -> np.ndarray:
    """Normalised entropy ``H(p_OFF) / log 3`` in [0, 1] per row of a ``[n, 3]`` probability array."""
    p = np.clip(np.asarray(p_off, dtype=np.float64), EPS, 1.0)
    return -(p * np.log(p)).sum(-1) / math.log(3.0)


def r1_features(p_off: np.ndarray, experts: list[np.ndarray], gate: str = "none") -> list[np.ndarray]:
    """Log-prob features of one finding. ``gate="entropy"`` (R1H) appends ``h * log p_E`` per expert, so the
    fitted weight of expert E is ``a0[E] + a1[E] * h`` with ``h = off_entropy(p_off)``; both bounded [0, A_MAX]."""
    if gate not in GATES:
        raise ValueError(f"gate {gate!r} not in {GATES}")
    logs = [_logp(e) for e in experts]
    if gate == "none":
        return logs
    h = off_entropy(p_off)[:, None]
    return logs + [h * lg for lg in logs]


def apply_r1_gated(p_off: np.ndarray, experts: list[np.ndarray], a: np.ndarray, b: np.ndarray,
                   gate: str = "none") -> np.ndarray:
    """``apply_r1`` for ``gate="none"`` (bit for bit); R1H otherwise, ``a = [a0 per expert, a1 per expert]``."""
    if gate == "none":
        return apply_r1(p_off, experts, a, b)
    if not np.any(a) and not np.any(b):
        return np.asarray(p_off).copy()
    z = _logp(p_off) + b
    for coef, ft in zip(a, r1_features(p_off, experts, gate), strict=True):
        z = z + coef * ft
    return _softmax(z).astype(np.asarray(p_off).dtype)


def apply_r1(p_off: np.ndarray, experts: list[np.ndarray], a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``[n, 3]`` ON probabilities of one finding; exactly ``p_off`` when every parameter is 0."""
    if not np.any(a) and not np.any(b):
        return np.asarray(p_off).copy()
    z = _logp(p_off) + b
    for coef, lp in zip(a, experts, strict=True):
        z = z + coef * _logp(lp)
    return _softmax(z).astype(np.asarray(p_off).dtype)


def r1_crossfit(p_off: np.ndarray, experts: list[np.ndarray], labels: np.ndarray, fold: np.ndarray,
                lam: float = LAM, gate: str = "none") -> tuple[np.ndarray, dict]:
    """Out-of-fold ON probabilities ``[n, F, 3]`` and per-fold parameters ``{fold: (a [F, k], b [F, 3])}``."""
    n, F, _ = p_off.shape
    out = p_off.copy()
    params = {}
    for f_fit, f_score in ((0, 1), (1, 0)):
        fit_rows, score_rows = fold == f_fit, fold == f_score
        A, B = np.zeros((F, len(experts) * (1 if gate == "none" else 2))), np.zeros((F, 3))
        for j in range(F):
            m = fit_rows & (labels[:, j] >= 0)
            A[j], B[j] = fit_r1_finding(_logp(p_off[m, j]),
                                        r1_features(p_off[m, j], [e[m, j] for e in experts], gate),
                                        labels[m, j], lam)
            out[score_rows, j] = apply_r1_gated(p_off[score_rows, j], [e[score_rows, j] for e in experts],
                                                A[j], B[j], gate)
        params[f_score] = (A, B)
    return out, params


def cutpoint_decisions_from(ref_probs: np.ndarray, ref_labels: np.ndarray, probs: np.ndarray, names) -> np.ndarray:
    """Step 3: headline cutpoints fitted on the reference (val) rows, applied to the scored (test) rows."""
    from training.evaluation.threshold_calibration import apply_cutpoints, fit_severity_cutpoints

    cut = fit_severity_cutpoints(SimpleNamespace(probabilities=ref_probs, labels=ref_labels,
                                                 pathology_names=tuple(names)), grid_size=CUTPOINT_GRID)
    return apply_cutpoints(probs, cut, tuple(names))


def r1_apply_params(p_off: np.ndarray, experts: list[np.ndarray], A: np.ndarray, B: np.ndarray,
                    gate: str = "none") -> np.ndarray:
    out = p_off.copy()
    for j in range(p_off.shape[1]):
        out[:, j] = apply_r1_gated(p_off[:, j], [e[:, j] for e in experts], A[j], B[j], gate)
    return out


def r1_trainfit(p_off_tr: np.ndarray, ex_tr: list[np.ndarray], y_tr: np.ndarray,
                p_off_va: np.ndarray, ex_va: list[np.ndarray], lam: float = LAM,
                gate: str = "none") -> tuple[np.ndarray, dict]:
    """B2a: R1 fitted on train PL (out-of-fold experts), applied to every val PL row.

    Same per-finding fit as ``r1_crossfit``; parameters are returned under both fold ids so the
    callers that average over folds read them unchanged.
    """
    F = p_off_va.shape[1]
    out = p_off_va.copy()
    A, B = np.zeros((F, len(ex_tr) * (1 if gate == "none" else 2))), np.zeros((F, 3))
    for j in range(F):
        m = y_tr[:, j] >= 0
        A[j], B[j] = fit_r1_finding(_logp(p_off_tr[m, j]),
                                    r1_features(p_off_tr[m, j], [e[m, j] for e in ex_tr], gate), y_tr[m, j], lam)
        out[:, j] = apply_r1_gated(p_off_va[:, j], [e[:, j] for e in ex_va], A[j], B[j], gate)
    return out, {0: (A, B), 1: (A, B)}


def permute_within_folds(x: np.ndarray, fold: np.ndarray, seed: int = BOOT_SEED) -> np.ndarray:
    """Rows of ``x`` shuffled among the rows of the same fold."""
    rng = np.random.default_rng(seed)
    out = x.copy()
    for f in np.unique(fold):
        idx = np.flatnonzero(fold == f)
        out[idx] = x[rng.permutation(idx)]
    return out


# ----------------------------------------------------------------------------- metrics
def per_finding_scores(probs: np.ndarray, labels: np.ndarray, idx=None) -> dict[str, np.ndarray]:
    """Per finding: AP of Positive, AUROC of Positive and of Uncertain, AP of "present"."""
    if idx is not None:
        probs, labels = probs[idx], labels[idx]
    F = labels.shape[1]
    ap_pos, au_pos, au_unc, ap_prs = (np.full(F, np.nan) for _ in range(4))
    sev = probs[..., 1] / np.maximum(probs[..., 1] + probs[..., 0], 1e-12)
    for j in range(F):
        m = labels[:, j] >= 0
        y, p = labels[m, j], probs[m, j]
        if (y == 1).any():
            ap_pos[j] = average_precision(p[:, 1], y == 1)
        au_pos[j] = roc_auc(p[:, 1], y == 1)
        au_unc[j] = roc_auc(p[:, 2], y == 2)
        if (y != 0).any():
            ap_prs[j] = average_precision(sev[m, j], y != 0)
    return {"ap_pos": ap_pos, "auroc_pos": au_pos, "auroc_unc": au_unc, "ap_present": ap_prs}


def bootstrap_scores(probs: np.ndarray, labels: np.ndarray, idx: np.ndarray) -> tuple[np.ndarray, float]:
    """Per-finding AP of Positive and mean AUROC of Positive on resample ``idx`` (bootstrap fast path)."""
    probs, labels = probs[idx], labels[idx]
    F = labels.shape[1]
    ap, au = np.full(F, np.nan), np.full(F, np.nan)
    for j in range(F):
        m = labels[:, j] >= 0
        y = labels[m, j] == 1
        if y.any():
            ap[j] = average_precision(probs[m, j, 1], y)
        au[j] = roc_auc(probs[m, j, 1], y)
    with np.errstate(all="ignore"):
        return ap, float(np.nanmean(au))


def summary(scores: dict[str, np.ndarray], t_idx: np.ndarray, other_idx: np.ndarray) -> dict[str, float]:
    with np.errstate(all="ignore"):
        return {
            "ap_pos_T": float(np.nanmean(scores["ap_pos"][t_idx])),
            "ap_pos_nonT": float(np.nanmean(scores["ap_pos"][other_idx])),
            "auroc_pos_mean": float(np.nanmean(scores["auroc_pos"])),
            "auroc_unc_mean": float(np.nanmean(scores["auroc_unc"])),
            "ap_present_macro": float(np.nanmean(scores["ap_present"])),
        }


def cutpoint_decisions_crossfit(probs: np.ndarray, labels: np.ndarray, names, fold: np.ndarray) -> np.ndarray:
    """Headline cutpoints (weighted F1) fitted on one fold and applied to the other."""
    from training.evaluation.threshold_calibration import apply_cutpoints, fit_severity_cutpoints

    out = np.zeros(labels.shape, dtype=int)
    for f_fit, f_score in ((0, 1), (1, 0)):
        fit_rows, score_rows = fold == f_fit, fold == f_score
        cut = fit_severity_cutpoints(SimpleNamespace(probabilities=probs[fit_rows], labels=labels[fit_rows],
                                                     pathology_names=tuple(names)), grid_size=CUTPOINT_GRID)
        out[score_rows] = apply_cutpoints(probs[score_rows], cut, tuple(names))
    return out


def decision_metrics(decisions: np.ndarray, labels: np.ndarray, names) -> dict[str, float]:
    wf1, mrec = {}, []
    for j, name in enumerate(names):
        m = labels[:, j] >= 0
        if not m.any():
            continue
        wf1[name] = weighted_prf(labels[m, j], decisions[m, j])["f1"]
        mrec.append(macro_recall(labels[m, j], decisions[m, j]))
    return {"weighted_f1": float(np.mean(list(wf1.values()))),
            "f1_5": float(np.mean([wf1[n] for n in PAPER_FIVE_FINDINGS if n in wf1])),
            "macro_recall": float(np.mean(mrec))}


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm-adjusted p-values (step-down, monotone, capped at 1)."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    out, running, m = {}, 0.0, len(items)
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def nb(values_by_arm: list[list[float]]) -> float:
    """2 * sqrt(sum of per-arm seed variances); an arm with one value (OFF) contributes 0."""
    var = sum(float(np.var(v, ddof=1)) if len(v) > 1 else 0.0 for v in values_by_arm)
    return 2.0 * math.sqrt(var)


# ----------------------------------------------------------------------------- driver
def run(p_off, labels, names, keys, experts: dict[str, list[np.ndarray]], *, lam=LAM,
        boot_n=BOOT_N, boot_seed=BOOT_SEED, train: dict | None = None,
        cutpoints_from_train: bool = False, gate: str = "none") -> dict:
    """Every arm, metric, bootstrap and verdict on the PL rows given (all arrays aligned).

    ``train=None``: R1 cross-fitted on these (val) rows, as in Steps 0/1. ``train={"p_off", "labels",
    "keys", "experts"}`` (B2a): R1 fitted on train PL with out-of-fold experts and applied to every val
    row; ``Lperm`` shuffles L's rows within train for the fit and within each val fold for scoring.
    Everything downstream (metrics, bootstrap, cutpoints, verdict) is unchanged.
    ``cutpoints_from_train`` (Step 3, needs ``train``): each arm's PL cutpoints are fitted on the fit rows'
    probabilities under that arm (OFF: OFF's) and applied to the scored rows, instead of cross-fitted on them.
    ``gate="entropy"`` (R1H, 2026-11-07): every residual arm uses the entropy-gated weight
    ``a0 + a1 * H(p_OFF)/log 3``; an ungated ``OFF+L_R1`` arm is added and ``DH = (OFF+L) - (OFF+L_R1)``
    is read by ``rule_gate`` (lower CI > 0, delta > NB/sqrt(k), non-T >= -NB/sqrt(k)).
    """
    if gate not in GATES:
        raise ValueError(f"gate {gate!r} not in {GATES}")
    if cutpoints_from_train and train is None:
        raise ValueError("cutpoints_from_train needs train")
    names = tuple(names)
    t_idx = np.array([names.index(n) for n in T_FINDINGS])
    other_idx = np.array([j for j in range(len(names)) if j not in t_idx])
    fold = crossfit_folds(keys)
    has_c = "C" in experts
    k = len(experts["L"])
    if has_c and len(experts["C"]) != k:
        raise ValueError("L and C need the same number of seeds")

    arms: dict[str, list[np.ndarray]] = {"OFF": [p_off]}
    ref_arms: dict[str, list[np.ndarray]] = {"OFF": [train["p_off"]]} if cutpoints_from_train else {}
    params: dict[str, list[dict]] = {}
    specs = {"OFF+L": ["L"], "OFF+Lperm": ["Lperm"]}
    if has_c:
        specs.update({"OFF+C": ["C"], "OFF+C+L": ["C", "L"]})
    arm_gate = {arm: gate for arm in specs}
    if gate != "none":
        specs["OFF+L_R1"] = ["L"]
        arm_gate["OFF+L_R1"] = "none"
    if train is not None:
        tr_zero = np.zeros(len(train["labels"]), dtype=np.int64)
        if any(len(train["experts"][m]) != k for m in experts):
            raise ValueError("train and val need the same experts and seeds")
    for arm, members in specs.items():
        arms[arm], params[arm] = [], []
        if cutpoints_from_train:
            ref_arms[arm] = []
        for s in range(k):
            ex = [permute_within_folds(experts["L"][s], fold) if m == "Lperm" else experts[m][s] for m in members]
            g = arm_gate[arm]
            if train is None:
                probs, prm = r1_crossfit(p_off, ex, labels, fold, lam, g)
            else:
                ex_tr = [permute_within_folds(train["experts"]["L"][s], tr_zero) if m == "Lperm"
                         else train["experts"][m][s] for m in members]
                probs, prm = r1_trainfit(train["p_off"], ex_tr, train["labels"], p_off, ex, lam, g)
                if cutpoints_from_train:
                    ref_arms[arm].append(r1_apply_params(train["p_off"], ex_tr, *prm[0], g))
            arms[arm].append(probs)
            params[arm].append(prm)
    arms["L_alone"] = list(experts["L"])
    if has_c:
        arms["C_alone"] = list(experts["C"])

    point = {a: [summary(per_finding_scores(p, labels), t_idx, other_idx) for p in ps] for a, ps in arms.items()}
    per_find = {a: [per_finding_scores(p, labels)["ap_pos"] for p in ps] for a, ps in arms.items()}
    if cutpoints_from_train:
        decisions = {a: [cutpoint_decisions_from(r, train["labels"], p, names) for r, p in zip(ref_arms[a], ps,
                                                                                               strict=True)]
                     for a, ps in arms.items() if not a.endswith("_alone")}
    else:
        decisions = {a: [cutpoint_decisions_crossfit(p, labels, names, fold) for p in ps]
                     for a, ps in arms.items() if not a.endswith("_alone")}
    dec = {a: [decision_metrics(d, labels, names) for d in ds] for a, ds in decisions.items()}

    # paired bootstrap over PL studies, the same resamples for every arm and seed
    rng = np.random.default_rng(boot_seed)
    n = len(labels)
    boot = {a: np.full((boot_n, len(ps), len(names)), np.nan) for a, ps in arms.items()}
    boot_auc = {a: np.full((boot_n, len(ps)), np.nan) for a, ps in arms.items()}
    for r in range(boot_n):
        i = rng.integers(0, n, n)
        for a, ps in arms.items():
            for s, p in enumerate(ps):
                boot[a][r, s], boot_auc[a][r, s] = bootstrap_scores(p, labels, i)

    def diff(arm_b, arm_a):
        """Seed-mean difference b - a: point, CI, NB, per-finding point + Holm p over T."""
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out = {}
            for metric, sel in (("ap_pos_T", t_idx), ("ap_pos_nonT", other_idx)):
                vb = [x[metric] for x in point[arm_b]]
                va = [x[metric] for x in point[arm_a]]
                d = float(np.mean(vb) - np.mean(va))
                bd = np.nanmean(boot[arm_b][:, :, sel], 2).mean(1) - np.nanmean(boot[arm_a][:, :, sel], 2).mean(1)
                lo, hi = np.nanpercentile(bd, [2.5, 97.5])
                bar = nb([vb, va])
                kk = len(vb)
                out[metric] = {"delta": d, "lo": float(lo), "hi": float(hi), "nb": bar,
                               "nb_over_sqrt_k": bar / math.sqrt(kk), "k_seeds": kk}
            vb = [x["auroc_pos_mean"] for x in point[arm_b]]
            va = [x["auroc_pos_mean"] for x in point[arm_a]]
            bd = boot_auc[arm_b].mean(1) - boot_auc[arm_a].mean(1)
            lo, hi = np.nanpercentile(bd, [2.5, 97.5])
            out["auroc_pos_mean"] = {"delta": float(np.mean(vb) - np.mean(va)), "lo": float(lo), "hi": float(hi),
                                     "nb": nb([vb, va])}
            pf_b = np.mean(per_find[arm_b], 0)
            pf_a = np.mean(per_find[arm_a], 0)
            bf = boot[arm_b].mean(1) - boot[arm_a].mean(1)  # [B, F]
            pvals = {}
            per = {}
            for j, name in enumerate(names):
                col = bf[:, j][~np.isnan(bf[:, j])]
                lo, hi = (np.percentile(col, [2.5, 97.5]) if len(col) else (np.nan, np.nan))
                per[name] = {"delta": float(pf_b[j] - pf_a[j]), "lo": float(lo), "hi": float(hi)}
                if name in T_FINDINGS and len(col):
                    pvals[name] = float(min(1.0, 2 * min(np.mean(col <= 0), np.mean(col >= 0))))
            for name, p in holm(pvals).items():
                per[name]["holm_p"] = p
            out["per_finding_ap_pos"] = per
            if arm_b in dec and arm_a in dec:
                out["cutpoint_pl"] = {m: float(np.mean([x[m] for x in dec[arm_b]]) - np.mean([x[m] for x in dec[arm_a]]))
                                      for m in ("weighted_f1", "f1_5", "macro_recall")}
            for m in ("auroc_unc_mean", "ap_present_macro"):
                out[m] = {"delta": float(np.mean([x[m] for x in point[arm_b]]) - np.mean([x[m] for x in point[arm_a]]))}
        return out

    diffs = {"D1": diff("OFF+L", "OFF"), "Dperm": diff("OFF+Lperm", "OFF")}
    if has_c:
        diffs.update({"D2": diff("OFF+C+L", "OFF+C"), "D3": diff("OFF+C", "OFF"),
                      "L_minus_C_alone": diff("L_alone", "C_alone")})

    rule_gate = None
    if gate != "none":
        diffs["DH"] = diff("OFF+L", "OFF+L_R1")
        dh = diffs["DH"]["ap_pos_T"]
        dh_non = diffs["DH"]["ap_pos_nonT"]
        rule_gate = {"gate": gate,
                     "lower_ci_pass": bool(dh["lo"] > 0),
                     "nb_pass": bool(dh["delta"] > dh["nb_over_sqrt_k"]),
                     "guardrail_pass": bool(dh_non["delta"] >= -dh_non["nb_over_sqrt_k"])}
        ok = rule_gate["lower_ci_pass"] and rule_gate["nb_pass"] and rule_gate["guardrail_pass"]
        rule_gate["verdict"] = "R1H TOT HON R1" if ok else "R1H KHONG CAI THIEN"
    d1 = diffs["D1"]
    d1_pass = d1["ap_pos_T"]["lo"] > 0 and d1["ap_pos_T"]["delta"] > d1["ap_pos_T"]["nb_over_sqrt_k"]
    guard = d1["ap_pos_nonT"]["delta"] >= -d1["ap_pos_nonT"]["nb_over_sqrt_k"]
    d2_pass = has_c and diffs["D2"]["ap_pos_T"]["lo"] > 0
    if d1_pass and guard and d2_pass:
        verdict = "QUA: lateral giup va dac thu lateral (D1, rao chan, D2 deu qua)"
    elif d1_pass and guard:
        verdict = "PARTIAL: loi ich chu yeu den tu viec ghep them mot mo hinh"
    else:
        verdict = "KHONG QUA"

    def a_table(arm):
        """Mean over seeds and folds of a[f, E] per finding, for the arm's experts."""
        members = specs.get(arm, [])
        if arm_gate.get(arm, "none") != "none":
            members = [f"{m}:a0" for m in members] + [f"{m}:a1_entropy" for m in members]
        A = np.mean([np.mean([prm[f][0] for f in (0, 1)], 0) for prm in params[arm]], 0)  # [F, k]
        return {name: {m: float(A[j, e]) for e, m in enumerate(members)} for j, name in enumerate(names)}

    return {
        "n_pl": int(n), "fold_sizes": [int((fold == 0).sum()), int((fold == 1).sum())],
        "fit_on": "val_crossfit" if train is None else train.get("fit_on", "train_pl_oof"),
        "cutpoints": "fit_rows" if cutpoints_from_train else "crossfit_scored_rows",
        "n_train_pl": None if train is None else int(len(train["labels"])),
        "T": list(T_FINDINGS), "lambda": lam, "a_max": A_MAX, "k_seeds": k,
        "arms": {a: {"per_seed": v, "mean": {m: float(np.mean([x[m] for x in v])) for m in v[0]},
                     "sd": {m: float(np.std([x[m] for x in v], ddof=1)) if len(v) > 1 else 0.0 for m in v[0]}}
                 for a, v in point.items()},
        "cutpoint_pl": {a: {m: float(np.mean([x[m] for x in v])) for m in v[0]} for a, v in dec.items()},
        "diffs": diffs,
        "a_per_finding": {arm: a_table(arm) for arm in specs},
        "rule": {"D1_pass": bool(d1_pass), "guardrail_pass": bool(guard), "D2_pass": bool(d2_pass),
                 "views_1_equal_OFF": True},
        "verdict": verdict,
        "gate": gate, "rule_gate": rule_gate,
        "_probs": {a: ps for a, ps in arms.items()},
        "_decisions": decisions,
    }


def _load_expert(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as h:
        labels = h["labels"].astype(np.int64)
        labels[labels < 0] = MISSING
        return h["keys"], h["probabilities"].astype(np.float64), labels


def load_expert_parts(spec: str, mode: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One seed's expert from ``a.npz+b.npz+...``. ``mode="mean"``: same studies in every part, probabilities
    averaged (val scored by the fold models). ``mode="concat"``: disjoint studies, rows concatenated
    (out-of-fold train). A single path is loaded as is in either mode."""
    parts = [_load_expert(Path(p)) for p in spec.split("+")]
    if len(parts) == 1:
        return parts[0]
    if mode == "mean":
        k0, _p0, l0 = parts[0]
        for k, _p, lab in parts[1:]:
            if not (np.array_equal(k, k0) and np.array_equal(lab, l0)):
                raise SystemExit(f"{spec}: parts to average hold other studies or labels")
        return k0, np.mean([p for _k, p, _l in parts], 0), l0
    if mode == "concat":
        keys = np.concatenate([k for k, _p, _l in parts])
        if len(set(keys.tolist())) != len(keys):
            raise SystemExit(f"{spec}: out-of-fold parts overlap")
        return keys, np.concatenate([p for _k, p, _l in parts]), np.concatenate([lab for _k, _p, lab in parts])
    raise ValueError(mode)


def main(argv=None) -> int:
    from training.evaluation.schemas import ClassificationPredictions
    from training.evaluation.threshold_calibration import apply_cutpoints, load_cutpoint_file
    from training.evaluation.view_keys import PL, ViewMap, align

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--off", type=Path, required=True, help="OFF val predictions npz")
    ap.add_argument("--view-map", type=Path, required=True, help="view_map.npz of the same split")
    ap.add_argument("--off-cutpoints", type=Path, required=True, help="OFF's cutpoint file (non-PL decisions)")
    ap.add_argument("--expert", action="append", required=True, help="NAME=seed1.npz,seed2.npz,... (NAME L or C)")
    ap.add_argument("--expect-counts", default="PL=797,F1=930,other=81")
    ap.add_argument("--train-off", type=Path, help="B2a: OFF train predictions npz (fit R1 on train PL)")
    ap.add_argument("--train-expert", action="append", default=[],
                    help="B2a: NAME=s16_f0_oof.npz+s16_f1_oof.npz,... (same names and seed order as --expert)")
    ap.add_argument("--train-parts", choices=("concat", "mean"), default="concat",
                    help="how '+' joins --train-expert parts: concat = out-of-fold train (B2a), "
                         "mean = fold models on val (Step 3, fit on all val PL)")
    ap.add_argument("--cutpoints-from-train", action="store_true",
                    help="Step 3: PL cutpoints fitted on the fit rows (val) and applied to the scored rows (test)")
    ap.add_argument("--fit-split-name", default="train_pl_oof", help="recorded as fit_on in the JSON")
    ap.add_argument("--scored-split", choices=("val", "test"), default="val",
                    help="split of --off / --view-map / --expert (Step 3: test, read once)")
    ap.add_argument("--gate", choices=GATES, default="none",
                    help="entropy = R1H: weight a0 + a1 * H(p_OFF)/log3 per finding, compared with ungated R1 (DH)")
    ap.add_argument("--lam", type=float, default=LAM)
    ap.add_argument("--boot", type=int, default=BOOT_N)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")

    vmap = ViewMap.load(args.view_map)
    counts = vmap.counts()
    expected = {k: int(v) for k, v in (kv.split("=") for kv in args.expect_counts.split(","))}
    if counts != expected:
        raise SystemExit(f"group counts {counts} != expected {expected}")
    off = ClassificationPredictions.load(args.off)
    names = tuple(off.pathology_names)
    o = align(off.sample_keys, vmap.frontal_keys)  # raises unless every study is found
    p_all = np.asarray(off.probabilities, dtype=np.float64)[o]
    y_all = np.asarray(off.labels, dtype=np.int64)[o]
    y_all[y_all < 0] = MISSING
    pl = np.flatnonzero(vmap.groups == PL)
    pl_keys = vmap.frontal_keys[pl]

    experts: dict[str, list[np.ndarray]] = {}
    for spec in args.expert:
        name, paths = spec.split("=", 1)
        experts[name] = []
        for path in paths.split(","):
            keys, probs, labels = load_expert_parts(path, "mean")
            idx = align(keys, pl_keys)
            if not np.array_equal(labels[idx], y_all[pl]):
                raise SystemExit(f"labels of {path} differ from OFF's on PL studies")
            experts[name].append(probs[idx])
    if "L" not in experts:
        raise SystemExit("need --expert L=...")

    train = None
    if args.train_off is not None or args.train_expert:
        if args.train_off is None or not args.train_expert:
            raise SystemExit("--train-off and --train-expert go together")
        toff = ClassificationPredictions.load(args.train_off)
        if tuple(toff.pathology_names) != names:
            raise SystemExit("train OFF npz has another finding order")
        train = {"experts": {}}
        tr_keys = None
        for spec in args.train_expert:
            name, paths = spec.split("=", 1)
            train["experts"][name] = []
            for path in paths.split(","):
                keys, probs, labels = load_expert_parts(path, args.train_parts)
                if tr_keys is None:
                    tr_keys = keys
                    o_tr = align(toff.sample_keys, tr_keys)
                    y_tr = np.asarray(toff.labels, dtype=np.int64)[o_tr]
                    y_tr[y_tr < 0] = MISSING
                idx = align(keys, tr_keys)
                if not np.array_equal(labels[idx], y_tr):
                    raise SystemExit(f"labels of {path} differ from OFF's train labels")
                train["experts"][name].append(probs[idx])
        if set(train["experts"]) != set(experts):
            raise SystemExit("--train-expert names differ from --expert names")
        train.update(p_off=np.asarray(toff.probabilities, dtype=np.float64)[o_tr], labels=y_tr, keys=tr_keys,
                     fit_on=args.fit_split_name)
        # No cross-split key check: sample keys digest the split-local annotation row index
        # (batch["image_id"]), so train and val keys can coincide (217 of 797 on 2026-11-05) although
        # the splits are patient- and study-disjoint by construction.
        print(f"[residual] R1 fitted on {len(tr_keys)} PL studies ({args.fit_split_name})", flush=True)
    if args.cutpoints_from_train and train is None:
        raise SystemExit("--cutpoints-from-train needs --train-off / --train-expert")

    result = run(p_all[pl], y_all[pl], names, pl_keys, experts, lam=args.lam, boot_n=args.boot, train=train,
                 cutpoints_from_train=args.cutpoints_from_train, gate=args.gate)
    arm_probs = result.pop("_probs")
    arm_decisions = result.pop("_decisions")

    # whole val: non-PL studies keep OFF's probabilities and OFF's cutpoint decisions
    cut, _meta = load_cutpoint_file(args.off_cutpoints)
    off_dec = apply_cutpoints(p_all, cut, names)
    whole = {}
    for arm in ("OFF", "OFF+C", "OFF+L", "OFF+C+L", "OFF+L_R1"):
        if arm != "OFF" and arm not in result["arms"]:
            continue
        rows = []
        for p_pl, d_pl in zip(arm_probs[arm], arm_decisions[arm], strict=True):
            p = p_all.copy()
            p[pl] = p_pl
            assert np.array_equal(np.delete(p, pl, 0), np.delete(p_all, pl, 0))  # non-PL bit-identical to OFF
            d = off_dec.copy()
            d[pl] = d_pl
            t_idx = np.array([names.index(n) for n in T_FINDINGS])
            other_idx = np.array([j for j in range(len(names)) if j not in t_idx])
            rows.append({**summary(per_finding_scores(p, y_all), t_idx, other_idx),
                         **decision_metrics(d, y_all, names)})
        whole[arm] = {m: float(np.mean([r[m] for r in rows])) for m in rows[0]}
    result["whole_val" if args.scored_split == "val" else f"whole_{args.scored_split}"] = whole
    result["scored_split"] = args.scored_split
    result["name"], result["group_counts"] = args.name, counts

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"[residual] {args.name}: n_PL {result['n_pl']}, groups {counts}")
    for dname, d in result["diffs"].items():
        x = d["ap_pos_T"]
        print(f"  {dname:16s} AP_pos(T) {x['delta']:+.4f} [{x['lo']:+.4f}, {x['hi']:+.4f}]  "
              f"NB/sqrt(k) {x.get('nb_over_sqrt_k', float('nan')):.4f}")
    print(f"  verdict: {result['verdict']}")
    if result["rule_gate"] is not None:
        print(f"  gate {args.gate}: {result['rule_gate']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
