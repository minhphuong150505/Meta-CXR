#!/usr/bin/env python3
"""Linear and attention probes on per-encoder features; reading rule of the encoder probe.

Reading rule: ``docs/handoff/PLAN-2026-10-31-encoder-probe.md``. Features come from
``scripts/encoder_probe_features.py`` (one directory per split). Every probe is fit on
the train features (10% held out to pick L2 / epoch), scored once on val:

* linear    -- per-finding 3-class softmax on the standardised global vector;
* attention -- LayerNorm -> 256-d projection + learned positions per encoder, the global
               vector as one extra token, 14 learned queries (one per finding), one-head
               attention, linear 3-class head. Mean probabilities of ``--seeds`` fits.

Deciding metric: macro AP of "present" (P or U vs N) on s = p_pos/(p_pos+p_neg), mean of
14 findings. Encoder X is a CANDIDATE iff attention(X) - attention(swin) >= +0.02 or
attention(swin+X) - attention(swin) >= +0.02, paired study bootstrap lower bound > 0.
Gate G0: the metric recomputed from MHCAC's stored val npz must give the recorded values.
Prints aggregates only.

    python scripts/encoder_probe_fit.py --train-dir <dir> --val-dir <dir> \\
        --mhcac C42=<val npz> --mhcac R1C=<val npz> --report <dir>/report.json
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

ENCODERS = ("swin", "pubmedclip", "biovil", "raddino", "medsiglip")
CANDIDATE_MARGIN = 0.02
RARE = ("Enlarged Cardiomediastinum", "Fracture", "Pleural Other", "Lung Lesion", "Consolidation",
        "Pneumothorax")


def ap_auc(y: np.ndarray, s: np.ndarray) -> tuple[float, float]:
    """AP and AUROC of score ``s`` for boolean ``y``, ties evaluated at the end of each run.

    Same arithmetic as the recall pilots' ``recall_metrics.ap_auc_rp``.
    """
    o = np.argsort(-s, kind="stable")
    ys, ss = y[o], s[o]
    tp = np.cumsum(ys)
    k = np.arange(1, len(ys) + 1)
    last = np.r_[ss[1:] != ss[:-1], True]
    tp, k = tp[last], k[last]
    p = ys.sum()
    if p == 0 or p == len(ys):
        return float("nan"), float("nan")
    prec, rec = tp / k, tp / p
    ap = float((prec * np.diff(np.r_[0.0, rec])).sum())
    tpr, fpr = np.r_[0.0, rec], np.r_[0.0, (k - tp) / (len(ys) - p)]
    return ap, float(np.trapezoid(tpr, fpr) if hasattr(np, "trapezoid") else np.trapz(tpr, fpr))


def presence_scores(probs: np.ndarray) -> np.ndarray:
    """``p_pos / (p_pos + p_neg)`` per cell; probs ``[N, 14, 3]`` in (neg, pos, unc) order."""
    return probs[..., 1] / np.maximum(probs[..., 1] + probs[..., 0], 1e-12)


def per_finding(probs: np.ndarray, labels: np.ndarray, idx=None) -> tuple[np.ndarray, np.ndarray]:
    """Per-finding AP and AUROC of "present" over labelled cells."""
    if idx is not None:
        probs, labels = probs[idx], labels[idx]
    s = presence_scores(probs)
    ap, auc = [], []
    for j in range(labels.shape[1]):
        m = labels[:, j] >= 0
        a, u = ap_auc(np.isin(labels[m, j], (1, 2)), s[m, j])
        ap.append(a), auc.append(u)
    return np.array(ap), np.array(auc)


def macro_ap(probs, labels, idx=None) -> float:
    return float(np.nanmean(per_finding(probs, labels, idx)[0]))


def paired_bootstrap(pa, pb, labels, n=1000, seed=16):
    """macro AP(b) - macro AP(a), with a 95% paired study bootstrap interval."""
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        i = rng.integers(0, len(labels), len(labels))
        d.append(macro_ap(pb, labels, i) - macro_ap(pa, labels, i))
    lo, hi = np.percentile(d, [2.5, 97.5])
    return macro_ap(pb, labels) - macro_ap(pa, labels), float(lo), float(hi)


def _masked_ce(logits, labels):
    import torch.nn.functional as F

    return F.cross_entropy(logits.reshape(-1, 3), labels.reshape(-1).long(), ignore_index=-100)


def build_attention_probe(token_shapes, global_dims, d=256, n_findings=14, dropout=0.1):
    import torch
    from torch import nn

    class _Probe(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok = nn.ModuleList(nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, d))
                                     for _t, dim in token_shapes)
            self.glob = nn.ModuleList(nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, d)) for dim in global_dims)
            seq = sum(t for t, _ in token_shapes) + len(global_dims)
            self.pos = nn.Parameter(torch.randn(seq, d) * 0.02)
            self.query = nn.Parameter(torch.randn(n_findings, d) * 0.02)
            self.norm = nn.LayerNorm(d)
            self.k, self.v = nn.Linear(d, d), nn.Linear(d, d)
            self.drop = nn.Dropout(dropout)
            self.head_w = nn.Parameter(torch.randn(n_findings, d, 3) * 0.02)
            self.head_b = nn.Parameter(torch.zeros(n_findings, 3))

        def forward(self, tokens, globals_):
            parts = [p(t.float()) for p, t in zip(self.tok, tokens)]
            parts += [p(g.float()).unsqueeze(1) for p, g in zip(self.glob, globals_)]
            x = self.norm(torch.cat(parts, 1) + self.pos)
            att = torch.softmax(self.query @ self.k(x).transpose(1, 2) / d ** 0.5, -1)  # [B, F, S]
            z = self.drop(att @ self.v(x))  # [B, F, d]
            return torch.einsum("bfd,fdc->bfc", z, self.head_w) + self.head_b

    return _Probe()


def _softmax_np(logits):
    e = np.exp(logits - logits.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def fit_linear(g_fit, y_fit, g_hold, y_hold, g_val, device, epochs=30, seed=16):
    import torch

    mu, sd = g_fit.mean(0), g_fit.std(0) + 1e-6
    T = lambda a: torch.as_tensor((a - mu) / sd, dtype=torch.float32, device=device)  # noqa: E731
    xf, xh, xv = T(g_fit), T(g_hold), T(g_val)
    yf = torch.as_tensor(y_fit, device=device)
    best = (-1.0, None, None)
    for wd in (1e-4, 1e-3, 1e-2, 1e-1):
        torch.manual_seed(seed)
        lin = torch.nn.Linear(xf.shape[1], 14 * 3).to(device)
        opt = torch.optim.AdamW(lin.parameters(), lr=1e-3, weight_decay=wd)
        for _ in range(epochs):
            for i in torch.randperm(len(xf), device=device).split(1024):
                opt.zero_grad()
                _masked_ce(lin(xf[i]).view(-1, 14, 3), yf[i]).backward()
                opt.step()
            with torch.no_grad():
                ph = _softmax_np(lin(xh).view(-1, 14, 3).cpu().numpy())
            score = macro_ap(ph, y_hold)
            if score > best[0]:
                with torch.no_grad():
                    best = (score, wd, _softmax_np(lin(xv).view(-1, 14, 3).cpu().numpy()))
    return best


def fit_attention(feats, names, fit, hold, y_tr, val_feats, y_val, device, seeds=(16, 17, 18), epochs=12):
    import torch

    tok_shapes = [feats[f"{n}_grid"].shape[1:] for n in names]
    g_dims = [feats[f"{n}_global"].shape[1] for n in names]

    def batches(src, idx):
        return ([torch.from_numpy(np.asarray(src[f"{n}_grid"][idx])).to(device) for n in names],
                [torch.from_numpy(np.asarray(src[f"{n}_global"][idx])).to(device) for n in names])

    def predict(model, src, idx):
        model.eval()
        out = []
        with torch.no_grad():
            for i in np.array_split(idx, max(1, len(idx) // 512)):
                t, g = batches(src, i)
                out.append(torch.softmax(model(t, g), -1).cpu().numpy())
        return np.concatenate(out)

    val_idx = np.arange(len(y_val))
    probs, holds = [], []
    for seed in seeds:
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        model = build_attention_probe(tok_shapes, g_dims).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
        steps = epochs * int(np.ceil(len(fit) / 256))
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, total_steps=steps, pct_start=0.1)
        best = (-1.0, None)
        for _ in range(epochs):
            model.train()
            for i in np.array_split(rng.permutation(fit), int(np.ceil(len(fit) / 256))):
                i = np.sort(i)
                t, g = batches(feats, i)
                opt.zero_grad()
                _masked_ce(model(t, g), torch.as_tensor(y_tr[i], device=device)).backward()
                opt.step()
                sched.step()
            score = macro_ap(predict(model, feats, hold), y_tr[hold])
            if score > best[0]:
                best = (score, predict(model, val_feats, val_idx))
        holds.append(best[0])
        probs.append(best[1])
    return float(np.mean(holds)), np.mean(probs, 0)


def _load(dirpath: Path, names):
    out = {"labels": np.load(dirpath / "labels.npy"), "keys": np.load(dirpath / "keys.npy")}
    for n in names:
        for part in ("global", "grid"):
            out[f"{n}_{part}"] = np.load(dirpath / f"{n}_{part}.npy", mmap_mode="r")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-dir", type=Path, required=True)
    ap.add_argument("--val-dir", type=Path, required=True)
    ap.add_argument("--encoders", default=",".join(ENCODERS))
    ap.add_argument("--mhcac", action="append", default=[], help="NAME=val npz (reference + gate G0)")
    ap.add_argument("--expect", action="append", default=[], help="NAME=recorded macro AP for gate G0")
    ap.add_argument("--report", type=Path, required=True)
    args = ap.parse_args(argv)

    import torch

    from training.evaluation.schemas import ClassificationPredictions

    names = [n for n in args.encoders.split(",") if n]
    assert names[0] == "swin", "swin is the reference and must come first"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr, va = _load(args.train_dir, names), _load(args.val_dir, names)
    y_tr, y_val = tr["labels"], va["labels"]
    labelled = np.flatnonzero((y_tr >= 0).any(1))
    perm = np.random.default_rng(16).permutation(labelled)
    hold, fit = np.sort(perm[: len(perm) // 10]), np.sort(perm[len(perm) // 10:])
    print(f"[probe] train fit {len(fit)}, holdout {len(hold)}, val {len(y_val)}", flush=True)
    finding_names = None
    report: dict = {"n_fit": int(len(fit)), "n_hold": int(len(hold)), "n_val": int(len(y_val))}

    # Gate G0 and MHCAC reference, aligned on the eval hook's salted keys.
    expect = dict(e.split("=", 1) for e in args.expect)
    ref_probs = {}
    pos = {k: i for i, k in enumerate(va["keys"].tolist())}
    for spec in args.mhcac:
        name, path = spec.split("=", 1)
        p = ClassificationPredictions.load(path)
        finding_names = list(p.pathology_names)
        value = macro_ap(p.probabilities, p.labels)
        print(f"[probe] G0 {name}: macro AP {value:.4f} (recorded {expect.get(name, '?')})", flush=True)
        if name in expect and abs(value - float(expect[name])) > 5e-5:
            raise SystemExit(f"G0 FAILED for {name}")
        order = np.array([pos[k] for k in p.sample_keys.tolist()])
        # "No label" is -1 in the eval hook's npz and -100 here; compare the labelled cells exactly.
        if not np.array_equal(np.where(p.labels < 0, -1, p.labels), np.where(y_val[order] < 0, -1, y_val[order])):
            raise SystemExit(f"G0 FAILED: {name} labels differ from the probe's val labels")
        aligned = np.empty_like(p.probabilities)
        aligned[order] = p.probabilities
        ref_probs[f"mhcac_{name}"] = aligned
    print("[probe] G0 passed", flush=True)

    probs = dict(ref_probs)
    for n in names:
        _hs, _wd, pv = fit_linear(np.asarray(tr[f"{n}_global"][fit], np.float32), y_tr[fit],
                                  np.asarray(tr[f"{n}_global"][hold], np.float32), y_tr[hold],
                                  np.asarray(va[f"{n}_global"], np.float32), device)
        probs[f"linear_{n}"] = pv
        print(f"[probe] linear {n}: holdout {_hs:.4f} (wd {_wd:g}), val macro AP {macro_ap(pv, y_val):.4f}",
              flush=True)
    combos = [(n,) for n in names] + [("swin", n) for n in names[1:]]
    combos += [tuple(c) for c in (("swin", "biovil", "pubmedclip"), tuple(names)) if set(c) <= set(names)]
    for c in dict.fromkeys(combos):
        hs, pv = fit_attention(tr, list(c), fit, hold, y_tr, va, y_val, device)
        key = "attn_" + "+".join(c)
        probs[key] = pv
        print(f"[probe] {key}: holdout {hs:.4f}, val macro AP {macro_ap(pv, y_val):.4f}", flush=True)

    table = {}
    for k, pv in probs.items():
        apf, aucf = per_finding(pv, y_val)
        table[k] = {"macro_ap": float(np.nanmean(apf)), "auroc_present": float(np.nanmean(aucf)),
                    "ap_per_finding": apf.tolist()}
    report["table"] = table
    print("\n[probe] val, macro AP / AUROC present:")
    for k, v in table.items():
        print(f"  {k:36s} {v['macro_ap']:.4f} / {v['auroc_present']:.4f}")
    if finding_names:
        print("\n[probe] AP per finding (rare marked *): " + " | ".join(table))
        for j, f in enumerate(finding_names):
            print(f"  {f + ('*' if f in RARE else ''):30s} " + " ".join(f"{v['ap_per_finding'][j]:.3f}"
                                                                      for v in table.values()))

    print("\n[probe] reading rule (attention probe, macro AP, vs attn_swin):")
    report["candidates"], report["comparisons"] = [], {}
    base = probs["attn_swin"]
    for n in names[1:]:
        for key in (f"attn_{n}", f"attn_swin+{n}"):
            d, lo, hi = paired_bootstrap(base, probs[key], y_val)
            report["comparisons"][key] = {"delta": d, "lo": lo, "hi": hi}
            ok = d >= CANDIDATE_MARGIN and lo > 0
            print(f"  {key:28s} d {d:+.4f} [{lo:+.4f}, {hi:+.4f}] {'CANDIDATE' if ok else ''}")
            if ok and n not in report["candidates"]:
                report["candidates"].append(n)
    for k in ref_probs:
        d, lo, hi = paired_bootstrap(probs[k], base, y_val)
        report["comparisons"][f"attn_swin-{k}"] = {"delta": d, "lo": lo, "hi": hi}
        print(f"  attn_swin - {k:16s} d {d:+.4f} [{lo:+.4f}, {hi:+.4f}] (descriptive)")
    print("VERDICT:", ("CANDIDATES " + ", ".join(report["candidates"])) if report["candidates"]
          else "NO CANDIDATE")
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
