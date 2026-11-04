#!/usr/bin/env python3
"""Lateral expert ``L_enc`` and its frontal control ``C_enc`` on PL studies, one npz per seed.

Plan and reading rule: ``docs/handoff/PLAN-2026-11-04-residual-lateral.md`` (Step 0).
Both experts are the encoder probe's attention head (``encoder_probe_fit.build_attention_probe``)
on ONE frozen encoder, three classes, trained with the study's labels on the SAME PL train
studies, with the same 10% holdout and the same recipe as ``encoder_probe_fit.fit_attention``
(AdamW 3e-4, wd 0.05, OneCycle, batch 256, 12 epochs, epoch chosen on the holdout's macro AP
of "present"). They differ only in the image:

* ``--view lateral`` (L): features from ``encoder_probe_features.py --views lateral``;
* ``--view frontal`` (C): the encoder probe's frontal features, restricted to the same studies.

The study set is always the lateral directories' ``keys.npy`` (frontal-anchor keys of the PL
studies); frontal features are aligned to it by key. Gate G0: the two directories' labels
must agree on every study (they are study labels), else nothing is fit.

Writes ``<out>/<encoder>_<view>_s<seed>.npz`` (``keys``, ``probabilities`` [N, 14, 3] on val
PL, ``labels`` with -100 = no label) and ``report.json``. Derived data: keep on the host.
Prints aggregates only.

    python scripts/lateral_expert_fit.py --encoder swin --view lateral \\
        --lateral-train <dir> --lateral-val <dir> --frontal-train <dir> --frontal-val <dir> \\
        --out-dir <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (_REPO_ROOT, _REPO_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from encoder_probe_fit import ENCODERS, _masked_ce, build_attention_probe, macro_ap  # noqa: E402

from training.evaluation.view_keys import align  # noqa: E402

EPOCHS, BATCH, LR, WEIGHT_DECAY = 12, 256, 3e-4, 0.05


def holdout_split(labels: np.ndarray, seed: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """Same as ``swin_finetune_fit.holdout_split``: 10% of labelled studies held out."""
    labelled = np.flatnonzero((labels >= 0).any(1))
    perm = np.random.default_rng(seed).permutation(labelled)
    return np.sort(perm[len(perm) // 10:]), np.sort(perm[: len(perm) // 10])


def normalise_missing(labels: np.ndarray) -> np.ndarray:
    """-100 for every missing cell (the eval hook writes -1, the probe -100)."""
    out = np.asarray(labels).astype(np.int64).copy()
    out[out < 0] = -100
    return out


def load_view(dirpath: Path, encoder: str, target_keys: np.ndarray) -> dict:
    """Labels and the encoder's features of ``dirpath``, rows aligned to ``target_keys``."""
    idx = align(np.load(dirpath / "keys.npy"), target_keys)
    out = {"labels": normalise_missing(np.load(dirpath / "labels.npy")[idx])}
    for part in ("global", "grid"):
        out[f"{encoder}_{part}"] = np.asarray(np.load(dirpath / f"{encoder}_{part}.npy", mmap_mode="r")[idx])
    return out


def check_labels(a: np.ndarray, b: np.ndarray, what: str) -> None:
    if a.shape != b.shape or not np.array_equal(a, b):
        raise SystemExit(f"G0 FAILED ({what}): lateral and frontal labels differ")


def fit_one_seed(feats, name, fit, hold, y_tr, val_feats, device, seed, epochs=EPOCHS):
    """One attention-probe fit; returns (holdout history, chosen epoch, val probabilities)."""
    import torch

    tok_shapes = [feats[f"{name}_grid"].shape[1:]]
    g_dims = [feats[f"{name}_global"].shape[1]]

    def batches(src, idx):
        return ([torch.from_numpy(src[f"{name}_grid"][idx]).to(device)],
                [torch.from_numpy(src[f"{name}_global"][idx]).to(device)])

    def predict(model, src, idx):
        model.eval()
        out = []
        with torch.no_grad():
            for i in np.array_split(idx, max(1, len(idx) // 512)):
                t, g = batches(src, i)
                out.append(torch.softmax(model(t, g), -1).cpu().numpy())
        return np.concatenate(out)

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = build_attention_probe(tok_shapes, g_dims).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    n_batches = int(np.ceil(len(fit) / BATCH))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=epochs * n_batches, pct_start=0.1)
    val_idx = np.arange(len(val_feats[f"{name}_global"]))
    history, best = [], (-1.0, -1, None)
    for epoch in range(epochs):
        model.train()
        for i in np.array_split(rng.permutation(fit), n_batches):
            i = np.sort(i)
            t, g = batches(feats, i)
            opt.zero_grad()
            _masked_ce(model(t, g), torch.as_tensor(y_tr[i], device=device)).backward()
            opt.step()
            sched.step()
        score = macro_ap(predict(model, feats, hold), y_tr[hold])
        history.append(score)
        if score > best[0]:
            best = (score, epoch, predict(model, val_feats, val_idx))
    return history, best[1], best[2]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", required=True, choices=ENCODERS)
    ap.add_argument("--view", required=True, choices=("lateral", "frontal"))
    ap.add_argument("--lateral-train", type=Path, required=True)
    ap.add_argument("--lateral-val", type=Path, required=True)
    ap.add_argument("--frontal-train", type=Path, required=True)
    ap.add_argument("--frontal-val", type=Path, required=True)
    ap.add_argument("--seeds", default="16,17,18")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args(argv)
    seeds = [int(s) for s in args.seeds.split(",")]

    import torch

    keys_tr = np.load(args.lateral_train / "keys.npy")
    keys_val = np.load(args.lateral_val / "keys.npy")
    lat_tr = load_view(args.lateral_train, args.encoder, keys_tr) if args.view == "lateral" else None
    lat_val = load_view(args.lateral_val, args.encoder, keys_val) if args.view == "lateral" else None
    fr_tr_labels = normalise_missing(np.load(args.frontal_train / "labels.npy")[align(np.load(args.frontal_train / "keys.npy"), keys_tr)])
    fr_val_labels = normalise_missing(np.load(args.frontal_val / "labels.npy")[align(np.load(args.frontal_val / "keys.npy"), keys_val)])
    check_labels(normalise_missing(np.load(args.lateral_train / "labels.npy")), fr_tr_labels, "train")
    check_labels(normalise_missing(np.load(args.lateral_val / "labels.npy")), fr_val_labels, "val")
    if args.view == "frontal":
        lat_tr = load_view(args.frontal_train, args.encoder, keys_tr)
        lat_val = load_view(args.frontal_val, args.encoder, keys_val)
    feats, val_feats = lat_tr, lat_val
    y_tr = feats["labels"]
    fit, hold = holdout_split(y_tr)
    print(f"[expert] {args.encoder}/{args.view}: train PL {len(y_tr)} (fit {len(fit)}, hold {len(hold)}), "
          f"val PL {len(keys_val)}; G0 labels agree", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = {"encoder": args.encoder, "view": args.view, "n_train": int(len(y_tr)), "n_fit": int(len(fit)),
              "n_hold": int(len(hold)), "n_val": int(len(keys_val)), "epochs": args.epochs, "seeds": {}}
    for seed in seeds:
        out = args.out_dir / f"{args.encoder}_{args.view}_s{seed}.npz"
        if out.exists():
            raise SystemExit(f"refusing to overwrite {out}")
        started = time.time()
        history, epoch, probs = fit_one_seed(feats, args.encoder, fit, hold, y_tr, val_feats, device, seed, args.epochs)
        np.savez(out, keys=keys_val, probabilities=probs.astype(np.float32), labels=val_feats["labels"])
        val_ap = macro_ap(probs, val_feats["labels"])
        report["seeds"][str(seed)] = {"holdout_macro_ap_present": history, "chosen_epoch": epoch,
                                      "val_macro_ap_present": val_ap, "seconds": round(time.time() - started, 1)}
        print(f"[expert] seed {seed}: epoch {epoch}, holdout {max(history):.4f}, "
              f"val macro AP present {val_ap:.4f}", flush=True)
    rp = args.out_dir / f"report_{args.encoder}_{args.view}.json"
    rp.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
