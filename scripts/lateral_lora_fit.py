#!/usr/bin/env python3
"""Step 1: adapt MedCLIP-Swin to one view with LoRA; L_frozen / L_lora / C_lora experts on PL studies.

Plan, gates and reading rule: ``docs/handoff/PLAN-2026-11-04-residual-lateral.md`` (Step 1,
gate G1). Input is ``scripts/lateral_lora_cache.py``'s cache (``<cache>/train``, ``<cache>/val``,
both views of the same PL studies). Every arm uses the same studies, the same 10% holdout
(``swin_finetune_fit.holdout_split``), the same head (``encoder_probe_fit.build_attention_probe``
on Swin's 49 grid tokens + pooled token), the same augmentation (train only, no flip), the same
schedule and budget; they differ only in the image and in whether LoRA trains:

* ``L_frozen``: lateral image, Swin frozen (control for the larger data and the augmentation);
* ``L_lora``:   lateral image, LoRA r=8 / alpha=16 / dropout 0.05 on query+value of stages 3-4;
* ``C_lora``:   frontal image, same LoRA (the frontal control of D2).

Fixed before any number: head LR 3e-4, LoRA LR 2e-4, weight decay 0.05 on both, batch 64, bf16,
at most 8 epochs with a warmup-cosine schedule over 8, early stop after 2 epochs without a better
holdout macro AP of "present" (the encoder probe's selection metric, not the deciding one).

``fit`` gates: G0 -- the cache's labels equal Step 0's feature dirs and frozen Swin on the cache
reproduces Step 0's Swin features (cosine >= 0.999 on 64 val studies, both views); G2 -- LoRA
tensors moved and every backbone tensor is bit-identical (frozen arm: all bit-identical). Writes
``<out>/<arm>_s<seed>.npz`` (``keys``, ``probabilities`` [N, 14, 3] on val PL, ``labels``), the
format ``residual_stack.py --expert`` reads, and ``report_<arm>_s<seed>.json``.

``compare --base NAME=a.npz,b.npz,c.npz --new NAME=x.npz,...``: G1(a), seed-mean difference of
mean AP Positive over T (F1-5) on val PL, paired study bootstrap (2,000, seed 16), NB from the
seeds; passes iff lower bound > 0 and delta > NB/sqrt(k). G1(b) is ``residual_stack.py`` with
``L=L_lora`` and ``C=C_lora``. Aggregates only; derived data stays on the host.

    python scripts/lateral_lora_fit.py fit --arm L_lora --seed 16 --cache-dir <c> --ref-dir <step0> --out-dir <o>
    python scripts/lateral_lora_fit.py compare --base L_frozen=<o>/L_frozen_s16.npz,... \\
        --new L_lora=<o>/L_lora_s16.npz,... --names-from <OFF val npz> --out <o>/G1a.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (_REPO_ROOT, _REPO_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from encoder_probe_fit import _masked_ce, build_attention_probe, macro_ap  # noqa: E402
from residual_stack import (  # noqa: E402
    BOOT_N,
    BOOT_SEED,
    T_FINDINGS,
    bootstrap_scores,
    nb,
    per_finding_scores,
)
from swin_finetune_fit import holdout_split, to_swin_input, warmup_cosine  # noqa: E402

from training.evaluation.view_keys import align  # noqa: E402

ARMS = {"L_frozen": ("lateral", False), "L_lora": ("lateral", True), "C_lora": ("frontal", True)}
STEP0_DIRS = {"lateral": "val_lat", "frontal": "val_front"}
LORA_R, LORA_ALPHA, LORA_DROPOUT = 8, 16, 0.05
HEAD_LR, LORA_LR, WEIGHT_DECAY = 3e-4, 2e-4, 0.05
EPOCHS, PATIENCE, BATCH, WARMUP_FRAC = 8, 2, 64, 0.05


def normalise(labels: np.ndarray) -> np.ndarray:
    out = np.asarray(labels).astype(np.int64).copy()
    out[out < 0] = -100
    return out


def should_stop(history: list[float], patience: int = PATIENCE) -> bool:
    """True once the last ``patience`` epochs all failed to beat the best epoch before them."""
    if len(history) <= patience:
        return False
    return max(history[-patience:]) <= max(history[:-patience])


def fold_of(keys, n_folds: int) -> np.ndarray:
    """Fold id of each study from a hash of its key: deterministic, independent of row order and of
    the seed, so every arm and seed sees the same split (L and C out-of-fold rows line up)."""
    import hashlib

    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")
    return np.asarray([int.from_bytes(hashlib.blake2b(str(k).encode(), digest_size=8).digest(), "big") % n_folds
                       for k in np.asarray(keys).tolist()], dtype=np.int64)


def fold_rows(keys, n_folds: int, fold: int) -> tuple[np.ndarray, np.ndarray]:
    """(rows the fold model trains on, rows it predicts out of fold)."""
    if not 0 <= fold < n_folds:
        raise ValueError(f"fold {fold} not in [0, {n_folds})")
    f = fold_of(keys, n_folds)
    return np.flatnonzero(f != fold), np.flatnonzero(f == fold)


def fit_holdout_rows(labels: np.ndarray, pool: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """``holdout_split`` on the rows a model may train on. ``pool=None`` is Step 1's split, unchanged."""
    if pool is None:
        return holdout_split(labels)
    fit, hold = holdout_split(labels[pool])
    return pool[fit], pool[hold]


def out_stem(arm: str, seed: int, n_folds: int, fold: int | None) -> str:
    return f"{arm}_s{seed}" if not n_folds else f"{arm}_s{seed}_f{fold}of{n_folds}"


def shared_rows(cache_keys: np.ndarray, ref_keys: np.ndarray, ref_is_pl_only: bool):
    """(cache rows, ref rows) of the studies both hold. A Step-0 lateral dir holds PL studies only,
    so every one of its keys must be in the cache (it is a subset of the same split's PL studies)."""
    pos = {k: i for i, k in enumerate(np.asarray(cache_keys).tolist())}
    pairs = [(pos[k], j) for j, k in enumerate(np.asarray(ref_keys).tolist()) if k in pos]
    if ref_is_pl_only and len(pairs) != len(ref_keys):
        raise SystemExit("G0 FAILED: a Step-0 PL study is missing from the cache")
    if not pairs:
        raise SystemExit("G0 FAILED: the cache shares no study with the Step-0 dir")
    a, b = zip(*pairs, strict=True)
    return np.asarray(a), np.asarray(b)


def _load(d: Path, view: str):
    return np.load(d / f"{view}.npy", mmap_mode="r"), normalise(np.load(d / "labels.npy")), np.load(d / "keys.npy")


def gate0(cache_dir: Path, ref_dir: Path, swin, device, n_check: int = 64) -> dict:
    """Cache labels = Step 0's; frozen Swin on the cache reproduces Step 0's Swin features."""
    import torch

    out = {}
    for split, refs in (("train", ("train_lat", "train_front")), ("val", ("val_lat", "val_front"))):
        _img, lab, keys = _load(cache_dir / split, "lateral")
        for ref in refs:
            rk, rl = np.load(ref_dir / ref / "keys.npy"), normalise(np.load(ref_dir / ref / "labels.npy"))
            common = shared_rows(keys, rk, ref.endswith("_lat"))
            if not np.array_equal(rl[common[1]], lab[common[0]]):
                raise SystemExit(f"G0 FAILED: {split} cache labels differ from {ref_dir / ref}")
            out[f"n_{split}_vs_{ref}"] = int(len(common[0]))
        out[f"n_{split}"] = int(len(keys))
    _img, _lab, keys = _load(cache_dir / "val", "lateral")
    swin.eval()
    for view, ref in STEP0_DIRS.items():
        img = np.load(cache_dir / "val" / f"{view}.npy", mmap_mode="r")
        idx = align(np.load(ref_dir / ref / "keys.npy"), keys[:n_check])
        g = np.load(ref_dir / ref / "swin_global.npy", mmap_mode="r")[idx].astype(np.float32)
        t = np.load(ref_dir / ref / "swin_grid.npy", mmap_mode="r")[idx].astype(np.float32)
        refv = torch.from_numpy(np.concatenate([g[:, None], t], 1)).flatten(1)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            new = swin(to_swin_input(torch.from_numpy(np.asarray(img[:n_check])).to(device))).float().cpu().flatten(1)
        cos = float(torch.nn.functional.cosine_similarity(new, refv, dim=1).min())
        out[f"cosine_min_{view}"] = cos
        print(f"[lora] G0 {view}: feature cosine min {cos:.5f} over {n_check} val studies", flush=True)
        if cos < 0.999:
            raise SystemExit(f"G0 FAILED: frozen Swin on the {view} cache does not reproduce Step 0's features")
    return out


def cmd_fit(args) -> int:
    import torch
    from swin_finetune_fit import _swin_weights

    from vision_encoders.lora import (
        SWIN_STAGE34_QV,
        apply_lora,
        base_state_dict,
        load_lora_state_dict,
        lora_modules,
        lora_parameters,
        lora_state_dict,
        set_lora_train,
    )
    from vision_encoders.swin.swin_encoder import SwinEncoder

    view, use_lora = ARMS[args.arm]
    if bool(args.n_folds) != (args.fold is not None):
        raise SystemExit("--n-folds and --fold go together")
    stem = out_stem(args.arm, args.seed, args.n_folds, args.fold)
    keep_best = bool(args.save_weights or args.n_folds)
    out_npz = args.out_dir / f"{stem}.npz"
    if out_npz.exists():
        raise SystemExit(f"refusing to overwrite {out_npz}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    model_name, weights = _swin_weights(args.config)
    swin = SwinEncoder(model_name=model_name, backend="medclip", normalize=False, weights_path=weights,
                       frozen=True).to(device)
    report: dict = {"arm": args.arm, "view": view, "seed": args.seed, "lora": use_lora,
                    "n_folds": args.n_folds, "fold": args.fold,
                    "hyper": {"r": LORA_R, "alpha": LORA_ALPHA, "dropout": LORA_DROPOUT, "head_lr": HEAD_LR,
                              "lora_lr": LORA_LR, "weight_decay": WEIGHT_DECAY, "epochs": args.epochs,
                              "patience": PATIENCE, "batch": BATCH},
                    "gate0": gate0(args.cache_dir, args.ref_dir, swin, device)}
    if use_lora:
        torch.manual_seed(args.seed)
        report["lora_targets"] = apply_lora(swin.model, SWIN_STAGE34_QV, LORA_R, LORA_ALPHA, LORA_DROPOUT)
    initial = base_state_dict(swin)

    tr_img, y_tr, tr_keys = _load(args.cache_dir / "train", view)
    va_img, y_val, va_keys = _load(args.cache_dir / "val", view)
    pool, oof = fold_rows(tr_keys, args.n_folds, args.fold) if args.n_folds else (None, None)
    fit, hold = fit_holdout_rows(y_tr, pool)
    print(f"[lora] {stem}: fit {len(fit)}, holdout {len(hold)}, out-of-fold {0 if oof is None else len(oof)}, "
          f"val {len(y_val)}, LoRA layers {len(report.get('lora_targets', []))}", flush=True)

    torch.manual_seed(args.seed)
    head = build_attention_probe([(49, swin.embed_dim)], [swin.embed_dim]).to(device)
    groups = [{"params": list(head.parameters()), "lr": HEAD_LR, "weight_decay": WEIGHT_DECAY}]
    if use_lora:
        groups.append({"params": lora_parameters(swin), "lr": LORA_LR, "weight_decay": WEIGHT_DECAY})
    trainable = {id(p) for g in groups for p in g["params"]}
    if any(p.requires_grad and id(p) not in trainable for p in swin.parameters()):
        raise SystemExit("a backbone tensor would train")
    report["n_trainable"] = {"head": sum(p.numel() for p in head.parameters()),
                             "lora": sum(p.numel() for p in lora_parameters(swin))}
    opt = torch.optim.AdamW(groups)
    steps_per_epoch = math.ceil(len(fit) / BATCH)
    epochs = args.epochs
    total = epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: warmup_cosine(s, total, max(1, int(WARMUP_FRAC * total))))
    gen = torch.Generator(device=device).manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    def tokens(images_u8, augmented):
        out = swin(to_swin_input(torch.from_numpy(np.asarray(images_u8)).to(device), gen if augmented else None))
        return [out[:, 1:]], [out[:, 0]]

    def predict(images, idx):
        swin.eval(), head.eval()
        set_lora_train(swin, False)
        res = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in np.array_split(idx, max(1, math.ceil(len(idx) / 256))):
                res.append(torch.softmax(head(*tokens(images[np.sort(i)], False)).float(), -1).cpu().numpy())
        return np.concatenate(res)

    history, best, started, snapshot = [], (-1.0, -1, None), time.time(), None
    for epoch in range(epochs):
        swin.train(), head.train()
        set_lora_train(swin, use_lora)
        losses = []
        for i in np.array_split(rng.permutation(fit), steps_per_epoch):
            i = np.sort(i)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if use_lora:
                    t, g = tokens(tr_img[i], True)
                else:
                    with torch.no_grad():
                        t, g = tokens(tr_img[i], True)
                logits = head(t, g)
            loss = _masked_ce(logits.float(), torch.as_tensor(y_tr[i], device=device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            losses.append(float(loss))
        hold_ap = macro_ap(predict(tr_img, hold), y_tr[hold])
        history.append(hold_ap)
        report.setdefault("history", []).append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                                                 "holdout_macro_ap": hold_ap,
                                                 "minutes": (time.time() - started) / 60})
        print(f"[lora] {args.arm} s{args.seed} epoch {epoch}: train loss {np.mean(losses):.4f}, holdout macro AP "
              f"{hold_ap:.4f} ({(time.time() - started) / 60:.1f} min)", flush=True)
        if hold_ap > best[0]:
            best = (hold_ap, epoch, predict(va_img, np.arange(len(y_val))))
            if keep_best:
                snapshot = ({k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
                            lora_state_dict(swin))
        if should_stop(history):
            break

    final = base_state_dict(swin)
    identical = all(torch.equal(final[k], initial[k]) for k in initial)
    rel = {n: float(m.delta_weight().norm() / m.base.weight.norm()) for n, m in lora_modules(swin)}
    moved = bool(rel) and min(rel.values()) > 0
    g2 = identical and (moved if use_lora else not rel)
    report.update(chosen_epoch=best[1], holdout_macro_ap=best[0], epochs_run=len(history),
                  max_mem_mib=torch.cuda.max_memory_allocated() / 2**20,
                  val_macro_ap_present=macro_ap(best[2], y_val), n_fit=int(len(fit)), n_hold=int(len(hold)),
                  n_val=int(len(y_val)),
                  gate2={"backbone_bit_identical": identical, "lora_relative_change_min": min(rel.values(), default=0.0),
                         "lora_relative_change_max": max(rel.values(), default=0.0), "passed": g2})
    if keep_best:
        # Back to the selected epoch's weights. G3: they reproduce the saved val probabilities.
        head.load_state_dict(snapshot[0])
        load_lora_state_dict(swin, snapshot[1])
        check = np.arange(min(256, len(y_val)))
        g3 = float(np.abs(predict(va_img, check) - best[2][check]).max())
        report["gate3_max_abs_dprob"] = g3
        if g3 > 1e-4:
            raise SystemExit(f"G3 FAILED: restored weights move val probabilities by {g3:.2e}")
        if oof is not None:
            p_oof = predict(tr_img, oof)
            np.savez(args.out_dir / f"{stem}_oof.npz", keys=tr_keys[oof], probabilities=p_oof.astype(np.float32),
                     labels=y_tr[oof])
            report.update(n_oof=int(len(oof)), oof_macro_ap_present=macro_ap(p_oof, y_tr[oof]))
        if args.save_weights:
            import torch as _t
            _t.save({"arm": args.arm, "seed": args.seed, "n_folds": args.n_folds, "fold": args.fold,
                     "chosen_epoch": best[1], "hyper": report["hyper"], "lora_targets": report.get("lora_targets", []),
                     "head": snapshot[0], "lora": snapshot[1]}, args.out_dir / f"{stem}_weights.pt")
    np.savez(out_npz, keys=va_keys, probabilities=best[2].astype(np.float32), labels=y_val)
    (args.out_dir / f"report_{stem}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"[lora] {args.arm} s{args.seed}: chosen epoch {best[1]}, holdout {best[0]:.4f}, val macro AP present "
          f"{report['val_macro_ap_present']:.4f}, max mem {report['max_mem_mib']:.0f} MiB; G2 "
          f"{'passed' if g2 else 'FAILED'} (bit-identical {identical}, LoRA change "
          f"{report['gate2']['lora_relative_change_min']:.2e}-{report['gate2']['lora_relative_change_max']:.2e})",
          flush=True)
    if not g2:
        raise SystemExit("G2 FAILED")
    return 0


def load_set(spec: str):
    """``NAME=a.npz,b.npz`` -> (name, keys, [probabilities per seed], labels)."""
    name, paths = spec.split("=", 1)
    keys, probs, labels = None, [], None
    for p in paths.split(","):
        with np.load(p, allow_pickle=False) as h:
            k, lab = h["keys"], normalise(h["labels"])
            if keys is None:
                keys, labels = k, lab
            elif not (np.array_equal(k, keys) and np.array_equal(lab, labels)):
                raise SystemExit(f"{p}: other studies or labels than the first seed of {name}")
            probs.append(h["probabilities"].astype(np.float64))
    return name, keys, probs, labels


def compare(base, new, labels, names, boot_n=BOOT_N, boot_seed=BOOT_SEED) -> dict:
    """G1(a): seed-mean AP_pos(T) of ``new`` minus ``base``, CI, NB/sqrt(k), verdict."""
    t_idx = np.array([list(names).index(n) for n in T_FINDINGS])
    lab = labels.copy()
    lab[lab < 0] = -1
    with np.errstate(all="ignore"):
        vb = [float(np.nanmean(per_finding_scores(p, lab)["ap_pos"][t_idx])) for p in new]
        va = [float(np.nanmean(per_finding_scores(p, lab)["ap_pos"][t_idx])) for p in base]
        rng = np.random.default_rng(boot_seed)
        n = len(lab)
        d = np.empty(boot_n)
        for r in range(boot_n):
            i = rng.integers(0, n, n)
            mb = np.mean([np.nanmean(bootstrap_scores(p, lab, i)[0][t_idx]) for p in new])
            ma = np.mean([np.nanmean(bootstrap_scores(p, lab, i)[0][t_idx]) for p in base])
            d[r] = mb - ma
    lo, hi = np.nanpercentile(d, [2.5, 97.5])
    delta = float(np.mean(vb) - np.mean(va))
    bar = nb([vb, va]) / math.sqrt(min(len(vb), len(va)))
    return {"delta": delta, "lo": float(lo), "hi": float(hi), "nb_over_sqrt_k": bar,
            "base_per_seed": va, "new_per_seed": vb, "passed": bool(lo > 0 and delta > bar)}


def cmd_compare(args) -> int:
    from training.evaluation.schemas import ClassificationPredictions

    names = tuple(ClassificationPredictions.load(args.names_from).pathology_names)
    bn, bk, bp, bl = load_set(args.base)
    nn_, nk, np_, nl = load_set(args.new)
    if not (np.array_equal(bk, nk) and np.array_equal(bl, nl)):
        raise SystemExit("the two sets were scored on different studies")
    res = compare(bp, np_, bl, names, boot_n=args.boot)
    res.update(base=bn, new=nn_, n=int(len(bk)), T=list(T_FINDINGS))
    print(f"[lora] G1a {nn_} - {bn}: AP_pos(T) {res['delta']:+.4f} [{res['lo']:+.4f}, {res['hi']:+.4f}], "
          f"NB/sqrt(k) {res['nb_over_sqrt_k']:.4f} -> {'PASS' if res['passed'] else 'FAIL'}", flush=True)
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    args.out.write_text(json.dumps(res, indent=2) + "\n")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--arm", required=True, choices=tuple(ARMS))
    f.add_argument("--seed", type=int, required=True)
    f.add_argument("--cache-dir", type=Path, required=True)
    f.add_argument("--ref-dir", type=Path, required=True, help="Step 0 output dir (val_lat, val_front, ...)")
    f.add_argument("--config", type=Path, default=_REPO_ROOT / "pretraining/configs/mimic_cxr_full.yaml")
    f.add_argument("--epochs", type=int, default=EPOCHS, help="smoke only; the reading rule uses 8")
    f.add_argument("--out-dir", type=Path, required=True)
    f.add_argument("--n-folds", type=int, default=0, help="B1: cross-fit over train PL (0 = Step 1, no folds)")
    f.add_argument("--fold", type=int, default=None, help="B1: which fold to predict out of fold")
    f.add_argument("--save-weights", action="store_true", help="B1: save the selected epoch's head + LoRA")
    c = sub.add_parser("compare")
    c.add_argument("--base", required=True)
    c.add_argument("--new", required=True)
    c.add_argument("--names-from", type=Path, required=True, help="OFF val npz (finding order)")
    c.add_argument("--boot", type=int, default=BOOT_N)
    c.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    return cmd_fit(args) if args.cmd == "fit" else cmd_compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
