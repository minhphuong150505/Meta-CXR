#!/usr/bin/env python3
"""Swin fine-tune probe: frozen (F) vs fully unfrozen (T) MedCLIP-Swin, same head and budget.

Plan and reading rule: ``docs/handoff/PLAN-2026-10-31-swin-finetune-probe.md``. Input is the
uint8 cache of ``scripts/swin_finetune_cache.py``; the head is the encoder probe's
``build_attention_probe`` on Swin's 49 grid tokens + pooled token.

``fit --arm frozen|finetune``: gate G0 (cache keys/labels equal the encoder probe's, and frozen
Swin on the cache reproduces its stored Swin features), then 8 epochs on the 35,654 fit studies,
epoch chosen on the 3,961 holdout studies (the encoder probe's split), val read at that epoch
only. G1: F's val macro AP >= 0.4203. G2: T changes Swin's stage-1 and stage-4 weights, F leaves
every Swin tensor bit-identical. Writes ``val_probs.npy`` (host only) and ``report.json``.

``compare``: delta = T - F in val macro AP of "present", paired study bootstrap, and the verdict
GO_ON (delta >= +0.02 and lower bound > 0) / NOT_ENOUGH / STOP (delta <= 0). Aggregates only.

    python scripts/swin_finetune_fit.py fit --arm frozen --cache-dir <c> --encprobe-dir <e> --out-dir <o>/F
    python scripts/swin_finetune_fit.py compare --frozen-dir <o>/F --finetune-dir <o>/T \\
        --val-cache <c>/val --mhcac R1C=<val npz> --report <o>/compare.json
"""

from __future__ import annotations

import argparse
import hashlib
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

from encoder_probe_fit import (  # noqa: E402
    RARE,
    _masked_ce,
    build_attention_probe,
    macro_ap,
    paired_bootstrap,
    per_finding,
)

from vision_encoders.swin.medclip_swin import MEDCLIP_IMG_MEAN, MEDCLIP_IMG_STD  # noqa: E402

GO_MARGIN = 0.02
G1_FLOOR = 0.4263 - 0.006  # the encoder probe's attn_swin minus 0.006
OLD_ATTN_SWIN = 0.4263
SWIN_BASE_LR, SWIN_LR_DECAY, HEAD_LR, WEIGHT_DECAY = 5e-5, 0.7, 3e-4, 0.05
EPOCHS, BATCH, WARMUP_FRAC, SEED = 8, 64, 0.05, 16
ROTATE_DEG, SCALE, TRANSLATE, BRIGHT, CONTRAST = 10.0, (0.9, 1.1), 0.05, 0.1, 0.1


def holdout_split(labels: np.ndarray, seed: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """The encoder probe's split: 10% of the labelled studies held out, permutation ``seed``."""
    labelled = np.flatnonzero((labels >= 0).any(1))
    perm = np.random.default_rng(seed).permutation(labelled)
    return np.sort(perm[len(perm) // 10:]), np.sort(perm[: len(perm) // 10])


def swin_stage(name: str) -> int:
    """Depth of a SwinEncoder parameter: 0 = patch embedding, 1-4 = stages, 4 also final LayerNorm."""
    if name.startswith("model.embeddings."):
        return 0
    if name.startswith("model.encoder.layers."):
        return int(name.split(".")[3]) + 1
    if name.startswith("model.layernorm."):
        return 4
    raise ValueError(f"unassigned Swin parameter {name}")


def swin_param_groups(named_params, base_lr=SWIN_BASE_LR, decay=SWIN_LR_DECAY, weight_decay=WEIGHT_DECAY):
    """Layer-wise LR decay: stage 4 gets ``base_lr``, each level below x ``decay``; no weight
    decay on 1-d tensors (biases, LayerNorm). Returns AdamW groups, each with a ``names`` list."""
    groups: dict = {}
    for name, p in named_params:
        depth = swin_stage(name)
        wd = 0.0 if p.ndim <= 1 else weight_decay
        g = groups.setdefault((depth, wd), {"params": [], "names": [], "weight_decay": wd,
                                            "lr": base_lr * decay ** (4 - depth), "depth": depth})
        g["params"].append(p)
        g["names"].append(name)
    return [groups[k] for k in sorted(groups)]


def warmup_cosine(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / warmup
    return 0.5 * (1.0 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))


def affine_theta(angle_deg, scale, tx, ty):
    """[B, 2, 3] ``affine_grid`` matrices: rotation, isotropic zoom, translation in image
    fractions. Determinant 1/scale^2 > 0, so never a reflection (no horizontal flip)."""
    import torch

    a = angle_deg * math.pi / 180.0
    c, s = torch.cos(a) / scale, torch.sin(a) / scale
    return torch.stack([torch.stack([c, -s, 2 * tx], -1), torch.stack([s, c, 2 * ty], -1)], 1)


def augment(x, gen):
    """Train-only augmentation on a [B, 1, H, W] batch in [0, 1]: rotate +-10 deg, scale
    0.9-1.1, translate +-5%, brightness and contrast x[0.9, 1.1]; black (zero) fill."""
    import torch
    import torch.nn.functional as F

    b = x.shape[0]

    def u(lo, hi):
        return lo + (hi - lo) * torch.rand(b, generator=gen, device=x.device)

    theta = affine_theta(u(-ROTATE_DEG, ROTATE_DEG), u(*SCALE), u(-TRANSLATE, TRANSLATE),
                         u(-TRANSLATE, TRANSLATE))
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    x = (x * u(1 - BRIGHT, 1 + BRIGHT).view(-1, 1, 1, 1)).clamp(0, 1)
    m = x.mean((1, 2, 3), keepdim=True)
    return ((x - m) * u(1 - CONTRAST, 1 + CONTRAST).view(-1, 1, 1, 1) + m).clamp(0, 1)


def to_swin_input(images_u8, gen=None):
    """uint8 [B, 224, 224] -> MedCLIP-normalised [B, 3, 224, 224], as ``medclip_preprocess``;
    augmented first when ``gen`` is given."""
    x = images_u8.float().unsqueeze(1) / 255.0
    if gen is not None:
        x = augment(x, gen)
    return ((x - MEDCLIP_IMG_MEAN) / MEDCLIP_IMG_STD).repeat(1, 3, 1, 1)


def keys_digest(keys) -> str:
    return hashlib.sha256("\n".join(map(str, keys)).encode()).hexdigest()[:16]


def _load_cache(d: Path):
    return (np.load(d / "images.npy", mmap_mode="r"), np.load(d / "labels.npy"), np.load(d / "keys.npy"))


def _swin_weights(config: Path) -> tuple[str, str]:
    from omegaconf import OmegaConf

    swin = OmegaConf.load(config).model.swin
    if str(swin.backend).lower() != "medclip":
        raise SystemExit("the probe is defined for model.swin.backend medclip")
    return str(swin.model_name), str(swin.weights_path)


def gate0(cache_dir: Path, encprobe_dir: Path, swin, device, n_check=64) -> dict:
    import torch

    out = {}
    for split in ("train", "val"):
        _img, lab, keys = _load_cache(cache_dir / split)
        ref_lab, ref_keys = np.load(encprobe_dir / split / "labels.npy"), np.load(encprobe_dir / split / "keys.npy")
        if not (np.array_equal(keys, ref_keys) and np.array_equal(lab, ref_lab)):
            raise SystemExit(f"G0 FAILED: {split} cache keys/labels differ from {encprobe_dir / split}")
        out[f"n_{split}"] = int(len(keys))
    img = _load_cache(cache_dir / "val")[0]
    ref_g = np.load(encprobe_dir / "val" / "swin_global.npy", mmap_mode="r")[:n_check].astype(np.float32)
    ref_t = np.load(encprobe_dir / "val" / "swin_grid.npy", mmap_mode="r")[:n_check].astype(np.float32)
    ref = torch.from_numpy(np.concatenate([ref_g[:, None], ref_t], 1)).flatten(1)
    swin.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        new = swin(to_swin_input(torch.from_numpy(np.asarray(img[:n_check])).to(device))).float().cpu().flatten(1)
    cos = torch.nn.functional.cosine_similarity(new, ref, dim=1)
    out["feature_cosine_min"] = float(cos.min())
    print(f"[swinft] G0 keys/labels equal; feature cosine min {out['feature_cosine_min']:.5f} "
          f"over {n_check} val studies", flush=True)
    if out["feature_cosine_min"] < 0.999:
        raise SystemExit("G0 FAILED: frozen Swin on the cache does not reproduce the stored features")
    return out


def cmd_fit(args) -> int:
    import torch

    from vision_encoders.swin.swin_encoder import SwinEncoder

    finetune = args.arm == "finetune"
    args.out_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda")
    model_name, weights = _swin_weights(args.config)
    swin = SwinEncoder(model_name=model_name, backend="medclip", normalize=False, weights_path=weights,
                       frozen=not finetune).to(device)
    report: dict = {"arm": args.arm, "gate0": gate0(args.cache_dir, args.encprobe_dir, swin, device)}
    initial = {k: v.detach().cpu().clone() for k, v in swin.state_dict().items()}

    tr_img, y_tr, _ = _load_cache(args.cache_dir / "train")
    va_img, y_val, va_keys = _load_cache(args.cache_dir / "val")
    fit, hold = holdout_split(y_tr)
    print(f"[swinft] {args.arm}: fit {len(fit)}, holdout {len(hold)}, val {len(y_val)}", flush=True)

    torch.manual_seed(args.seed)
    head = build_attention_probe([(49, swin.embed_dim)], [swin.embed_dim]).to(device)
    groups = [{"params": list(head.parameters()), "lr": HEAD_LR, "weight_decay": WEIGHT_DECAY, "names": ["head"]}]
    if finetune:
        groups += swin_param_groups(swin.named_parameters())
    else:
        assert not any(p.requires_grad for p in swin.parameters())
    opt = torch.optim.AdamW([{k: v for k, v in g.items() if k not in ("names", "depth")} for g in groups])
    report["lr_groups"] = [{"depth": g.get("depth", "head"), "lr": g["lr"], "weight_decay": g["weight_decay"],
                            "n_tensors": len(g["params"])} for g in groups]
    steps_per_epoch = math.ceil(len(fit) / BATCH)
    total = EPOCHS * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: warmup_cosine(s, total, max(1, int(WARMUP_FRAC * total))))
    gen = torch.Generator(device=device).manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    def tokens(images_u8, augmented):
        x = to_swin_input(torch.from_numpy(np.asarray(images_u8)).to(device), gen if augmented else None)
        out = swin(x)
        return [out[:, 1:]], [out[:, 0]]

    def predict(images, idx):
        swin.eval(), head.eval()
        out = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in np.array_split(idx, max(1, math.ceil(len(idx) / 256))):
                out.append(torch.softmax(head(*tokens(images[i], False)).float(), -1).cpu().numpy())
        return np.concatenate(out)

    best, history, started = (-1.0, -1, None), [], time.time()
    for epoch in range(EPOCHS):
        swin.train(finetune), head.train()
        losses = []
        for i in np.array_split(rng.permutation(fit), steps_per_epoch):
            i = np.sort(i)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if finetune:
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
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "holdout_macro_ap": hold_ap,
                        "minutes": (time.time() - started) / 60})
        print(f"[swinft] {args.arm} epoch {epoch}: train loss {np.mean(losses):.4f}, holdout macro AP "
              f"{hold_ap:.4f} ({(time.time() - started) / 60:.1f} min)", flush=True)
        if hold_ap > best[0]:
            best = (hold_ap, epoch, predict(va_img, np.arange(len(y_val))))
    report.update(history=history, chosen_epoch=best[1], holdout_macro_ap=best[0],
                  max_mem_mib=torch.cuda.max_memory_allocated() / 2**20,
                  val_keys_sha=keys_digest(va_keys), n_fit=int(len(fit)), n_hold=int(len(hold)),
                  n_val=int(len(y_val)))
    apf, aucf = per_finding(best[2], y_val)
    report.update(val_macro_ap=float(np.nanmean(apf)), val_auroc_present=float(np.nanmean(aucf)),
                  val_ap_per_finding=apf.tolist())
    np.save(args.out_dir / "val_probs.npy", best[2].astype(np.float32))

    # G2: did Swin's weights move (T) / stay bit-identical (F)?
    final = swin.state_dict()
    change = {}
    for depth in range(5):
        names = [k for k in initial if k.startswith("model.") and initial[k].is_floating_point()
                 and _safe_stage(k) == depth]
        num = sum(float((final[k].detach().cpu().float() - initial[k].float()).norm() ** 2) for k in names)
        den = sum(float(initial[k].float().norm() ** 2) for k in names)
        change[f"depth{depth}"] = math.sqrt(num / max(den, 1e-30))
    identical = all(torch.equal(final[k].detach().cpu(), initial[k]) for k in initial)
    report["gate2"] = {"relative_change": change, "bit_identical": identical}
    g2_ok = (change["depth1"] > 0 and change["depth4"] > 0) if finetune else identical
    print(f"[swinft] G2 {'passed' if g2_ok else 'FAILED'}: relative change {change}, bit-identical {identical}",
          flush=True)
    report["gate2"]["passed"] = g2_ok
    g1_ok = finetune or report["val_macro_ap"] >= G1_FLOOR
    report["gate1"] = None if finetune else {"floor": G1_FLOOR, "passed": g1_ok}
    print(f"[swinft] {args.arm}: chosen epoch {best[1]}, holdout {best[0]:.4f}, val macro AP "
          f"{report['val_macro_ap']:.4f}, AUROC present {report['val_auroc_present']:.4f}"
          + ("" if finetune else f"; G1 {'passed' if g1_ok else 'FAILED'} (floor {G1_FLOOR:.4f})"), flush=True)
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not g2_ok:
        raise SystemExit("G2 FAILED")
    if not g1_ok:
        raise SystemExit("G1 FAILED")
    return 0


def _safe_stage(name: str):
    try:
        return swin_stage(name)
    except ValueError:
        return None


def verdict(delta: float, lo: float) -> str:
    if delta <= 0:
        return "STOP"
    if delta >= GO_MARGIN and lo > 0:
        return "GO_ON"
    return "NOT_ENOUGH"


def cmd_compare(args) -> int:
    from training.evaluation.schemas import ClassificationPredictions

    y_val, keys = np.load(args.val_cache / "labels.npy"), np.load(args.val_cache / "keys.npy")
    arms = {}
    for name, d in (("F", args.frozen_dir), ("T", args.finetune_dir)):
        rep = json.loads((d / "report.json").read_text(encoding="utf-8"))
        if rep["val_keys_sha"] != keys_digest(keys):
            raise SystemExit(f"{d} was scored on other val studies")
        arms[name] = np.load(d / "val_probs.npy")
    probs = dict(arms)
    names = None
    pos = {k: i for i, k in enumerate(keys.tolist())}
    for spec in args.mhcac:
        name, path = spec.split("=", 1)
        p = ClassificationPredictions.load(path)
        names = list(p.pathology_names)
        order = np.array([pos[k] for k in p.sample_keys.tolist()])
        if not np.array_equal(np.where(p.labels < 0, -1, p.labels), np.where(y_val[order] < 0, -1, y_val[order])):
            raise SystemExit(f"{name}: labels differ from the cache's val labels")
        aligned = np.empty_like(p.probabilities)
        aligned[order] = p.probabilities
        probs[f"mhcac_{name}"] = aligned
    names = names or [f"finding_{j}" for j in range(y_val.shape[1])]

    report: dict = {"n_val": int(len(y_val)), "table": {}, "comparisons": {}}
    print("[swinft] val, macro AP / AUROC present:")
    for k, pv in probs.items():
        apf, aucf = per_finding(pv, y_val)
        report["table"][k] = {"macro_ap": float(np.nanmean(apf)), "auroc_present": float(np.nanmean(aucf)),
                              "ap_per_finding": apf.tolist()}
        print(f"  {k:14s} {np.nanmean(apf):.4f} / {np.nanmean(aucf):.4f}")
    print(f"  (encoder probe attn_swin, 3 seeds, no augmentation: {OLD_ATTN_SWIN:.4f})")
    print("[swinft] AP per finding (rare marked *): " + " | ".join(report["table"]))
    for j, f in enumerate(names):
        print(f"  {f + ('*' if f in RARE else ''):30s} "
              + " ".join(f"{v['ap_per_finding'][j]:.3f}" for v in report["table"].values()))
    d, lo, hi = paired_bootstrap(arms["F"], arms["T"], y_val)
    report["comparisons"]["T-F"] = {"delta": d, "lo": lo, "hi": hi}
    for k in probs:
        if k.startswith("mhcac_"):
            dd, ll, hh = paired_bootstrap(probs[k], arms["T"], y_val)
            report["comparisons"][f"T-{k}"] = {"delta": dd, "lo": ll, "hi": hh}
            print(f"[swinft] T - {k}: {dd:+.4f} [{ll:+.4f}, {hh:+.4f}] (descriptive)")
    report["verdict"] = verdict(d, lo)
    print(f"[swinft] T - F macro AP {d:+.4f} [{lo:+.4f}, {hi:+.4f}] -> VERDICT {report['verdict']}")
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--arm", required=True, choices=("frozen", "finetune"))
    f.add_argument("--cache-dir", type=Path, required=True, help="holds train/ and val/ from swin_finetune_cache.py")
    f.add_argument("--encprobe-dir", type=Path, required=True, help="the encoder probe's feature dir (gate G0)")
    f.add_argument("--config", type=Path, default=_REPO_ROOT / "pretraining/configs/mimic_cxr_full.yaml")
    f.add_argument("--seed", type=int, default=SEED)
    f.add_argument("--out-dir", type=Path, required=True)
    c = sub.add_parser("compare")
    c.add_argument("--frozen-dir", type=Path, required=True)
    c.add_argument("--finetune-dir", type=Path, required=True)
    c.add_argument("--val-cache", type=Path, required=True)
    c.add_argument("--mhcac", action="append", default=[], help="NAME=val npz (descriptive reference)")
    c.add_argument("--report", type=Path, required=True)
    args = ap.parse_args(argv)
    return cmd_fit(args) if args.cmd == "fit" else cmd_compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
