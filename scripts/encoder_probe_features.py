#!/usr/bin/env python3
"""Per-encoder features (global vector + coarse token grid) for the encoder probe.

Reading rule and design: ``docs/handoff/PLAN-2026-10-31-encoder-probe.md``. For every
study of a split (or a seeded subset), the ANCHOR image only, clean (no training
augmentation), pretrained encoders, this writes per encoder ``<name>_global.npy``
``[N, Dg]`` and ``<name>_grid.npy`` ``[N, T, D]`` (float16 memmaps), plus
``labels.npy`` (``[N, 14]``, 0/1/2, -100 = no label) and ``keys.npy`` -- the
salted digests the Stage-1 eval hook writes as ``sample_keys``, so the probe can
pair with MHCAC's own val predictions.

* ``swin``       MedCLIP-Swin on the dataset's ``swin_image``: pooled token; 7x7 tokens.
* ``pubmedclip`` PubMedCLIP on ``pubmedclip_image``: CLS; 7x7 patch tokens.
* ``biovil``     BioViL-T's OWN 512-d output (``patch_embeddings`` / ``img_embedding``),
                 not the randomly initialised 1408-d projector the pipeline adds; 14x14
                 averaged to 7x7. ``biovil128_global``: mean of BioViL-T's pretrained
                 128-d projector over the grid.
* ``raddino``    (``--raddino DIR``) its own preprocessing (shortest edge 518, centre
                 crop 518): [CLS ; mean patch]; 37x37 averaged to ``--grid``.
* ``medsiglip``  (``--medsiglip ID``) its own preprocessing (448x448, mean/std 0.5):
                 [pooler ; mean patch]; 32x32 averaged to ``--grid``.

Derived features of MIMIC-CXR: keep them on the training host. Prints counts only.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (_REPO_ROOT, _REPO_ROOT / "training"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

RADDINO_SIZE, RADDINO_MEAN, RADDINO_STD = 518, 0.5307, 0.2583
MEDSIGLIP_SIZE, MEDSIGLIP_MEAN, MEDSIGLIP_STD = 448, 0.5, 0.5


def _to_chw(rgb: Image.Image, mean: float, std: float) -> np.ndarray:
    x = np.asarray(rgb, dtype=np.float32) / 255.0
    return ((x - mean) / std).transpose(2, 0, 1).copy()


def raddino_preprocess(image: Image.Image) -> np.ndarray:
    """RAD-DINO's processor: shortest edge 518 (bicubic), centre crop 518, its mean/std."""
    rgb = image.convert("RGB")
    w, h = rgb.size
    if w <= h:
        size = (RADDINO_SIZE, int(RADDINO_SIZE * h / w))
    else:
        size = (int(RADDINO_SIZE * w / h), RADDINO_SIZE)
    rgb = rgb.resize(size, Image.BICUBIC)
    left, top = (rgb.width - RADDINO_SIZE) // 2, (rgb.height - RADDINO_SIZE) // 2
    rgb = rgb.crop((left, top, left + RADDINO_SIZE, top + RADDINO_SIZE))
    return _to_chw(rgb, RADDINO_MEAN, RADDINO_STD)


def medsiglip_preprocess(image: Image.Image) -> np.ndarray:
    """MedSigLIP's processor: 448x448 (bicubic, no aspect preservation), mean/std 0.5."""
    rgb = image.convert("RGB").resize((MEDSIGLIP_SIZE, MEDSIGLIP_SIZE), Image.BICUBIC)
    return _to_chw(rgb, MEDSIGLIP_MEAN, MEDSIGLIP_STD)


def grid_tokens(tokens, side: int, out: int):
    """[B, side*side, D] row-major patch tokens -> [B, out*out, D] by average pooling."""
    import torch.nn.functional as F

    b, n, d = tokens.shape
    assert n == side * side, (n, side)
    g = tokens.transpose(1, 2).reshape(b, d, side, side)
    if side != out:
        g = F.adaptive_avg_pool2d(g, out)
    return g.flatten(2).transpose(1, 2)


class _WithNative:
    """Adds RAD-DINO / MedSigLIP inputs, decoded from the same anchor image, to a dataset item."""

    def __init__(self, base, indices, raddino: bool, medsiglip: bool):
        self.base, self.indices, self.raddino, self.medsiglip = base, indices, raddino, medsiglip

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        import torch

        sample = self.base[self.indices[i]]
        if self.raddino or self.medsiglip:
            image = self.base.load_image(Path(sample["image_path"]))
            if self.raddino:
                sample["raddino_image"] = torch.from_numpy(raddino_preprocess(image))
            if self.medsiglip:
                sample["medsiglip_image"] = torch.from_numpy(medsiglip_preprocess(image))
        return sample


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True, choices=("train", "val", "test"))
    ap.add_argument("--limit", type=int, default=None, help="seeded random subset of studies")
    ap.add_argument("--seed", type=int, default=16)
    ap.add_argument("--config", type=Path, default=_REPO_ROOT / "pretraining/configs/mimic_cxr_full.yaml")
    ap.add_argument("--raddino", type=Path, default=None, help="local microsoft/rad-dino dir")
    ap.add_argument("--medsiglip", default=None, help="e.g. google/medsiglip-448")
    ap.add_argument("--grid", type=int, default=8, help="output grid side for RAD-DINO / MedSigLIP")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.split == "test":
        raise SystemExit("the encoder probe reads train and val only (plan section 2)")

    import torch
    from torch.utils.data import DataLoader

    from biovil_t.model import ImageModel
    from biovil_t.pretrained import _download_biovil_t_image_model_weights
    from biovil_t.types import ImageEncoderType
    from training.evaluation.schemas import build_sample_keys
    from training.run_context import Stage1Context
    from training.stage1 import lavis_loader as L

    args.out_dir.mkdir(parents=True, exist_ok=False)
    cfg = L.build_cfg(Stage1Context(run_name="encprobe", config_path=args.config, checkpoint_path=None))
    cfg.datasets_cfg.mimic_cxr.vis_processor.train.augmentation.enabled = False
    cfg.model_cfg.data.max_aux_views = 0
    cfg.model_cfg.explanation.mask_cache_dir = None
    model = L.tasks.setup_task(cfg).build_model(cfg)
    device = torch.device("cuda")
    model.to(device).eval()
    if not (model.use_biovil and model.use_pubmedclip and model.use_swin):
        raise RuntimeError("expected the shipped three encoders")

    # BioViL-T's pretrained 128-d projector: the pipeline drops it on load.
    proj128 = ImageModel(img_encoder_type=ImageEncoderType.RESNET50_MULTI_IMAGE, joint_feature_size=128)
    proj128.load_state_dict(torch.load(_download_biovil_t_image_model_weights(), map_location="cpu"), strict=True)
    proj128 = proj128.projector.to(device).eval()

    raddino = medsiglip = None
    if args.raddino is not None:
        from transformers import AutoModel

        raddino = AutoModel.from_pretrained(str(args.raddino)).to(device).eval()
    if args.medsiglip is not None:
        from transformers import SiglipVisionModel

        medsiglip = SiglipVisionModel.from_pretrained(args.medsiglip).to(device).eval()

    dataset = L.MIMIC_CXR_Dataset(vis_processor=None, text_processor=None, vis_root=L.VIS_ROOT,
                                  split=args.split, cfg=cfg)
    if getattr(dataset, "augmentation_enabled", False):
        raise RuntimeError("training augmentation is still on; features would be noisy")
    order = np.arange(len(dataset))
    if args.limit and args.limit < len(order):
        order = np.sort(np.random.default_rng(args.seed).choice(order, args.limit, replace=False))
    n = len(order)
    print(f"[encprobe] {args.split}: {n} of {len(dataset)} studies", flush=True)
    loader = DataLoader(_WithNative(dataset, order.tolist(), raddino is not None, medsiglip is not None),
                        batch_size=args.batch, num_workers=args.workers, collate_fn=dataset.collater)

    store: dict[str, np.ndarray] = {}

    def put(name, pos, value):
        value = value.float().cpu().numpy().astype(np.float16)
        if name not in store:
            store[name] = np.lib.format.open_memmap(args.out_dir / f"{name}.npy", mode="w+",
                                                    dtype=np.float16, shape=(n,) + value.shape[1:])
        store[name][pos:pos + len(value)] = value

    labels = np.full((n, 14), -100, dtype=np.int16)
    keys: list[str] = []
    pos, started = 0, time.time()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for b, batch in enumerate(loader):
            bs = batch["image"].shape[0]
            lab = batch["classification_labels"].numpy().astype(np.int16)
            lab[~batch["classification_mask"].numpy().astype(bool)] = -100
            labels[pos:pos + bs] = lab
            ids = batch["image_id"]
            keys.extend(build_sample_keys([str(v) for v in (ids.tolist() if torch.is_tensor(ids) else ids)]))

            out = model.visual_encoder(batch["image"].to(device))
            patch = out.patch_embeddings  # [B, 512, 14, 14]
            put("biovil_global", pos, out.img_embedding)
            put("biovil_grid", pos, grid_tokens(patch.flatten(2).transpose(1, 2), 14, 7))
            put("biovil128_global", pos, proj128(patch).mean((2, 3)))

            pmc = model._pubmedclip_tokens(None, batch["pubmedclip_image"].to(device))
            put("pubmedclip_global", pos, pmc[:, 0])
            put("pubmedclip_grid", pos, pmc[:, 1:])

            sw = model.swin(model._swin_input(None, batch["swin_image"].to(device)))
            put("swin_global", pos, sw[:, 0])
            put("swin_grid", pos, sw[:, 1:])

            if raddino is not None:
                h = raddino(pixel_values=batch["raddino_image"].to(device)).last_hidden_state
                put("raddino_global", pos, torch.cat([h[:, 0], h[:, 1:].mean(1)], -1))
                put("raddino_grid", pos, grid_tokens(h[:, 1:], RADDINO_SIZE // 14, args.grid))
            if medsiglip is not None:
                o = medsiglip(pixel_values=batch["medsiglip_image"].to(device))
                h = o.last_hidden_state
                put("medsiglip_global", pos, torch.cat([o.pooler_output, h.mean(1)], -1))
                put("medsiglip_grid", pos, grid_tokens(h, MEDSIGLIP_SIZE // 14, args.grid))
            pos += bs
            if (b + 1) % 100 == 0:
                print(f"[encprobe] {pos}/{n} {pos / (time.time() - started):.1f} study/s", flush=True)
    assert pos == n, (pos, n)
    np.save(args.out_dir / "labels.npy", labels)
    np.save(args.out_dir / "keys.npy", np.array(keys))
    for name, arr in store.items():
        arr.flush()
        print(f"[encprobe] wrote {name} {arr.shape}", flush=True)
    print(f"[encprobe] labelled studies {(labels >= 0).any(1).sum()} of {n}; "
          f"{time.time() - started:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
