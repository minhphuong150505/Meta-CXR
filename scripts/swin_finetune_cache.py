#!/usr/bin/env python3
"""uint8 cache of MedCLIP-Swin's 224x224 input, for the Swin fine-tune probe.

Plan and reading rule: ``docs/handoff/PLAN-2026-10-31-swin-finetune-probe.md``. Same
studies, order, labels and keys as ``scripts/encoder_probe_features.py`` (anchor image
only, clean, seeded ``--limit`` subset), but instead of features it stores the image
``medclip_preprocess`` builds BEFORE ``/255`` and the MedCLIP mean/std: padded to a black
square, resized 224x224 bicubic, one channel. The uint8 value is recovered exactly from
the dataset's own ``swin_image`` (:func:`swin_image_to_uint8`), so the image is decoded
once and is the pipeline's input bit for bit.

Writes ``images.npy`` ``[N, 224, 224]`` uint8, ``labels.npy`` ``[N, 14]`` (0/1/2, -100 =
no label) and ``keys.npy`` (the eval hook's salted ``sample_keys``). Derived data of
MIMIC-CXR: keep it on the training host. Prints counts only.

    python scripts/swin_finetune_cache.py --split train --limit 40000 --out-dir <dir>/train
    python scripts/swin_finetune_cache.py --split val --out-dir <dir>/val
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (_REPO_ROOT, _REPO_ROOT / "training"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from vision_encoders.swin.medclip_swin import (  # noqa: E402
    MEDCLIP_IMG_MEAN,
    MEDCLIP_IMG_SIZE,
    MEDCLIP_IMG_STD,
)


def swin_image_to_uint8(swin_image) -> np.ndarray:
    """``medclip_preprocess`` output ``[..., 3, 224, 224]`` -> the uint8 image it came from.

    The three channels are copies; the float32 round trip is far below half a grey level.
    """
    x = np.asarray(swin_image, dtype=np.float32)[..., 0, :, :]
    return np.clip(np.rint((x * MEDCLIP_IMG_STD + MEDCLIP_IMG_MEAN) * 255.0), 0, 255).astype(np.uint8)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True, choices=("train", "val", "test"))
    ap.add_argument("--limit", type=int, default=None, help="seeded random subset of studies")
    ap.add_argument("--seed", type=int, default=16)
    ap.add_argument("--config", type=Path, default=_REPO_ROOT / "pretraining/configs/mimic_cxr_full.yaml")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.split == "test":
        raise SystemExit("the Swin fine-tune probe reads train and val only (plan section 2.1)")

    import torch
    from torch.utils.data import DataLoader

    from training.evaluation.schemas import build_sample_keys
    from training.run_context import Stage1Context
    from training.stage1 import lavis_loader as L

    args.out_dir.mkdir(parents=True, exist_ok=False)
    cfg = L.build_cfg(Stage1Context(run_name="swinft", config_path=args.config, checkpoint_path=None))
    cfg.datasets_cfg.mimic_cxr.vis_processor.train.augmentation.enabled = False
    cfg.model_cfg.data.max_aux_views = 0
    cfg.model_cfg.explanation.mask_cache_dir = None
    dataset = L.MIMIC_CXR_Dataset(vis_processor=None, text_processor=None, vis_root=L.VIS_ROOT,
                                  split=args.split, cfg=cfg)
    if getattr(dataset, "augmentation_enabled", False):
        raise RuntimeError("training augmentation is still on")
    if not getattr(dataset, "emit_swin_image", False):
        raise RuntimeError("the dataset does not emit swin_image (needs encoders.swin with backend medclip)")
    order = np.arange(len(dataset))
    if args.limit and args.limit < len(order):
        order = np.sort(np.random.default_rng(args.seed).choice(order, args.limit, replace=False))
    n = len(order)
    print(f"[swinft-cache] {args.split}: {n} of {len(dataset)} studies", flush=True)
    loader = DataLoader(torch.utils.data.Subset(dataset, order.tolist()), batch_size=args.batch,
                        num_workers=args.workers, collate_fn=dataset.collater)

    side = MEDCLIP_IMG_SIZE
    images = np.lib.format.open_memmap(args.out_dir / "images.npy", mode="w+", dtype=np.uint8,
                                       shape=(n, side, side))
    labels = np.full((n, 14), -100, dtype=np.int16)
    keys: list[str] = []
    pos, started = 0, time.time()
    for b, batch in enumerate(loader):
        bs = batch["swin_image"].shape[0]
        lab = batch["classification_labels"].numpy().astype(np.int16)
        lab[~batch["classification_mask"].numpy().astype(bool)] = -100
        labels[pos:pos + bs] = lab
        ids = batch["image_id"]
        keys.extend(build_sample_keys([str(v) for v in (ids.tolist() if torch.is_tensor(ids) else ids)]))
        images[pos:pos + bs] = swin_image_to_uint8(batch["swin_image"].numpy())
        pos += bs
        if (b + 1) % 100 == 0:
            print(f"[swinft-cache] {pos}/{n} {pos / (time.time() - started):.1f} study/s", flush=True)
    assert pos == n, (pos, n)
    images.flush()
    np.save(args.out_dir / "labels.npy", labels)
    np.save(args.out_dir / "keys.npy", np.array(keys))
    print(f"[swinft-cache] wrote images {images.shape}; labelled studies {(labels >= 0).any(1).sum()} "
          f"of {n}; {time.time() - started:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
