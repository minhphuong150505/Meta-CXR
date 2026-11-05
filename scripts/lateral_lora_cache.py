#!/usr/bin/env python3
"""uint8 cache of MedCLIP-Swin's 224 input for BOTH views of every PL study (Step 1, LoRA).

Plan and reading rule: ``docs/handoff/PLAN-2026-11-04-residual-lateral.md`` (Step 1). One
dataset pass with the default anchor priority ``[PA, AP, lateral]`` and ``max_aux_views=1``:
for a PL study (frontal anchor + one lateral aux view, ``training/evaluation/view_keys.py``) the
anchor's ``swin_image`` is the frontal image and ``aux_swin_image`` is the lateral one -- the
same lateral ``encoder_probe_features.py --views lateral`` encodes. Both are stored as the uint8
image ``medclip_preprocess`` builds before ``/255`` (``swin_finetune_cache.swin_image_to_uint8``).

Writes ``frontal.npy`` / ``lateral.npy`` ``[N, 224, 224]`` uint8, ``labels.npy`` ``[N, 14]``
(-100 = no label), ``keys.npy`` (frontal-anchor salted key, the key every OFF npz carries),
``image_keys.npy`` (the lateral image's key) and ``view_map.npz``. Checks, all fatal: the
dataset's study order equals the rebuilt view map's; every kept study's aux row is the PL
lateral; every batch's keys equal the expected frontal keys and its aux view is a real lateral.
``--limit`` takes the encoder probe's seeded study subset first, then keeps its PL studies.
Derived data of MIMIC-CXR: keep it on the training host. Prints counts only.

    python scripts/lateral_lora_cache.py --split train --out-dir <dir>/train
    python scripts/lateral_lora_cache.py --split val --out-dir <dir>/val
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (_REPO_ROOT, _REPO_ROOT / "training", _REPO_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from swin_finetune_cache import swin_image_to_uint8  # noqa: E402

from training.evaluation.view_keys import LATERAL_VIEW_ID, PL, ViewMap, build_view_map  # noqa: E402


def pl_positions(view_map: ViewMap, studies, order: np.ndarray) -> np.ndarray:
    """Dataset indices of the PL studies in ``order``; raises unless each one's aux row is the PL lateral."""
    if len(studies) != len(view_map):
        raise ValueError(f"dataset has {len(studies)} studies, map has {len(view_map)}")
    keep = np.asarray([int(i) for i in order if view_map.groups[i] == PL], dtype=np.int64)
    for i in keep:
        s = studies[i]
        if list(s["aux"]) != [int(view_map.lateral_rows[i])] or list(s["aux_view_ids"]) != [LATERAL_VIEW_ID]:
            raise ValueError(f"study at dataset index {i}: aux view is not the PL lateral")
        if int(s["anchor"]) != int(view_map.frontal_rows[i]):
            raise ValueError(f"study at dataset index {i}: anchor is not the default frontal anchor")
    return keep


def check_batch(batch, expected_keys: np.ndarray, build_sample_keys) -> None:
    """Fatal unless the batch is exactly the expected PL studies, each with one real lateral."""
    import torch

    ids = batch["image_id"]
    keys = build_sample_keys([str(v) for v in (ids.tolist() if torch.is_tensor(ids) else ids)])
    if not np.array_equal(np.asarray(keys), np.asarray(expected_keys)):
        raise RuntimeError("batch keys differ from the expected frontal keys")
    if batch["aux_swin_image"].shape[1] != 1 or not bool(batch["aux_mask"][:, 0].all()):
        raise RuntimeError("a PL study arrived without its lateral view")
    if not bool((batch["aux_view_ids"][:, 0] == LATERAL_VIEW_ID).all()):
        raise RuntimeError("aux view is not a lateral")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True, choices=("train", "val", "test"))
    ap.add_argument("--limit", type=int, default=None, help="seeded random subset of studies, before the PL filter")
    ap.add_argument("--seed", type=int, default=16)
    ap.add_argument("--config", type=Path, default=_REPO_ROOT / "pretraining/configs/mimic_cxr_full.yaml")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--allow-test", action="store_true",
                    help="Step 3 only (the single test reading the user approved); refused otherwise")
    args = ap.parse_args(argv)
    if args.split == "test" and not args.allow_test:
        raise SystemExit("test is read once, in Step 3: pass --allow-test")
    if args.split == "test" and args.limit:
        raise SystemExit("Step 3 caches the whole test split; --limit is refused")

    import torch
    from torch.utils.data import DataLoader

    from model.lavis.data.mimic_cxr_utils import build_study_index
    from training.evaluation.schemas import build_sample_keys
    from training.evaluation.view_keys import DEFAULT_PRIORITY
    from training.run_context import Stage1Context
    from training.stage1 import lavis_loader as L

    args.out_dir.mkdir(parents=True, exist_ok=False)
    cfg = L.build_cfg(Stage1Context(run_name="lateral_lora", config_path=args.config, checkpoint_path=None))
    cfg.datasets_cfg.mimic_cxr.vis_processor.train.augmentation.enabled = False
    cfg.model_cfg.data.max_aux_views = 1
    cfg.model_cfg.data.anchor_priority = list(DEFAULT_PRIORITY)
    cfg.model_cfg.explanation.mask_cache_dir = None
    dataset = L.MIMIC_CXR_Dataset(vis_processor=None, text_processor=None, vis_root=L.VIS_ROOT,
                                  split=args.split, cfg=cfg)
    if getattr(dataset, "augmentation_enabled", False):
        raise RuntimeError("training augmentation is still on")
    if not getattr(dataset, "emit_swin_image", False):
        raise RuntimeError("the dataset does not emit swin_image (needs encoders.swin with backend medclip)")
    if not getattr(dataset, "multi_view", False):
        raise RuntimeError("the dataset is not multi-view, so it emits no aux_swin_image")

    cols = ["subject_id", "study_id", "ViewPosition"]
    rows = dataset.annotation[cols].to_dict("records")
    rebuilt = build_study_index(rows, anchor_priority=DEFAULT_PRIORITY, max_aux_views=1)
    if [s["study_key"] for s in rebuilt] != [s["study_key"] for s in dataset.studies]:
        raise RuntimeError("dataset study order differs from the rebuilt default study index")
    view_map = build_view_map(rows)
    view_map.save(args.out_dir / "view_map.npz")

    order = np.arange(len(dataset))
    if args.limit and args.limit < len(order):
        order = np.sort(np.random.default_rng(args.seed).choice(order, args.limit, replace=False))
    keep = pl_positions(view_map, dataset.studies, order)
    n = len(keep)
    print(f"[lora-cache] {args.split}: groups {view_map.counts()}; {n} PL of {len(order)} selected studies",
          flush=True)
    frontal_keys, lateral_keys = view_map.frontal_keys[keep], view_map.lateral_keys[keep]
    loader = DataLoader(torch.utils.data.Subset(dataset, keep.tolist()), batch_size=args.batch,
                        num_workers=args.workers, collate_fn=dataset.collater)

    side = 224
    out = {v: np.lib.format.open_memmap(args.out_dir / f"{v}.npy", mode="w+", dtype=np.uint8, shape=(n, side, side))
           for v in ("frontal", "lateral")}
    labels = np.full((n, 14), -100, dtype=np.int16)
    pos, started = 0, time.time()
    for b, batch in enumerate(loader):
        bs = batch["swin_image"].shape[0]
        check_batch(batch, frontal_keys[pos:pos + bs], build_sample_keys)
        lab = batch["classification_labels"].numpy().astype(np.int16)
        lab[~batch["classification_mask"].numpy().astype(bool)] = -100
        labels[pos:pos + bs] = lab
        out["frontal"][pos:pos + bs] = swin_image_to_uint8(batch["swin_image"].numpy())
        out["lateral"][pos:pos + bs] = swin_image_to_uint8(batch["aux_swin_image"][:, 0].numpy())
        pos += bs
        if (b + 1) % 100 == 0:
            print(f"[lora-cache] {pos}/{n} {pos / (time.time() - started):.1f} study/s", flush=True)
    assert pos == n, (pos, n)
    for arr in out.values():
        arr.flush()
    np.save(args.out_dir / "labels.npy", labels)
    np.save(args.out_dir / "keys.npy", frontal_keys)
    np.save(args.out_dir / "image_keys.npy", lateral_keys)
    print(f"[lora-cache] wrote frontal/lateral {(n, side, side)}; labelled studies {(labels >= 0).any(1).sum()} "
          f"of {n}; {time.time() - started:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
