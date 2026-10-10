#!/usr/bin/env python3
"""Pooled per-encoder image features for a linear-probe comparison of Stage-1 encoders.

Which frozen encoder carries the most CheXpert information? For the studies of
a cached Stage-2 record file (same seeded subset as
``scripts/retrieval_report_baselines.py`` / ``probe_image_features.py``), this
runs the Stage-1 dataset (so each encoder gets ITS OWN preprocessing) and
writes one ``.npz`` per encoder with ``keys`` (anchor image basename) and
``emb``:

* ``biovil``     -- mean of BioViL-T's projected patch embeddings (its global vector);
* ``pubmedclip`` -- PubMedCLIP's CLS token (post-layernorm, as the model reads it);
* ``swin``       -- MedCLIP-Swin's pooled token;
* ``raddino``    -- RAD-DINO (``--raddino <local dir>``): [CLS ; mean patch], the
                    usual DINOv2 linear-probe read-out. Not used by the pipeline;
                    measured to decide whether adding it is worth a Stage-1 run;
* ``softtok``    -- (``--softtok``) the mean of the 32 Q-Former soft tokens from
                    ``Blip2Qformer.forward_image``, exactly what Stage 2 caches.
                    Run per checkpoint (1a / 1b / 1c) to see where the soft tokens
                    gain or lose information.

``--checkpoint none`` reads the pretrained encoders (what any Stage-1 run starts
from); a Stage-1 checkpoint reads the fine-tuned ones. ``--no-encoder-features``
skips the per-encoder read-outs when only ``softtok`` is wanted. Derived
features -- private, keep them on the training host. Prints counts only.
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, required=True, help="Stage-2 record cache of the split")
    ap.add_argument("--split", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=16)
    ap.add_argument("--checkpoint", default="none")
    ap.add_argument("--config", type=Path,
                    default=_REPO_ROOT / "pretraining/configs/mimic_cxr_full.yaml")
    ap.add_argument("--raddino", type=Path, default=None, help="local microsoft/rad-dino dir")
    ap.add_argument("--softtok", action="store_true",
                    help="also write the mean Q-Former soft token (forward_image)")
    ap.add_argument("--no-encoder-features", action="store_true",
                    help="skip biovil/pubmedclip/swin read-outs")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out-prefix", type=Path, required=True)
    args = ap.parse_args(argv)

    import torch
    from torch.utils.data import DataLoader, Subset

    from training.run_context import Stage1Context
    from training.stage1 import lavis_loader as L

    records = torch.load(args.cache, map_location="cpu", weights_only=False, mmap=True)["records"]
    order = np.arange(len(records))
    if args.limit and args.limit < len(order):
        order = np.sort(np.random.default_rng(args.seed).choice(order, args.limit, replace=False))
    wanted = [Path(str(records[i]["image_path"])).stem for i in order]
    del records

    ckpt = None if args.checkpoint == "none" else Path(args.checkpoint)
    cfg = L.build_cfg(Stage1Context(run_name="enc", config_path=args.config, checkpoint_path=ckpt))
    model = L.tasks.setup_task(cfg).build_model(cfg)
    if ckpt is not None:
        state = L.load_torch_checkpoint(ckpt)
        state = state["model"] if isinstance(state, dict) and "model" in state else state
        L.load_state_dict_materializing_meta(model, L.filter_state_dict_for_model(model, state))
    device = torch.device("cuda")
    model.to(device).eval()
    raddino = None
    if args.raddino is not None:
        from vision_encoders.rad_dino import RadDinoEncoder

        raddino = RadDinoEncoder(model_name=str(args.raddino)).to(device).eval()

    # The train split would otherwise apply the random training augmentation
    # (affine + jitter), so train and test features would come from different
    # image distributions. Features are measured on clean images only.
    cfg.datasets_cfg.mimic_cxr.vis_processor.train.augmentation.enabled = False
    dataset = L.MIMIC_CXR_Dataset(vis_processor=None, text_processor=None, vis_root=L.VIS_ROOT,
                                  split=args.split, cfg=cfg)
    if getattr(dataset, "augmentation_enabled", False):
        raise RuntimeError("training augmentation is still on; features would be noisy")
    index_of = {str(dataset.annotation.iloc[s["anchor"]]["dicom_id"]): i
                for i, s in enumerate(dataset.studies)}
    keep = [index_of[d] for d in wanted if d in index_of]
    print(f"[enc] {len(keep)} of {len(wanted)} cached studies found in the {args.split} dataset",
          flush=True)
    loader = DataLoader(Subset(dataset, keep), batch_size=args.batch, num_workers=args.workers,
                        collate_fn=dataset.collater)
    input_keys = ()
    if args.softtok:
        from training.train_eval_figure9_llm_variants_200 import STAGE1_IMAGE_INPUT_KEYS

        input_keys = STAGE1_IMAGE_INPUT_KEYS
    encoders = not args.no_encoder_features
    out: dict[str, list] = {"biovil": [], "pubmedclip": [], "swin": [], "raddino": [],
                            "softtok": []}
    keys: list[str] = []
    started = time.time()
    with torch.no_grad():
        for b, batch in enumerate(loader):
            image = batch["image"].to(device)
            # basename WITH extension: probe_image_features.py joins on
            # image_path.rsplit("/", 1)[-1] of the cached records.
            keys.extend(Path(str(p)).name for p in batch["image_path"])
            if args.softtok:
                inputs = {k: v.to(device, non_blocking=True) for k, v in batch.items()
                          if k in input_keys and torch.is_tensor(v)}
                _logits, soft = model.forward_image(inputs)
                out["softtok"].append(soft.float().mean(1).cpu().numpy())
            if encoders and model.use_biovil:
                patches = model.visual_encoder(image).projected_patch_embeddings
                out["biovil"].append(patches.reshape(image.shape[0], -1, patches.shape[-1])
                                     .mean(1).float().cpu().numpy())
            if encoders and model.use_pubmedclip:
                pmc = batch.get("pubmedclip_image")
                tokens = model._pubmedclip_tokens(image, pmc.to(device) if pmc is not None else None)
                out["pubmedclip"].append(tokens[:, 0].float().cpu().numpy())
            if encoders and model.use_swin:
                swin_in = batch.get("swin_image")
                tokens = model.swin(model._swin_input(image, swin_in.to(device) if swin_in is not None else None))
                out["swin"].append(tokens[:, 0].float().cpu().numpy())
            if raddino is not None:
                hidden = raddino(image)
                out["raddino"].append(torch.cat([hidden[:, 0], hidden[:, 1:].mean(1)], -1)
                                      .float().cpu().numpy())
            if (b + 1) % 200 == 0:
                done = (b + 1) * args.batch
                print(f"[enc] {done}/{len(keep)} {done / (time.time() - started):.1f} study/s",
                      flush=True)
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    for name, chunks in out.items():
        if chunks:
            emb = np.concatenate(chunks).astype(np.float16)
            np.savez(f"{args.out_prefix}_{name}.npz", keys=np.array(keys), emb=emb)
            print(f"[enc] wrote {name} {emb.shape}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
