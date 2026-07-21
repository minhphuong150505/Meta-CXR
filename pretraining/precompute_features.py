"""Precompute frozen vision-encoder features for the MIMIC-CXR pretraining set.

All vision encoders (BioViL, PubMedCLIP, Swin, rad-dino) are frozen and run in eval mode,
and the training transform is fully deterministic (no augmentation), so the raw
output of each frozen encoder is identical for a given image across every epoch.
This script runs each enabled encoder once over the dataset and writes the raw
output -- BEFORE the trainable ``ln_vision`` / ``*_qformer_proj`` layers -- to
disk, keyed by ``dicom_id``. Training can then read these features instead of
re-running the encoders, which removes the dominant per-step compute cost.

Caches are written per encoder so any encoder-toggle combination can reuse them:

    <output_dir>/biovil/<split>_feats.npy   (memmap, (N, P, D) float16)
    <output_dir>/biovil/<split>_ids.json    (list[str] dicom_id, row order)
    <output_dir>/pubmedclip/...             (raw ViT patches, P x 768)
    <output_dir>/swin/...
    <output_dir>/raddino/...

Run single-process (no DDP) on a private GCP VM. Example::

    python -m pretraining.precompute_features \
        --cfg-path pretraining/configs/mimic_cxr_full_l4.yaml \
        --output-dir /mnt/private-feature-cache \
        --splits train val \
        --batch-size 8 \
        --options model.encoders.biovil=true model.encoders.pubmedclip=true \
                  model.encoders.swin=true \
                  model.encoders.raddino=true run.distributed=false run.world_size=1

Keep ``<output_dir>`` only on an access-controlled local disk or private GCS/GCP
volume, then pass its mounted local path as ``run.feature_cache_dir``. Feature IDs
are derived from credentialed MIMIC-CXR and must not be published. Because cache
features are deterministic, disable training image augmentation when consuming a
cache; ``ReportDataset`` rejects that incompatible combination explicitly.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import model.lavis.tasks as tasks
from model.lavis.common.config import Config
from model.lavis.data.ReportDataset import MIMIC_CXR_Dataset
from model.lavis.datasets.builders import *  # noqa: F401,F403  (registers builders/models)
from local_config import VIS_ROOT
from omegaconf import OmegaConf


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute frozen encoder features.")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--output-dir", required=True, help="where to write the feature cache.")
    parser.add_argument("--splits", nargs="+", default=["train", "val"],
                        help="dataset splits to precompute (default: train val).")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--options", nargs="+", default=[],
        help="override config, e.g. model.encoders.swin=true run.distributed=false",
    )
    return parser.parse_args()


@torch.no_grad()
def extract_raw_features(model, image, enabled):
    """Return {encoder: raw frozen output} for the enabled encoders.

    Matches blip2_qformer._encode_image_streams exactly, but stops BEFORE the
    trainable ln_vision layer and before SharedVisualTokenProjector, which owns
    every encoder's projection to the shared visual dimension. The cache is
    therefore unaffected by changes to that projection.
    """
    out = {}
    if enabled.get("biovil"):
        out["biovil"] = model.visual_encoder(image).projected_patch_embeddings.reshape(
            image.shape[0], -1, 1408
        )
    if enabled.get("pubmedclip"):
        out["pubmedclip"] = model.pubmedclip(image, apply_aug=False)[0]
    if enabled.get("swin"):
        out["swin"] = model.swin(image)
    if enabled.get("raddino"):
        out["raddino"] = model.raddino(image)
    return out


def precompute_split(model, dataset, split, output_dir, enabled, batch_size, num_workers):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    n = len(dataset)

    memmaps = {}      # encoder -> np.memmap, allocated lazily on the first batch
    ids = []          # dicom_id in row order (shared across encoders)
    cursor = 0

    for batch in tqdm(loader, total=len(loader), desc=f"precompute {split}"):
        feats = extract_raw_features(model, batch["image"].cuda(), enabled)
        b = len(batch["dicom_id"])

        if not memmaps:  # first batch: now we know (P, D) for each encoder
            for enc, feat in feats.items():
                p, d = feat.shape[1], feat.shape[2]
                enc_dir = Path(output_dir) / enc
                enc_dir.mkdir(parents=True, exist_ok=True)
                feats_path = enc_dir / f"{split}_feats.npy"
                memmaps[enc] = np.lib.format.open_memmap(
                    str(feats_path), mode="w+", dtype=np.float16, shape=(n, p, d)
                )
                print(f"[{split}] {enc}: ({n}, {p}, {d}) float16 -> {feats_path} "
                      f"({n * p * d * 2 / 1e9:.2f} GB)")

        for enc, arr in feats.items():
            memmaps[enc][cursor:cursor + b] = arr.detach().cpu().half().numpy()
        ids.extend(batch["dicom_id"])
        cursor += b

    assert cursor == n, f"Wrote {cursor} rows but dataset has {n}."
    for enc, mm in memmaps.items():
        mm.flush()
        with open(Path(output_dir) / enc / f"{split}_ids.json", "w") as f:
            json.dump(ids, f)
    print(f"[{split}] done: {cursor} images.")


def main():
    args = parse_args()
    if str(args.output_dir).startswith("gs://"):
        raise ValueError(
            "--output-dir must be a mounted private filesystem path, not a gs:// URI. "
            "Upload the completed cache with private GCS tooling afterwards."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("Feature precomputation requires a CUDA GPU.")

    cfg = Config(argparse.Namespace(cfg_path=args.cfg_path, options=args.options))
    enabled = {
        "biovil": bool(cfg.config.model.encoders.get("biovil", False)),
        "pubmedclip": bool(cfg.config.model.encoders.get("pubmedclip", False)),
        "swin": bool(cfg.config.model.encoders.get("swin", False)),
        "raddino": bool(cfg.config.model.encoders.get("raddino", False)),
    }
    if not any(enabled.values()):
        raise ValueError("Enable at least one frozen vision encoder before precomputing.")
    print(f"Enabled encoders to cache: {[k for k, v in enabled.items() if v]}")

    task = tasks.setup_task(cfg)
    model = task.build_model(cfg)
    model.cuda()
    model.eval()

    # Training samples one anchor per study, but its auxiliary DICOM must also
    # be present in every cache. Switch only dataset construction to image-row
    # mode after the model has consumed its multi-view configuration.
    OmegaConf.update(cfg.config, "model.data.study_sampling", False, merge=False)
    OmegaConf.update(cfg.config, "model.multi_view", False, merge=False)
    OmegaConf.update(
        cfg.config,
        "datasets.mimic_cxr.vis_processor.train.augmentation.enabled",
        False,
        merge=False,
    )

    for split in args.splits:
        dataset = MIMIC_CXR_Dataset(
            vis_processor=None, text_processor=None,
            vis_root=VIS_ROOT, split=split, cfg=cfg, truncate=None,
        )
        precompute_split(
            model, dataset, split, args.output_dir, enabled,
            args.batch_size, args.num_workers,
        )

    print(f"All features written to {args.output_dir}")


if __name__ == "__main__":
    main()
