> **KHÔNG publish feature cache.** Feature là dẫn xuất trực tiếp từ ảnh MIMIC-CXR
> và PhysioNet DUA cấm phân phối lại; cache chỉ được đặt trên đĩa cục bộ có kiểm
> soát truy cập của máy train.

# Frozen-Encoder Feature Cache

All vision encoders (BioViL, Swin, rad-dino) are **frozen** and the training
transform is **fully deterministic** (no augmentation — verified in
`model/lavis/data/ReportDataset.py`, `vis_augs` is defined but never applied).
So each frozen encoder produces the **same output for a given image in every
epoch**. Recomputing them every step × 15 epochs is the dominant training cost.

This cache precomputes the raw frozen-encoder outputs once and lets training read
them instead of running the encoders — removing both the encoder forward and the
448px JPG decode from every step.

## What was added

| File | Change |
|------|--------|
| `pretraining/precompute_features.py` | **New.** One-shot script: runs each enabled frozen encoder over train/val and writes raw outputs to disk, keyed by `dicom_id`. |
| `model/lavis/data/ReportDataset.py` | When `run.feature_cache_dir` is set, `__getitem__` returns cached `*_feat` tensors and skips JPG decode. Default (no flag) is unchanged. |
| `model/lavis/models/blip2_models/blip2_qformer.py` | `_encode_image_streams` / `forward` use cached features when present; the **trainable** `ln_vision` / `swin_qformer_proj` / `raddino_qformer_proj` still run, so training is identical. |

The cache stores the **raw frozen output BEFORE** the trainable projection layers,
so it is correct to train on. Cache is **per-encoder**, so any of the 9 encoder
toggle combinations reuses the same files.

```
<cache_dir>/biovil/<split>_feats.npy    # memmap (N, P, D) float16
<cache_dir>/biovil/<split>_ids.json     # list[dicom_id], row order
<cache_dir>/swin/...
<cache_dir>/raddino/...
```

Estimated size (P10, ~18k train + ~2.1k val, float16): BioViL ~11 GB, Swin ~3 GB,
rad-dino ~42 GB. Scale that up for the full p10–p19 split before choosing a
target disk. Keep `<cache_dir>` on the training host's own storage only — it is a
MIMIC-CXR derivative and must never be published.

## Step 1 — Precompute (run once per encoder set)

```bash
CUDA_VISIBLE_DEVICES=0 python -m pretraining.precompute_features \
    --cfg-path pretraining/configs/mimic_cxr_full.yaml \
    --output-dir /mnt/drive1tb/meta-cxr-feat-cache \
    --splits train val \
    --batch-size 32 \
    --num-workers 4 \
    --options model.encoders.biovil=true model.encoders.swin=true \
              model.encoders.raddino=true \
              run.distributed=false run.world_size=1 run.gpu=0
```

> `run.gpu=0` is required: the standalone script does not call
> `init_distributed_mode`, so `cfg.run_cfg.gpu` must be set explicitly.

Precompute one encoder at a time if disk is tight; the loader reads one
`<cache_dir>` containing a per-encoder subdirectory each, so separate runs
writing into the same `--output-dir` compose without extra work.

## Step 2 — Enable cache in training

Pass `run.feature_cache_dir` in the training `--options`:

```bash
--options run.feature_cache_dir=/mnt/drive1tb/meta-cxr-feat-cache
```

The enabled encoders must match what was precomputed. With the cache active the
dataloader no longer decodes JPGs and the encoders never run — only the Q-Former +
MHCAC + projections train.

## Caveats

- **Re-precompute** if you change `image_size`, the preprocessing in
  `general_trans`, the split CSVs, or any frozen-encoder weights/`model_name`.
- **PubMedCLIP is not cached** (its stream needs the live image). Don't combine
  `model.encoders.pubmedclip=true` with the cache.
- The cache is keyed by `dicom_id`, so it is robust to row ordering between
  precompute and training as long as the same split CSV is used.
