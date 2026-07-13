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
rad-dino ~42 GB. **Publish `<cache_dir>` as a Kaggle Dataset** and attach it as
`/kaggle/input/...` (read-only, doesn't count against the 20 GB working quota).

### ⚠️ Disk: all three at once may not fit `/kaggle/temp`

Staging biovil+swin+rad-dino together is ~56 GB, but Kaggle ephemeral disk is
~57 GB total and `/kaggle/working` already claims ~20 GB — precomputing all three
in one run will likely overflow mid-write.

- **biovil + swin only (~14 GB): fits trivially — ship this first.**
- **rad-dino:** pick one:
  - **Keep rad-dino live** (drop `model.encoders.raddino` from the precompute and
    from `run.feature_cache_dir`'s encoder set). Zero disk risk; you still cache the
    other two.
  - **Precompute per-encoder and publish each as its own Kaggle Dataset**, then in
    training symlink the `<encoder>/` subdirs into one staging dir so the single
    `feature_cache_dir` loader sees them all:
    ```python
    import os
    os.makedirs("/kaggle/temp/feat_cache", exist_ok=True)
    for enc, ds in {
        "biovil":  "/kaggle/input/meta-cxr-feat-biovil/biovil",
        "swin":    "/kaggle/input/meta-cxr-feat-swin/swin",
        "raddino": "/kaggle/input/meta-cxr-feat-raddino/raddino",
    }.items():
        os.symlink(ds, f"/kaggle/temp/feat_cache/{enc}")
    # then: run.feature_cache_dir=/kaggle/temp/feat_cache
    ```

## Step 1 — Precompute (run once per encoder set)

Add a Kaggle cell after Cell 5 (env_config) and **before** training:

```python
import subprocess, sys, os
env = os.environ.copy()
env["PYTHONPATH"] = "/kaggle/working/META-CXR"
subprocess.run([
    sys.executable, "-m", "pretraining.precompute_features",
    "--cfg-path", "pretraining/configs/mimic_cxr_2gpu.yaml",
    "--output-dir", "/kaggle/temp/feat_cache",
    "--splits", "train", "val",
    "--batch-size", "32",
    "--num-workers", "4",
    "--options",
    "model.encoders.biovil=true",
    "model.encoders.swin=true",
    "model.encoders.raddino=true",
    "run.distributed=false", "run.world_size=1", "run.gpu=0",
], cwd="/kaggle/working/META-CXR", env=env, check=True)
```

> `run.gpu=0` is required: the standalone script does not call
> `init_distributed_mode`, so `cfg.run_cfg.gpu` must be set explicitly.

Then save `/kaggle/temp/feat_cache` as a Kaggle Dataset (e.g. slug
`meta-cxr-feat-cache`) and attach it to the training notebook.

## Step 2 — Enable cache in training

In **Cell 6**, add `run.feature_cache_dir=...` to the `cfg_options` list, pointing
at the attached cache dataset:

```python
cfg_options.append("run.feature_cache_dir=/kaggle/input/meta-cxr-feat-cache")
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
