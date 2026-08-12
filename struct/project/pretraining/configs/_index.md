> Source: `pretraining/configs/`
> Status: ✅ ACTIVE + completed inference ablation
> Last verified against source: 2026-08-12

# `pretraining/configs/`

## Purpose

Siêu tham số của từng Stage-1 run. Tách khỏi `configs/env_config.yaml` — file đó
giữ đường dẫn **của máy**, file ở đây giữ cấu hình **của thí nghiệm**.

## Role in project

```text
configs/env_config.yaml  ──► local_config.py  ──► ĐƯỜNG DẪN (mỗi máy khác nhau)
pretraining/configs/*.yaml ──► LAVIS Config   ──► SIÊU THAM SỐ (mỗi run khác nhau)
```

## Parent

[`pretraining/`](../_index.md)

## Children

| File | Doc | Status | Dùng cho |
|---|---|---|---|
| `mimic_cxr_full_l4.yaml` | [📄](mimic_cxr_full_l4.yaml.doc.md) | ✅ ★ | **PRODUCTION**, 1 GPU |
| `mimic_cxr_2x3090.yaml` | [📄](mimic_cxr_2x3090.yaml.doc.md) | ✅ | 2-GPU DDP |
| `blip2_pretrain_stage1_emb.yaml` | [📄](blip2_pretrain_stage1_emb.yaml.doc.md) | ✅ | Demo Gradio (`inference.sh:7`) |
| `blip2_pretrain_stage1.yaml` | — | 🕰 | Zero reference |
| `mimic_cxr_2gpu.yaml` | — | 🕰 | 2×T4 cũ ⚠ `warmup_steps: 32000` |
| `encoder_comparison/07_all_three.yaml` | [📄](encoder_comparison/07_all_three.yaml.doc.md) | 🧪 | So sánh encoder; resolve theo `<run_name>.yaml` tại đây |
| `ablation/{biovil,pubmedclip,swin,all_three}.yaml` | [📄](ablation/_index.md) | ✅ | Table 5, inference-only; 4/4 completed |

## Main responsibilities

Khai báo ba khối mà `Config` merge: `model:`, `datasets:`, `run:`.

## Entry points

Không phải entrypoint; được truyền qua `--cfg-path`.

## Dependencies

`model/lavis/common/config.py::Config` · `omegaconf`

## Used by

`pretraining/train.py`, `pretraining/precompute_features.py`,
`training/run_medgemma_qlora.py:59`, `training/stage1/lavis_loader.py:50`,
`training/train_eval_figure9_llm_variants_200.py:324`, `cloud/env.sh:24`,
`scripts/vm_preflight.py:152`, `inference.sh:7`

## Execution flow

```text
--cfg-path <yaml>
   ↓
Config(args)   ← merge CHỈ run / model / datasets
   ↓
cfg.model_cfg  →  Blip2Qformer.from_config()
cfg.run_cfg    →  RunnerBase
cfg.datasets_cfg → vis_processor / text_processor
```

## Important configurations

### ⚠ Bẫy: `data:` phải nằm TRONG `model:`

`Config` chỉ merge ba block gốc. Đặt `data:` ở top level → **bị bỏ qua âm thầm**,
không có cảnh báo.

```yaml
model:
  data:                       # ← ĐÚNG
    study_sampling: true
    anchor_priority: [PA, AP, lateral]
    max_aux_views: 1
```

### Khác biệt giữa production và legacy

| Key | `full_l4` ✅ | `2gpu` 🕰 |
|---|---|---|
| `multi_view` | `true` | `false` |
| `warmup_steps` | `300` | ⚠ `32000` — **không bao giờ hoàn tất ramp** |
| `lambda_mpc` | `0.1` | `0.0` |
| `selection_metric` | `macro_auprc` | ⚠ khác |
| `amp_dtype` | `bfloat16` | ⚠ khác |

### Các key hay bị hiểu sai

| Key | Nghĩa thật |
|---|---|
| `warmup_steps` | **Optimizer update**, không phải microbatch |
| `save_freq: 5` | Checkpoint theo epoch mỗi 5 epoch; `0` → chỉ giữ best + last |
| `selection_metric: macro_auprc` | ⚠ `CLAUDE.md`/`README.md` nói `f1_positive_macro` — **sai**, config thắng |
| `batch_size_train: 8` + `accum_grad_iters: 8` | Effective batch = 64 |
| `test_splits: [test]` | Được giữ ngoài chọn checkpoint; eval **một lần** sau train |

## Status

```text
✅ ACTIVE — mimic_cxr_full_l4.yaml, mimic_cxr_2x3090.yaml, blip2_pretrain_stage1_emb.yaml
🧪 encoder_comparison/07_all_three.yaml
✅ ablation/*.yaml — evaluation hoàn tất; không retrain
🕰 blip2_pretrain_stage1.yaml, mimic_cxr_2gpu.yaml
```

## Notes

- ⚠ **Tên file nói dối:** `mimic_cxr_full_l4.yaml` gợi ý NVIDIA L4, nhưng comment
  trong file ghi *"Verified on RTX 5060 Ti 16 GB"*. Con số batch/memory trong đó
  đến từ 5060 Ti, không phải L4.

- **`encoder_comparison/` được resolve theo quy ước tên.** `lavis_loader.py:50`
  làm `PROJECT_DIR / "pretraining/configs/encoder_comparison" / f"{run_name}.yaml"`.
  Nên `--stage1-run 07_all_three` sẽ tìm file ở **đó**, không phải ở thư mục cha.
  Còn `--stage1-config` mặc định lại trỏ vào `mimic_cxr_full_l4.yaml`. Hai đường
  resolve khác nhau — dễ nhầm.

- **`ablation/` không phải encoder training recipe.** Bốn file đều dựng checkpoint
  ba encoder, đặt `active_encoders`, để `train_splits`/`valid_splits` rỗng và chỉ
  chạy test. Dùng chúng để tái lập Table 5, không dùng để huấn luyện encoder đơn.

- **`blip2_pretrain_stage1_emb.yaml` KHÔNG legacy** dù tên giống file legacy bên
  cạnh: `inference.sh` dùng nó ([D-002](../../_meta/DECISIONS.md#d-002--đường-vicuna-7b-legacy-vẫn-là-demo-active)).

## Related documentation

[ARCHITECTURE.md §6](../../_meta/ARCHITECTURE.md#6-cấu-hình-phân-tầng) ·
[PIPELINES.md → P1](../../_meta/PIPELINES.md#p1--stage-1-pretraining) ·
[`configs/_index.md`](../../configs/_index.md)

← [`pretraining/`](../_index.md) · [HOME](../../../HOME.md)
