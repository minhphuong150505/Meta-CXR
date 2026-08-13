> Source: `pretraining/configs/mimic_cxr_full.yaml` (config)
> Status: ✅ ACTIVE — ★ PRODUCTION, recipe Stage-1 duy nhất
> Last verified against source: 2026-08-13

# `pretraining/configs/mimic_cxr_full.yaml`

## Purpose
**Recipe Stage 1 production**: full MIMIC-CXR p10–p19, một GPU.

## ⚠ Default trong file không phải setting đã chạy
Đổi tên từ `mimic_cxr_full_l4.yaml` ngày 2026-08-13 (tên cũ nói NVIDIA L4 nhưng
con số memory trong file đến từ RTX 5060 Ti — máy train thật).

File ghi `batch_size_train: 8` / `accum_grad_iters: 8`, đã verify vừa 16 GB
(*"batch 8 peaked at 12,755 MB"*). Nhưng run 10-epoch tạo ra `checkpoint_best`
hiện tại chạy với override CLI `run.batch_size_train=6 run.batch_size_eval=6
run.accum_grad_iters=11`. Khi tái lập kết quả phải truyền lại ba override đó.

## Khối `model:`
| Key | Giá trị | Ghi chú |
|---|---|---|
| `arch` / `model_type` | `blip2` / `pretrain` | Registry lookup |
| `pretrained` | URL BLIP-2 | Tải lần đầu |
| `freeze_vit` | `true` | + `disabled_train` |
| `encoders` | biovil ✅ pubmedclip ✅ swin ✅ **raddino ❌** | |
| `num_query_token` / `cross_attention_freq` | 32 / 2 | |
| `max_txt_len` | 256 | |
| `mhcac.uncertain_policy` | `ignore_uncertain` | Cặp mơ hồ không phải target đáng tin |
| `mhcac.label_smoothing` | 0.05 | |
| `multi_view` | **`true`** | |
| `view_fusion.p_view_drop` | 0.15 | |
| `data.study_sampling` | `true` | ★ Một dòng = một study |
| `data.anchor_priority` | `[PA, AP, lateral]` | |
| `data.max_aux_views` | 1 | |

⚠ **`data:` nằm TRONG `model:`** — `Config` chỉ merge `run`/`model`/`datasets`.

## Trọng số loss
`itc`/`itm`/`lm`/`cls` = 1.0 · `teacher_cls`/`distill` = 0.5 ·
`mhcac_contrastive` = 0.1 · `orthogonality` = 0.05 · `sparsity` = 0.01 ·
`mpc` = 0.1 · `view_consistency` = 0.05 · `itc_queue_size` = 1024

## Khối `run:` — các key hay bị hiểu sai
| Key | Giá trị | Nghĩa thật |
|---|---|---|
| `selection_metric` | **`macro_auprc`** | ⚠ `CLAUDE.md`/`README.md` nói `f1_positive_macro` — **sai** |
| `warmup_steps` | 300 | **Optimizer update**, không phải microbatch |
| `save_freq` | 5 | Checkpoint theo epoch mỗi 5 epoch |
| `batch_size_train` × `accum_grad_iters` | 8 × 8 | Effective batch = **64** |
| `max_epoch` / `early_stop_patience` | **10 / 5** | ⚠ 5+5=10 > index cuối [9] → early stop **không thể kích hoạt**; đặt patience ≤ 4 nếu muốn nó sống |
| `eval_start_epoch` | **5** | Epoch [0]–[4] train nhưng **không** eval. Cũng chặn `checkpoint_best` trước [5], nên không epoch chưa chấm nào được tuyển |
| `amp` / `amp_dtype` | true / `bfloat16` | BF16 giữ dải mũ, tránh GradScaler sụp |
| `test_splits` | `[test]` | Held out; eval **một lần** sau train |
| `save_predictions` | `true` | ★ Cho phép calibrate threshold offline |
| `uncertain_policy` | `ignore_uncertain` | |
| `seed` | 42 | `+ get_rank()` mỗi process |

## Khối `datasets:`
`image_size: 448`, `resize_size: 512`; augmentation train: affine ±5°,
translate 0.02, scale ±0.05, `affine_p`/`jitter_p` 0.5, brightness/contrast 0.1.

## Consumer
`pretraining/train.py` · `precompute_features.py` · `run_medgemma_qlora.py:59`
(mặc định `--stage1-config`) · `scripts/vm_preflight.py:152`

## Developer notes
Comment về contrastive negative đáng nhớ: *"Contrastive negatives come from the
live microbatch plus the queue, not from gradient accumulation."* — tăng
`accum_grad_iters` **không** làm ITC có thêm negative.

← [`_index.md`](_index.md) · [HOME](../../../HOME.md)
