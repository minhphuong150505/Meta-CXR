> Source: `cloud/run_stage2.sh` (39 dòng)
> Status: ✅ ACTIVE — ⚠ alias cũ
> Last verified against source: 2026-08-12

# `cloud/run_stage2.sh`

## Purpose
Chạy Stage 2 trên VM GCP, upload lên bucket riêng tư.

## ⚠ Dùng alias deprecated
```bash
--image-mode "$STAGE2_IMAGE_MODE"     # :34
```
`--image-mode {native,qformer,both}` là **alias cũ**. Vẫn chạy (map sang tên mode
mới qua `LEGACY_IMAGE_MODE_ALIASES`) nhưng nên chuyển sang `--pipeline-mode`.
[I7](../_meta/LEGACY_AND_OPTIONAL.md#-potential-issues--ghi-nhận-không-sửa).

`:23-26` validate `STAGE2_IMAGE_MODE` phải là `qformer|native|both`, nếu không dừng.

## Execution flow
```text
env.sh + lib/common.sh → require_gcp_config → require_private_bucket × 2
   ↓ validate STAGE2_IMAGE_MODE
python training/run_medgemma_qlora.py \
   --checkpoint-root … --stage1-run … --stage1-config … \
   --image-mode … --output-dir … --gcs-output … "$@"
```
`"$@"` cho phép truyền thêm flag từ dòng lệnh.

## Environment
`CHECKPOINT_ROOT`, `STAGE2_OUTPUT_DIR`, `STAGE2_IMAGE_MODE` (mặc định `both`),
`STAGE1_RUN`, `STAGE1_CONFIG`, `PYTHON_BIN`.

## Developer notes
⚠ `PYTHON_BIN` nên trỏ **venv Stage 2** theo setup khuyến nghị. Lock Stage 2 hiện
include pin Stage 1 rồi thêm QLoRA packages; tách venv để cách ly dependency nặng.
Stage 2 **không có DDP**; script này chạy một GPU.

← [`_index.md`](_index.md) · [HOME](../../HOME.md)
