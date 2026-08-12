> Source: `cloud/run_stage1.sh` (31 dòng)
> Status: ✅ ACTIVE
> Last verified against source: 2026-08-12

# `cloud/run_stage1.sh`

## Purpose
Chạy Stage 1 trên VM GCP và upload **chỉ** model/log artifact lên bucket riêng tư.

## Execution flow
```text
source env.sh + lib/common.sh
   ↓
require_gcp_config
require_private_bucket "$GCS_BUCKET"        ← từ chối bucket không riêng tư
require_private_bucket "$GCS_DATA_BUCKET"
   ↓
RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)
   ↓
python -m pretraining.train --cfg-path "$STAGE1_CONFIG" \
       --options run.output_dir=$OUTPUT_BASE run.run_name=$STAGE1_RUN
   ↓
test -f "$RUN_DIR/checkpoint_best.pth"      ← FAIL nếu không có
   ↓
upload_gcs "$RUN_DIR" "gs://$GCS_BUCKET/stage1/$STAGE1_RUN/$RUN_ID"
```

`set -euo pipefail` — lỗi không bị nuốt.

## Environment
`GCP_PROJECT`, `GCS_BUCKET`, `GCS_DATA_BUCKET` (từ `cloud/env.local.sh` untracked),
`STAGE1_CONFIG`, `STAGE1_RUN`, `PYTHON_BIN`, `OUTPUT_BASE`.

## Failure points
Bucket không riêng tư → dừng · `checkpoint_best.pth` không tồn tại → `test -f` fail
→ **không upload** · Thiếu `GCP_PROJECT` → `require_gcp_config` dừng

## Developer notes
⚠ `PYTHON_BIN` mặc định `python3` — phải export trỏ đúng **venv Stage 1**.
Kiểm `checkpoint_best.pth` trước upload là chốt tránh upload một run hỏng.

← [`_index.md`](_index.md) · [HOME](../../HOME.md)
