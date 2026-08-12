> Source: `cloud/setup_vm.sh` (46 dòng)
> Status: 🧰 UTILITY — one-time VM setup
> Last verified against source: 2026-08-12

# `cloud/setup_vm.sh`

## Purpose

Chuẩn bị VM GCP riêng tư: cài system package, chọn project, tạo artifact bucket
nếu thiếu, cưỡng chế privacy cho cả data/artifact bucket và tạo prefix Stage 1/2.

## Execution flow

```text
require_gcp_config
  → apt-get install python3-pip/venv, jq, curl, libgl1, libglib2.0-0
  → gcloud config set project
  → create GCS_BUCKET nếu thiếu (uniform bucket-level access)
  → enforce_private_bucket(GCS_BUCKET, GCS_DATA_BUCKET)
  → tạo stage1/.keep và stage2/.keep
```

## Side effects / permissions

Cần `sudo`, thay active gcloud project, có thể tạo/chỉnh policy bucket và ghi
object. Script **không** tạo venv Python; chỉ nhắc cài hai environment riêng.

## Failure points

Thiếu auth/quyền IAM, thiếu biến môi trường, apt/gcloud lỗi hoặc bucket data không
tồn tại/không thể cưỡng chế private → dừng bởi `set -euo pipefail`.

← [`cloud/`](_index.md) · [HOME](../../HOME.md)
