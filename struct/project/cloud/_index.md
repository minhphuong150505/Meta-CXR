> Source: `cloud/` (11 shell script)
> Status: ✅ ACTIVE (launcher) / 🧪 (wrapper)
> Last verified against source: 2026-08-12

# `cloud/`

## Purpose

Launcher chạy training/eval trên VM GCP và upload artifact lên GCS **riêng tư**.

## ⚠ Không script nào chứa tên project/bucket thật

Remote này public. Identity đến từ biến môi trường, nạp từ một file **untracked**:

```bash
# cloud/env.local.sh   (git-ignored)
export GCP_PROJECT=...
export GCS_DATA_BUCKET=...   # MIMIC-CXR, PhysioNet credentialed
export GCS_BUCKET=...        # checkpoint và log

source cloud/env.local.sh && cloud/run_stage1.sh
```

`cloud/env.sh` chỉ chứa **mặc định rỗng** và hướng dẫn.

## `require_private_bucket` — chốt chặn thật, không phải kiểm tra hình thức

`cloud/lib/common.sh` **từ chối** bucket không bật đồng thời:
- uniform bucket-level access
- public-access prevention

Cả `run_stage1.sh` và `run_stage2.sh` gọi nó cho **cả hai** bucket trước khi làm
bất cứ việc gì.

## Parent

[`struct/project/`](../../HOME.md#source-code-tree)

## Children

| File | Doc | Status | Vai trò |
|---|---|---|---|
| `run_stage1.sh` (31) | [📄](run_stage1.sh.doc.md) | ✅ | `python -m pretraining.train` + upload |
| `run_stage2.sh` (39) | [📄](run_stage2.sh.doc.md) | ✅ | `python training/run_medgemma_qlora.py` ⚠ dùng alias cũ `--image-mode` |
| `env.sh` (26) | [📄](env.sh.doc.md) | ✅ | Mặc định rỗng + hướng dẫn |
| `lib/common.sh` (102) | [📄](lib/common.sh.doc.md) | 🧰 | `log`, `require_gcp_config`, `require_private_bucket`, `upload_gcs` |
| `setup_vm.sh` (46) | [📄](setup_vm.sh.doc.md) | 🧰 | Cài system package + enforce bucket private |
| `push_from_local.sh` (31) | [📄](push_from_local.sh.doc.md) | 🧰 | SSH rồi `git pull --ff-only`; không copy local data |
| `run_encoder_comparison.sh` (6) | [📄](run_encoder_comparison.sh.doc.md) | 🕰 | Compatibility alias → `run_stage1.sh`; không sweep |
| `run_medgemma_pipeline.sh` (5) | [📄](run_medgemma_pipeline.sh.doc.md) | 🕰 | Compatibility alias → `run_stage2.sh` |
| `run_medgemma_l4_bucket_pipeline.sh` (5) | [📄](run_medgemma_l4_bucket_pipeline.sh.doc.md) | 🕰 | Compatibility alias; không tự chọn L4/bucket |
| `run_medgemma_qformer_eval.sh` (7) | [📄](run_medgemma_qformer_eval.sh.doc.md) | 🕰 | Đặt alias `qformer`, rồi chạy full Stage 2 |
| `run_paper_assets.sh` (37) | [📄](run_paper_assets.sh.doc.md) | ⚠ | Gọi `paper_assets.py` không tồn tại trong repo |

## Execution flow — `run_stage1.sh`

```text
source env.sh + lib/common.sh
   ↓
require_gcp_config
require_private_bucket "$GCS_BUCKET"
require_private_bucket "$GCS_DATA_BUCKET"
   ↓
RUN_ID = $(date -u +%Y%m%dT%H%M%SZ)
   ↓
python -m pretraining.train --cfg-path "$STAGE1_CONFIG" \
       --options run.output_dir=… run.run_name=…
   ↓
test -f "$RUN_DIR/checkpoint_best.pth"     ← FAIL nếu không có
   ↓
upload_gcs "$RUN_DIR" "gs://$GCS_BUCKET/stage1/$STAGE1_RUN/$RUN_ID"
```

`set -euo pipefail` ở mọi script — lỗi không bị nuốt.

## Main responsibilities

1. Cưỡng chế bucket riêng tư trước khi chạm dữ liệu.
2. Chạy Stage 1 / Stage 2 với output có timestamp.
3. Upload **chỉ** model/log artifact.
4. Chuẩn bị VM.
5. Duy trì các tên launcher cũ bằng compatibility alias.

## Entry points

Xem [ENTRYPOINTS.md](../_meta/ENTRYPOINTS.md#shell--launcher).

## Dependencies

`gcloud`/`gsutil` · `bash` · `python3` (qua `$PYTHON_BIN`) · `jq`, `curl` (setup_vm)

## Important configurations

| Biến | Mặc định | |
|---|---|---|
| `STAGE1_CONFIG` | `pretraining/configs/mimic_cxr_full_l4.yaml` | |
| `STAGE1_RUN` | `mimic_cxr_full_l4_blip2` | |
| `STAGE2_IMAGE_MODE` | `both` | ⚠ giá trị alias cũ: `qformer`/`native`/`both` |
| `GCP_ZONE` / `GCP_REGION` | `us-central1-a` / `us-central1` | |
| `PYTHON_BIN` | `python3` | ⚠ phải trỏ đúng venv Stage 1 **hoặc** Stage 2 |

## Status

```text
✅ ACTIVE — run_stage1.sh, run_stage2.sh, env.sh, lib/common.sh
🧰 UTILITY — setup_vm.sh, push_from_local.sh
🕰 COMPATIBILITY — 4 alias mỏng
⚠ BROKEN / POTENTIALLY_UNUSED — run_paper_assets.sh (thiếu paper_assets.py)
```

## Notes

- ⚠ **`run_stage2.sh:34` dùng `--image-mode`** — alias deprecated. Vẫn chạy (map
  sang tên mode mới) nhưng nên chuyển sang `--pipeline-mode`. [I7](../_meta/LEGACY_AND_OPTIONAL.md#-potential-issues--ghi-nhận-không-sửa).

- ⚠ **`PYTHON_BIN` mặc định `python3`** — cloud setup khuyến nghị hai venv để
  cách ly Stage 1 và Stage 2. Export `PYTHON_BIN` trỏ đúng environment của run.

- **`cloud/outputs/` bị git-ignore.**

- ⚠ Bucket ghi trong tài liệu cũ (`gs://meta-cxr-checkpoint`) **đã bị xóa**. Đừng
  tin tên bucket trong `docs/` — export từ `env.local.sh`.

- `run_encoder_comparison.sh` và `run_medgemma_qformer_eval.sh` không thực hiện
  đúng nghĩa literal của tên: file đầu chỉ gọi Stage 1 production; file sau có
  thể train Stage 2, không phải eval-only.

## Related documentation

[ENTRYPOINTS.md](../_meta/ENTRYPOINTS.md) · [PIPELINES.md](../_meta/PIPELINES.md) ·
[PROJECT_OVERVIEW.md §3](../_meta/PROJECT_OVERVIEW.md#3-dữ-liệu)

← [Về HOME](../../HOME.md)
