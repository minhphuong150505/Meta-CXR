> Source: `cloud/lib/common.sh` (102 dòng)
> Status: 🧰 UTILITY — ★ chốt chặn dữ liệu
> Last verified against source: 2026-08-12

# `cloud/lib/common.sh`

## Purpose
Hàm dùng chung cho mọi launcher — trong đó có **chốt chặn dữ liệu PhysioNet**.

## Main functions
| Hàm | Vai trò |
|---|---|
| `log` | In có timestamp |
| `require_gcp_config` | Dừng nếu thiếu `GCP_PROJECT` / bucket |
| `require_private_bucket <bucket>` | ★ **Từ chối bucket không riêng tư** |
| `enforce_private_bucket <bucket>` | Bật uniform access + public-access prevention, rồi verify |
| `upload_gcs <src> <dst>` | Upload |

## ★ `require_private_bucket` — không phải kiểm tra hình thức
Từ chối bucket không bật **đồng thời**:
- uniform bucket-level access
- public-access prevention

Cả `run_stage1.sh` và `run_stage2.sh` gọi nó cho **cả hai** bucket **trước khi làm
bất cứ việc gì**. Một checkpoint hay log lọt lên bucket public là vi phạm DUA.

## Called by
`cloud/run_stage1.sh`, `run_stage2.sh`, `setup_vm.sh`, `run_paper_assets.sh` và
các compatibility wrapper.

## Failure points
`gcloud`/`gsutil` chưa auth → lệnh kiểm bucket fail → script dừng (`set -euo pipefail`).

## Developer notes
**Đừng thêm cờ bỏ qua kiểm tra này.** Nếu bucket không đạt, sửa bucket, đừng sửa
script.

← [`_index.md`](../_index.md) · [HOME](../../../HOME.md)
