> Source: `cloud/env.sh` (26 dòng)
> Status: ✅ ACTIVE
> Last verified against source: 2026-08-12

# `cloud/env.sh`

## Purpose
Mặc định dùng chung cho các launcher — và **cố ý không chứa tên thật**.

## ★ Không có identity nào trong file này
Comment `:12-19` ghi rõ: repo này public, nên tên project/bucket thật phải export
từ một wrapper **untracked**:

```bash
# cloud/env.local.sh   (git-ignored)
export GCP_PROJECT=...
export GCS_DATA_BUCKET=...   # MIMIC-CXR, PhysioNet credentialed
export GCS_BUCKET=...        # checkpoint và log

source cloud/env.local.sh
```

Mọi biến ở đây dùng `${VAR:-}` nên giá trị export trước sẽ thắng.

## Biến
| Biến | Mặc định |
|---|---|
| `GCP_PROJECT`, `VM_INSTANCE`, `GCS_DATA_BUCKET`, `GCS_BUCKET` | **rỗng** |
| `GCP_ZONE` / `GCP_REGION` | `us-central1-a` / `us-central1` |
| `STAGE1_CONFIG` | `pretraining/configs/mimic_cxr_full_l4.yaml` |
| `STAGE1_RUN` | `mimic_cxr_full_l4_blip2` |
| `STAGE2_IMAGE_MODE` | `both` |

Comment `:9-11`: **cả hai bucket phải có uniform bucket-level access VÀ
public-access prevention** — `require_private_bucket` từ chối nếu không.

## Developer notes
⚠ Tên bucket trong `docs/` cũ (`gs://meta-cxr-checkpoint`) **đã bị xóa**. Đừng tin
tài liệu; export từ `env.local.sh`.

← [`_index.md`](_index.md) · [HOME](../../HOME.md)
