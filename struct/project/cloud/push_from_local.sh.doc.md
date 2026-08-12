> Source: `cloud/push_from_local.sh` (31 dòng)
> Status: 🧰 UTILITY
> Last verified against source: 2026-08-12

# `cloud/push_from_local.sh`

## Purpose

Cập nhật checkout trên VM bằng `git pull --ff-only` qua `gcloud compute ssh`.
Tên gây hiểu nhầm: script **không push/copy file local** và không truyền data,
checkpoint hay credential.

## Inputs

`VM_INSTANCE` bắt buộc; `GCP_PROJECT`, `GCP_ZONE`; `REMOTE_REPO_DIR` mặc định
`META-CXR` và phải là relative shell-safe path không chứa `..`.

## Side effects

Thay working tree remote nếu pull fast-forward thành công. Local workspace không
bị sửa.

← [`cloud/`](_index.md) · [HOME](../../HOME.md)
