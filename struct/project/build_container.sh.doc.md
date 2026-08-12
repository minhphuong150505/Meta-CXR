> Source: `build_container.sh` (18 dòng)
> Status: ✅ ACTIVE — demo launcher
> Last verified against source: 2026-08-12

# `build_container.sh`

## Purpose

Chuẩn bị base image `meta-cxr:1.0.0` (pull `dasithdev/meta-cxr:1.0.0` nếu local
chưa có), rồi build `Dockerfile` thành `meta-cxr:2.0.0`.

## Flow / side effects

```text
docker images -q meta-cxr:1.0.0
  ├─ thiếu → docker pull + docker tag
  └─ có   → dùng local
  → docker build -t meta-cxr:2.0.0 -f Dockerfile .
```

Có network I/O, ghi Docker image/cache. Script không có `set -e`; lỗi pull/tag có
thể không dừng ngay trước lệnh build.

## Security note

Base image đến từ Docker Hub account bên ngoài repo. Cần kiểm provenance/digest
trước môi trường nhạy cảm; tag `1.0.0` không immutable theo nội dung.

← [HOME](../HOME.md) · [`Dockerfile`](Dockerfile.doc.md)
