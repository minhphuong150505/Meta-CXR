> Source: `run_container.sh` (25 dòng)
> Status: ✅ ACTIVE — demo launcher
> Last verified against source: 2026-08-12

# `run_container.sh`

## Purpose

Xóa container demo cùng tên rồi chạy `meta-cxr:2.0.0`, mount repository hiện tại
vào `/workspace/META-CXR` và cấp toàn bộ GPU.

## Important flags

| Flag | Effect |
|---|---|
| `--restart=unless-stopped` | Container tự chạy lại |
| `--privileged` | ⚠ quyền host rất rộng |
| `--gpus all` | Cấp mọi GPU cho container |
| `-v "$PWD:/workspace/META-CXR"` | Mount writable toàn repo |

Script không publish `-p 7860:7860`; UI hiện dựa vào `demo.launch(share=True)` để
tạo Gradio share URL công khai. Xem I11/I13 trong
[Legacy & Optional](_meta/LEGACY_AND_OPTIONAL.md#-potential-issues--ghi-nhận-không-sửa).

## Destructive behavior

Nếu container `meta-cxr-container` đã tồn tại, script `docker stop` rồi
`docker rm` nó trước khi chạy container mới.

← [HOME](../HOME.md) · [`Dockerfile`](Dockerfile.doc.md)
