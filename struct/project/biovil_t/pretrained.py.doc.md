> Source: `biovil_t/pretrained.py` (85 dòng)
> Status: ✅ ACTIVE — checkpoint loader
> Last verified against source: 2026-08-12

# `biovil_t/pretrained.py`

## Purpose

Dựng BioViL/BioViL-T đúng kiến trúc và tải weight pretrained về cache cục bộ.

## Main functions

| Function | Vai trò |
|---|---|
| `_download_biovil_image_model_weights()` | Tải weight BioViL cũ |
| `_download_biovil_t_image_model_weights()` | Tải weight BioViL-T |
| `get_biovil_image_encoder(pretrained=True)` | Dựng encoder BioViL |
| `get_biovil_t_image_encoder()` | ★ Dựng encoder được Stage 1 dùng |

## Data / side effects

Lần chạy đầu có network I/O và ghi weight vào cache. `get_biovil_t_image_encoder`
nạp state dict rồi trả `ImageModel`; caller phía LAVIS tiếp tục freeze model.

## Called by

`model/lavis/models/blip2_models/blip2.py::init_vision_encoder`.

## Error cases

Mạng bị chặn, cache không ghi được hoặc weight không khớp kiến trúc → khởi tạo
Stage 1 fail trước training.

← [`biovil_t/`](_index.md) · [HOME](../../HOME.md)
