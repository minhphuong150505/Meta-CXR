> Source: `biovil_t/modules.py` (85 dòng)
> Status: 🟡 CONDITIONAL — downstream heads
> Last verified against source: 2026-08-12

# `biovil_t/modules.py`

## Purpose

Các head nhỏ dùng trên embedding BioViL.

| Class | Vai trò |
|---|---|
| `MLP` | Projection/classification MLP có activation/dropout tùy cấu hình |
| `MultiTaskModel` | Nhiều classifier task trên cùng representation |

`ImageModel.create_downstream_classifier()` có thể dựng `MultiTaskModel`, nhưng
đường Meta-CXR Stage 1 lấy patch embedding và dùng MHCAC, không dùng head này.

## Status

Code khả dụng qua API BioViL, nhưng conditional đối với Meta-CXR hiện tại; không
gắn nhãn unused vì external/downstream caller có thể dùng.

← [`biovil_t/`](_index.md) · [HOME](../../HOME.md)
