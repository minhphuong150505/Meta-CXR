> Source: `biovil_t/encoder.py` (180 dòng)
> Status: ✅ ACTIVE
> Last verified against source: 2026-08-12

# `biovil_t/encoder.py`

## Purpose

Adapter quanh backbone ảnh, chuẩn hóa output thành patch feature và pooled feature.

## Main items

| Item | Vai trò |
|---|---|
| `ImageEncoder` | Dựng/forward một ảnh |
| `MultiImageEncoder` | Ghép current + previous image cho biến thể temporal |
| `get_encoder_output_dim(...)` | Probe chiều output mà vẫn phục hồi train/eval mode |
| `restore_training_mode(module)` | Context manager bảo toàn trạng thái `.training` |
| `get_encoder_from_type(...)` | Factory theo `ImageEncoderType` |

## Important behavior

`reload_encoder_with_dilation` thay stride bằng dilation; đổi nó làm thay đổi số
patch downstream. `get_encoder_output_dim` tạo forward probe, nên phải chạy trên
đúng device và không được để module sai mode sau khi đo.

## Called by

`biovil_t/model.py`, `pretrained.py`; Stage 1 dùng gián tiếp qua `ImageModel`.

← [`biovil_t/`](_index.md) · [HOME](../../HOME.md)
