# `vision_encoders/biovil_tokens.py`

> Thư mục: [`vision_encoders/`](_index.md) · Thêm 2026-10-11 · ✅ dùng trong `Blip2Qformer`

## Purpose
Đổi lưới patch của BioViL-T (`projected_patch_embeddings`, **`[B, D, H, W]`, kênh đứng
trước**) thành chuỗi token `[B, H·W, D]`. Có hai bố cục, chọn bằng
`model.biovil.token_layout`:

| Giá trị | Phép biến đổi | Ý nghĩa |
|---|---|---|
| vắng khóa / `legacy_reshape` | `grid.reshape(B, -1, D)` | hành vi của code gốc META-CXR và mọi checkpoint tới 2026-10-11; **không** hoán vị trục nên mỗi "token" là ~7 bản đồ kênh duỗi phẳng, một ô ảnh bị rải ra cả 196 token |
| `patch` | `grid.flatten(2).transpose(1, 2)` | token *k* là embedding của ô ảnh *k* (theo hàng); trung bình token = embedding toàn cục của BioViL |

## Hàm
- `resolve_biovil_token_layout(value)` — `None` → `legacy_reshape`; giá trị lạ → `ValueError`.
- `biovil_grid_to_tokens(grid, layout)` — yêu cầu tensor 4 chiều.

## Nơi gọi
`Blip2Qformer._biovil_tokens` (anchor và ảnh phụ) và, qua nó,
`pretraining/precompute_features.py`. Checkpoint và cache đặc trưng chỉ đúng với bố cục
đã dùng khi tạo ra chúng.

## Test
`tests/test_biovil_token_layout.py`: mặc định trùng từng bit với `reshape` cũ; `patch` đặt
đúng một ô ảnh vào một token.
