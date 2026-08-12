> Source: `biovil_t/model.py` (128 dòng)
> Status: ✅ ACTIVE — model facade
> Last verified against source: 2026-08-12

# `biovil_t/model.py`

## Purpose

Facade cấp cao của BioViL: nối image encoder, pooler/projection và classifier tùy
chọn, rồi trả `ImageModelOutput` có cả embedding toàn ảnh và patch.

## Main classes

| Class / method | Vai trò |
|---|---|
| `BaseImageModel` | API trừu tượng cho forward và patch projection |
| `ImageModel.forward(x)` | Encoder → `forward_post_encoder` |
| `ImageModel.forward_post_encoder(...)` | Projection + output object |
| `ImageModel.get_patchwise_projected_embeddings(...)` | ★ patch token cho Meta-CXR |
| `MultiImageModel` | Biến thể current/previous image; không nằm trên đường Stage 1 hiện tại |

## Data flow

```text
image [B,C,H,W]
  → ImageEncoder
  → patch_x + pooled_x
  → projection
  → ImageModelOutput.projected_patch_embeddings
  → Blip2Qformer
```

Shape số patch phụ thuộc backbone/input; chiều projected của checkpoint BioViL-T
được Stage 1 kỳ vọng là `1408`.

## Called by

`biovil_t/pretrained.py`; output được đọc trong
`Blip2Qformer._encode_image_streams` và `_encode_aux_streams`.

← [`biovil_t/`](_index.md) · [HOME](../../HOME.md)
