> Source: `biovil_t/types.py` (37 dòng)
> Status: ✅ ACTIVE
> Last verified against source: 2026-08-12

# `biovil_t/types.py`

## Purpose

Kiểu dữ liệu chung của BioViL-T.

## Main items

| Item | Vai trò |
|---|---|
| `ImageModelOutput` | Gom `img_embedding`, `patch_embeddings`, `projected_patch_embeddings`, `class_logits` |
| `ImageEncoderType` | Enum các backbone được hỗ trợ |
| `ImageEncoderType.get_members(...)` | Lọc enum theo single-image/multi-image |

`Blip2Qformer` đọc trực tiếp `projected_patch_embeddings`; đổi tên field này sẽ
phá Stage 1 dù encoder vẫn forward được.

## Called by

`biovil_t/model.py`, `encoder.py`, `pretrained.py` và
`model/lavis/models/blip2_models/blip2.py`.

← [`biovil_t/`](_index.md) · [HOME](../../HOME.md)
