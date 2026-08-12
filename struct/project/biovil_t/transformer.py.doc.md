> Source: `biovil_t/transformer.py` (266 dòng)
> Status: 🟡 CONDITIONAL — temporal/multi-image path
> Last verified against source: 2026-08-12

# `biovil_t/transformer.py`

## Purpose

Transformer/pooler cho biến thể BioViL temporal: attention giữa current image và
previous image, kèm positional/type embedding.

## Main items

| Item | Vai trò |
|---|---|
| `VisionTransformerPooler` | Pool current/previous image token |
| `MultiHeadAttentionLayer` | Attention trả output + attention weights |
| `Block` | Attention + MLP residual |
| `SinePositionEmbedding` | Positional encoding từ mask |

## Status in Meta-CXR

Stage 1 multi-view hiện dùng `mhcac/view_fusion.py` trên từng encoder stream,
không truyền previous image qua `MultiImageModel`; file này vẫn là dependency của
thư viện BioViL và conditional, không phải component view-fusion production.

## Modification risk

Đừng nhầm `VisionTransformerPooler` với Q-Former hoặc `ViewFusionModule`; ba khối
có lifecycle và checkpoint khác nhau.

← [`biovil_t/`](_index.md) · [HOME](../../HOME.md)
