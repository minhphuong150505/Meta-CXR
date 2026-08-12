> Source: `vision_encoders/shared_visual_tokens.py` (193 dòng)
> Status: ✅ ACTIVE — ★
> Last verified against source: 2026-08-12

# `vision_encoders/shared_visual_tokens.py`

## Purpose

**Điểm chiếu duy nhất** của Stage 1. Nhận đầu ra thô của mọi encoder (chiều khác
nhau), chiếu tất cả về `VISUAL_DIM = 1408`, nối theo trục token, và ghi lại
`spans` — token nào thuộc encoder nào.

## Why it exists

Trước khi có file này, MHCAC và Q-Former **mỗi bên tự chiếu đặc trưng**. Hai nhánh
có thể trôi dạt sang hai biểu diễn thị giác khác nhau, và không có gì phát hiện ra.
Giờ chỉ còn một điểm chiếu; `spans` cho phép MHCAC vẫn cấp pos-enc riêng cho từng
encoder mà **không cần chiếu lại**.

## Role in architecture

```text
{biovil [B,P₁,1408], pubmedclip [B,P₂,768], swin [B,P₃,Dₛ]}
        ▼ SharedVisualTokenProjector
SharedVisualTokens(tokens=[B,ΣP,1408], spans={biovil: slice(0,P₁), ...})
        ├─► MHCAC
        └─► Q-Former
```

## Status

```text
✅ ACTIVE
```

## Inputs / Outputs

`forward(streams: dict[str, Tensor]) -> SharedVisualTokens`

## Main classes / functions

| Tên | Dòng | Vai trò |
|---|---|---|
| `SharedVisualTokens` | 33 | Dataclass: `.tokens`, `.spans` + `batch_size`, `num_tokens`, `visual_dim`, `stream()`, `stream_mask()`, `without()` |
| `SharedVisualTokenProjector` | 122 | `nn.Module`; một `Linear` mỗi stream, **`Identity` cho stream đã đúng 1408** |
| `validate_shared_visual_tokens` | 79 | Kiểm bất biến |

`CANONICAL_STREAM_ORDER = ("biovil", "pubmedclip", "swin", "raddino")` (`:29`) —
**thứ tự cố định**, để `spans` ổn định giữa các lần chạy kể cả khi dict vào có thứ
tự khác.

## `without(*names)` — ablation encoder lúc chạy

Zero-out một stream **mà không đổi shape**. Cho phép ablation encoder tại thời
điểm inference thay vì train lại. Hiện chỉ test dùng.

## Calls / Called by

Gọi: `torch.nn`. Được gọi: `Blip2Qformer.__init__:338`,
`_encode_image_streams:542`, nhánh view-consistency `:964`;
`tests/test_shared_visual_tokens.py`.

## Side effects

Không.

## Error / edge cases

`validate_shared_visual_tokens` raise khi shape/dim sai. `stream(name)` với tên
không có trong `spans` → raise (`tests/test_shared_visual_tokens.py:111`).

## Related tests

`tests/test_shared_visual_tokens.py` (209 dòng) — thứ tự chuẩn hóa, spans đúng,
`without()` giữ shape, gradient chảy đúng luồng.

## Developer notes

1. **Thứ tự stream phải giữ chuẩn hóa.** Đổi `CANONICAL_STREAM_ORDER` làm mọi
   checkpoint cũ sai spans → MHCAC cấp pos-enc nhầm encoder, **im lặng**.
2. Stream đã đúng 1408 (biovil) nhận `Identity`, không phải `Linear` — đừng thay
   bằng Linear "cho đồng nhất", sẽ thêm tham số và đổi state dict.

## Source relationships

- **Parent:** [`vision_encoders/_index.md`](_index.md)
- **Related:** [`mhcac_12.py`](../mhcac/mhcac_12.py.doc.md) · [`blip2_qformer.py`](../model/lavis/models/blip2_models/blip2_qformer.py.doc.md)

← [HOME](../../HOME.md)
