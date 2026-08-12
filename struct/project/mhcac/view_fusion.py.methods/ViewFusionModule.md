> Source: `mhcac/view_fusion.py:78-153`
> Status: ✅ ACTIVE

# `class ViewFusionModule(nn.Module)`

## Located in

[`view_fusion.py`](../view_fusion.py.doc.md)

## Purpose
Wrapper quanh N khối `ViewFusionBlock`, lo view embedding, mask, gate, và view-drop.

## ★ Một module cho MỖI encoder
Docstring `:83`: mỗi encoder có `dim` riêng (biovil 1408, pubmedclip 768,
swin/raddino tùy model), nên phải có module riêng.

`Blip2Qformer.__init__:315` dựng `nn.ModuleDict` từ `stream_dims`.

## Constructor (`:87`)
```python
ViewFusionModule(dim, num_heads=8, ffn_ratio=4, num_view_types=4,
                 num_blocks=1, dropout=0.1, p_view_drop=0.2)
```
⚠ Prod đặt `p_view_drop: 0.15` (không phải default 0.2).

`num_view_types=4` → embedding cho PA / AP / LATERAL / UNKNOWN.

## `forward(anchor, aux, aux_mask=None, anchor_view_id=None, aux_view_ids=None)` (`:98`)

**Hợp đồng shape:** `[B,P,D]` vào → `[B,P,D]` ra. Bất khả xâm phạm — MHCAC và
Q-Former dựa vào nó.

## Execution flow
```text
view embedding cho anchor và aux (num_view_types)
   ↓
p_view_drop: ngẫu nhiên bỏ aux view (CHỈ lúc train)     ← regularization
   ↓
aux [B,N,P,D] → flatten → [B, N*P, D]
attn_bias [B,1,1,N*P] từ aux_mask, dùng MASK_NEG
gate [B,1,1] = 0 cho study không có aux                 ← không loại khỏi batch
   ↓
num_blocks × ViewFusionBlock(anchor, view_emb, aux_tokens, attn_bias, gate)
```

## ★ Gate thay vì loại khỏi batch
Study không có aux được nhân 0. Batch giữ nguyên shape → không có nhánh điều khiển
phụ thuộc dữ liệu → DDP không lệch giữa các rank.

## ★ `p_view_drop` — vì sao cần
Nếu model luôn có aux view lúc train, nó sẽ **phụ thuộc** vào aux. Ngẫu nhiên bỏ
aux buộc anchor tự đứng vững — quan trọng vì nhiều study thật chỉ có một view.

## Config dependencies
`model.view_fusion.{num_heads, ffn_ratio, num_blocks, num_view_types, dropout, p_view_drop}`
— **6 key này được `from_config:1370-1377` đọc tường minh**.

## Called by
`Blip2Qformer._fuse:465` — một lần cho mỗi encoder bật.

## Tests
`tests/test_view_fusion.py`

## Modification risk
Thêm key config mới mà quên thêm dòng đọc ở `from_config` → key không có hiệu lực,
không cảnh báo.
