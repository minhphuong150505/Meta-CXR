> Source: `model/lavis/models/blip2_models/blip2_qformer.py:1035-1117`
> Status: 🟡 CONDITIONAL — BLIP caption generation

# `Blip2Qformer.generate(samples, ...)`

## Purpose

Encode ảnh/aux/cache qua cùng `_encode_image_streams`, chạy query Q-Former rồi
generate text bằng Q-Former LM head với beam hoặc nucleus sampling.

## Data flow

`image_embeds [B,P,1408]` → repeat theo beam → query tokens `[B,32,768]` →
`Qformer.generate` → tokenizer decode. `P` dynamic; batch được repeat chỉ cho beam.

## Status / callers

Không phải report generation Stage 2 MedGemma. Giữ cho LAVIS caption API và các
workflow gọi `model.generate`; Stage-1 production chủ yếu dùng `forward`/`forward_image`.

← [`Blip2Qformer`](_index.md) · [`blip2_qformer.py`](../../blip2_qformer.py.doc.md)
