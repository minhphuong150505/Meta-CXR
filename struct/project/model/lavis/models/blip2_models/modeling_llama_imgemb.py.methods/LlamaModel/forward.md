> Source: `model/lavis/models/blip2_models/modeling_llama_imgemb.py:523-682`
> Status: ✅ ACTIVE — Vicuna demo only

# `LlamaModel.forward(...)`

## Purpose

Decoder forward đã sửa để trộn text embedding và image embedding cho đường
Vicuna demo, rồi chạy các `LlamaDecoderLayer` với cache/attention mask.

## Image behavior

`split_at_img` nhận tensor và chia theo marker ảnh; forward có branch `use_img` /
`img_embeds` để chèn biểu diễn Q-Former. Đây không phải cơ chế
`<qformer_soft_token>` của MedGemma.

## Risk

Code là fork transformers cũ và không có test. Mọi thay đổi mask/cache/image split
cần smoke test demo GPU; shape phụ thuộc prompt và số image embedding.

← [`modeling_llama_imgemb.py`](../../modeling_llama_imgemb.py.doc.md) · [HOME](../../../../../../../HOME.md)
