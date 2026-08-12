> Source: `model/lavis/models/blip2_models/Qformer.py:809-970`
> Status: ✅ ACTIVE

# `BertModel.forward(...)`

## Purpose

Điểm hợp nhất query token, text token và encoder visual states trong Q-Former.

## Flow

Validate `input_ids`/`inputs_embeds` → build embeddings và attention masks → nối
query attention mask → `BertEncoder` với cross-attention states/mask → tách
sequence/pool output. `query_length` quyết định block nào dùng query-specific FFN.

## Inputs / outputs

Text `input_ids [B,T]` có thể vắng khi chỉ chạy query; `query_embeds [B,Q,768]`;
visual `encoder_hidden_states [B,P,1408]` theo Meta-CXR config. Output
`BaseModelOutputWithPoolingAndCrossAttentions`; exact `P` dynamic theo encoder.

## Called by / risk

`Blip2Qformer.forward`, ITM, feature extraction. Sai attention mask có thể cho
text nhìn padding hoặc query không cross-attend ảnh mà loss vẫn hữu hạn.

← [`Qformer.py`](../../Qformer.py.doc.md) · [HOME](../../../../../../../HOME.md)
