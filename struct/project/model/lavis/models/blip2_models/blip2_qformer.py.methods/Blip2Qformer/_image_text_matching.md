> Source: `model/lavis/models/blip2_models/blip2_qformer.py:749-821`
> Status: ✅ ACTIVE

# `Blip2Qformer._image_text_matching(...)`

## Located in

[`blip2_qformer.py`](../../blip2_qformer.py.doc.md)

## Purpose
ITM loss với **hard negative mining** — phân loại nhị phân "cặp (ảnh, text) này có
khớp không".

## Signature
```python
def _image_text_matching(self, image_embeds, text_tokens, valid_mask,
                         valid_all, sim_i2t, sim_t2i) -> Tensor
```

## Execution flow
```text
valid_mask rỗng HOẶC valid_all.sum() < 2 → trả zero (cần ≥2 để có negative)
   ↓
image_embeds_all = _gather_with_local_grad(image_embeds)
text_ids_all / text_atts_all = concat_all_gather(...)
positive_indices = rank * B + arange(B)
   ↓ with no_grad:
weights_t2i = _hard_negative_sampling_weights(sim_t2i, valid_all, positive_indices)
weights_i2t = _hard_negative_sampling_weights(sim_i2t, valid_all, positive_indices)
   ↓
negative_image_indices = [multinomial(weights_t2i[i], 1) for i in local_indices]
negative_text_indices  = [multinomial(weights_i2t[i], 1) for i in local_indices]
   ↓ dựng BỘ BA
text_ids     = cat[pos,      pos,       neg_text]
image_inputs = cat[img_pos,  img_neg,   img_pos ]
labels       = cat[ones(n),  zeros(2n)          ]
   ↓
Qformer.bert(text_ids, query_embeds, encoder_hidden_states=image_inputs)
vl_embeddings = last_hidden_state[:, :32]
logits = itm_head(vl_embeddings).mean(dim=1)
return cross_entropy(logits, labels)
```

## Detailed logic
**Bộ ba, không phải cặp:** mỗi mẫu positive sinh ra ba hàng — (ảnh đúng, text đúng),
(ảnh sai, text đúng), (ảnh đúng, text sai). Nhãn `1, 0, 0`. Model phải phát hiện
sai lệch từ **cả hai phía**.

**Hard negative dùng batch hiện tại, không dùng queue** — comment `:710` giải
thích: ITM cần ảnh và token id thật, mà ITC queue (nhẹ) không giữ.

**`.mean(dim=1)` trên 32 query token** — gộp ý kiến của toàn bộ query thành một
quyết định.

## Error handling
`valid_all.sum() < 2` → trả `zero` nối đồ thị (không đủ mẫu để lấy negative).

## Config dependencies
`loss.lambda_itm`

## Tests
`tests/test_blip2_negative_sampling.py` ⚠ cần torchvision

## Modification risk
Đổi cấu trúc bộ ba phải đổi cả `labels` — lệch sẽ train ngược nhãn mà không báo lỗi.
