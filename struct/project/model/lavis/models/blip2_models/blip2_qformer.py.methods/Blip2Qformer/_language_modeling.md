> Source: `model/lavis/models/blip2_models/blip2_qformer.py:823-848`
> Status: ✅ ACTIVE

# `Blip2Qformer._language_modeling(...)`

## Located in

[`blip2_qformer.py`](../../blip2_qformer.py.doc.md)

## Purpose
LM loss — sinh lại FINDINGS từ query token.

## Signature
```python
def _language_modeling(self, text_tokens, query_tokens, query_output, valid_mask) -> Tensor
```

## Execution flow
```text
valid_mask rỗng → zero
   ↓
decoder_input_ids = text_tokens.input_ids.clone()
decoder_input_ids[:, 0] = bos_token_id                    ← thay CLS bằng BOS
labels = masked_fill(input_ids == pad_token_id, -100)     ← bỏ padding
labels[~valid_mask] = -100                                ← ★ bỏ cả hàng không hợp lệ
   ↓
Qformer(decoder_input_ids,
        attention_mask=cat([query_atts, text_attention_mask]),
        past_key_values=query_output.past_key_values,     ← ★ TÁI DÙNG
        labels=labels, reduction="none")
   ↓
token_count = (labels[:, 1:] != -100).sum()
token_count == 0 → zero
return output.loss.sum() / token_count                    ← ★ chuẩn hóa theo TOKEN
```

## Detailed logic
**`past_key_values` tái dùng** — query token đã được tính ở `forward:861` với
`use_cache=True`. Không tính lại.

**`reduction="none"` rồi tự chia** — chia cho **số token thật**, không phải số mẫu.
Nếu để `reduction="mean"`, một batch có mẫu dài và mẫu ngắn sẽ đánh trọng số sai.

**Ba tầng masking labels:** padding → `-100`; hàng `~valid_mask` → toàn `-100`;
kiểm `token_count == 0` để không chia cho 0.

## Error handling
`valid_mask` rỗng hoặc `token_count == 0` → trả `zero` nối đồ thị.

## Config dependencies
`loss.lambda_lm`, `max_txt_len`

## Modification risk
Đổi cách chuẩn hóa (token → mẫu) sẽ đổi tỉ lệ giữa `L_lm` và các loss khác, làm
`lambda_lm` hiện tại không còn đúng.
