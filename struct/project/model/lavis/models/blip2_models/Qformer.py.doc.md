> Source: `model/lavis/models/blip2_models/Qformer.py` (1.221 dòng)
> Status: ✅ ACTIVE
> Last verified against source: 2026-08-12

# `Qformer.py`

## Purpose

Q-Former của BLIP-2: một BERT được sửa để nhận **query token** học được và
cross-attend vào token thị giác.

## Why it exists

Cầu nối giữa biểu diễn thị giác (hàng trăm token) và không gian ngôn ngữ (32 token).
Nó nén thông tin và học căn chỉnh ảnh↔text qua ITC/ITM/LM.

## Role in architecture

```text
SharedVisualTokens [B,ΣP,1408]
        │  encoder_hidden_states
        ▼
query_tokens [B,32,768] ──► Qformer.bert ──► last_hidden_state [B,32,768]
                                                    ├─► vision_proj → ITC
                                                    ├─► itm_head → ITM
                                                    └─► past_key_values → LM
```

## Status

```text
✅ ACTIVE — phần lớn giữ nguyên upstream LAVIS
```

## Used in

Training ✅ · Validation ✅ · Inference ✅ (Stage 2 mode Q-Former)

## Entry point

Không.

## Inputs

`query_embeds`, `input_ids`, `attention_mask`, `encoder_hidden_states`,
`encoder_attention_mask`, `past_key_values`, `labels`, `use_cache`, `reduction`

## Outputs

`last_hidden_state`, `past_key_values`, và `loss` khi có `labels`.

## Important configuration

| Key | Giá trị prod | Ảnh hưởng |
|---|---|---|
| `num_query_token` | 32 | Số query token |
| `cross_attention_freq` | 2 | Cross-attention mỗi 2 block |
| `max_txt_len` | 256 | Độ dài text tối đa |

Khởi tạo qua `Blip2Base.init_Qformer(num_query_token, vis_num_feat, cross_attention_freq)`.

## Main classes

Kiến trúc BERT chuẩn đã sửa: embeddings, self/cross attention, layer, encoder,
pooler, LM head, và `BertLMHeadModel`. ⚠ Phần lớn là code upstream Salesforce —
tài liệu chi tiết từng lớp không nằm trong phạm vi
[D-007](../../../../_meta/DECISIONS.md#d-007--độ-sâu-documentation-cho-fork-lavis).

Hai đường `forward` quan trọng đã được tách riêng: [`BertModel.forward`](Qformer.py.methods/BertModel/forward.md)
và [`BertLMHeadModel.forward`](Qformer.py.methods/BertLMHeadModel/forward.md).

## Ba chi tiết Meta-CXR cần biết

### 1. Query token có trọng số riêng (`_query`)

`blip2_qformer.py:204-207` copy trọng số từ tên gốc sang tên có hậu tố `_query`:

```python
for name, param in self.Qformer.named_parameters():
    if "_query" in name:
        param.data.copy_(state_dict[name.replace("_query", "")])
```

Nghĩa là nhánh xử lý query khởi tạo từ trọng số text, rồi tách ra học riêng.

### 2. Vá bias sau resize vocab

```python
self.Qformer.cls.predictions.bias = self.Qformer.cls.predictions.decoder.bias  # :202
```

`transformers ≥ 4.50` thay decoder khi `resize_token_embeddings` nhưng để lại
bias alias ở vocab size cũ. Không vá thì checkpoint BLIP-2 30.523 token không load
được (shape mismatch).

### 3. `use_cache=True` cho LM

`forward` gọi `Qformer.bert(..., use_cache=True)` rồi truyền `past_key_values`
sang `_language_modeling`. Query token được tính **một lần**, dùng lại cho LM.

## Calls

`transformers` BERT primitives, `torch.nn`.

## Called by

`Blip2Qformer.__init__` (`:192` qua `init_Qformer`) · `Blip2Qformer.forward`
(`:861` query, `:879` text, `:772` ITM, `:807` LM) ·
`Blip2Qformer.initialize_expert_tokens` (`:557`)

## Data flow

Xem [DATA_FLOW.md §2.4](../../../../_meta/DATA_FLOW.md#24-trong-model).

## Side effects

Không có ngoài cấp phát tham số.

## Error / edge cases

⚠ Shape mismatch khi load checkpoint nếu vá bias ở `:202` bị gỡ.

## Related tests

`tests/test_blip2_negative_sampling.py` ⚠ cần torchvision · `tests/test_stage1_objectives.py` (gián tiếp)

## Related documentation

[ARCHITECTURE.md §2.5](../../../../_meta/ARCHITECTURE.md#25-q-former--căn-chỉnh-ảnh--text) ·
[`blip2_qformer.py`](blip2_qformer.py.doc.md)

## Developer notes

1. **Đừng reformat file này** — bị loại khỏi ruff để diff upstream còn đọc được.
2. Vá `:202` nằm ở `blip2_qformer.py`, không ở đây. Sửa Q-Former mà quên nó → load
   checkpoint hỏng.
3. Số query token phải khớp `num_img_tokens` của soft token Stage 2 (mặc định 32).

## Source relationships

- **Parent:** [`model/lavis/_index.md`](../../_index.md)
- **Related:** [`blip2_qformer.py`](blip2_qformer.py.doc.md)

← [HOME](../../../../../HOME.md)
