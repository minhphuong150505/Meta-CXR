> Source: `model/lavis/models/blip2_models/Qformer.py:992-1100`
> Status: ✅ ACTIVE

# `BertLMHeadModel.forward(...)`

## Purpose

Bọc `BertModel`, áp MLM/LM decoder và tính cross-entropy labels; hỗ trợ
`past_key_values` để query prefix chỉ tính một lần trong nhánh LM.

## Important behavior

Khi có `labels`, logits được shift cho autoregressive loss và `reduction` được
truyền theo caller. `prepare_inputs_for_generation` nối dummy token/mask cho bước
generate tiếp theo; `_reorder_cache` phục vụ beam search.

## Called by

`Blip2Qformer._language_modeling`, `generate` và feature paths.

← [`Qformer.py`](../../Qformer.py.doc.md) · [HOME](../../../../../../../HOME.md)
