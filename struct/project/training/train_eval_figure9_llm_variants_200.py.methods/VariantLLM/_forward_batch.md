> Source: `training/train_eval_figure9_llm_variants_200.py:1082-1107`
> Status: ✅ ACTIVE

# `VariantLLM._forward_batch(batch)`

## Located in

[`training/train_eval_figure9_llm_variants_200.py`](../../train_eval_figure9_llm_variants_200.py.doc.md)

## Purpose
Một forward: chuyển batch lên device, (mode Q-Former) tiêm soft token, gọi model.

## Execution flow
```text
batch → device
   ↓
IF mode Q-Former:
   img_proj(qformer_embeds) → [B,32,hidden]
   cài SoftTokenEmbeddingWrapper tạm thời
      └─ wrapper.forward gọi validate_soft_token_batch() cho từng row
         rồi THAY THẾ tại vị trí <qformer_soft_token>
   ↓
model(**batch) → outputs.loss
   ↓ finally
khôi phục embedding gốc
```

## ★ THAY THẾ, không cộng
Vector Q-Former **thay** embedding tại vị trí soft token. Cộng vào sẽ trộn hai tín
hiệu và không mode nào hoạt động đúng.

## Returns
`outputs` — có `.loss`.

## Side effects
GPU compute.

## Error handling
`SoftTokenEmbeddingWrapper.forward` gọi `validate_soft_token_batch`; nó raise khi
số vị trí soft token ≠ `num_img_tokens` hoặc batch size lệch. Khối `finally` vẫn
khôi phục embedding gốc nếu model forward raise.

## Called by
`train_fine`, `evaluate_loss`

## Tests
`tests/test_soft_token_injection.py`

## Modification risk
⚠ Đây là điểm hội tụ của lỗi im lặng nguy hiểm nhất repo. Đừng nới lỏng validate.
