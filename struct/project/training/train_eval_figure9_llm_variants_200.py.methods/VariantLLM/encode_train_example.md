> Source: `training/train_eval_figure9_llm_variants_200.py:1017-1058`
> Status: ✅ ACTIVE

# `VariantLLM.encode_train_example(record, prompt_style, max_length=768)`

## Located in

[`training/train_eval_figure9_llm_variants_200.py`](../../train_eval_figure9_llm_variants_200.py.doc.md)

## Purpose
Một record → tensor sẵn sàng cho forward, **có mask prompt prefix**.

## ★ Mask prompt khỏi label
Dùng `training.stage2_utils.masked_label_ids` (`:114`): mọi token thuộc prompt
prefix được đặt `-100`.

Không có nó, model học **sinh lại prompt** thay vì học sinh báo cáo — loss vẫn
giảm, nhưng phần lớn tín hiệu học là vô ích.

## Execution flow
```text
_chat_texts(record, prompt_style) → (prompt_text, target_text)
   ↓
processor / tokenizer(prompt + target, max_length, truncation)
   ↓
masked_label_ids(input_ids, prompt_len)  → labels với -100 ở prefix
   ↓
(mode Q-Former) đánh dấu vị trí <qformer_soft_token>
   ↓
return {"input_ids", "attention_mask", "labels", "pixel_values"?, ...}
```

## Parameters
`record` · `prompt_style` · `max_length=768`

## ⚠ `max_length=768` cắt cụt
Prompt v2 + FINDINGS + IMPRESSION có thể vượt. Xem
`scripts/prompt_length_statistics.py` để đo trước khi chọn giá trị này.

## Returns
dict tensor cho một mẫu (chưa batch).

## Side effects
Không.

## Called by
`collate_train` (`:1059`)

## Tests
`tests/test_stage2_prompts.py` (masking)

## Modification risk
Sai `prompt_len` → mask lệch → model học sai phần. Lỗi này **im lặng**.
