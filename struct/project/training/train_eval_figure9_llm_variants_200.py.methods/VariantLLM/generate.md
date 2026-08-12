> Source: `training/train_eval_figure9_llm_variants_200.py:1307-1372`
> Status: ✅ ACTIVE

# `VariantLLM.generate(record, prompt_style, max_new_tokens)`

## Located in

[`training/train_eval_figure9_llm_variants_200.py`](../../train_eval_figure9_llm_variants_200.py.doc.md)

## Purpose
Sinh báo cáo cho **một** record.

## Execution flow
```text
_chat_texts(record, prompt_style)  hoặc  _native_chat_inputs(...)
   ↓ (mode Q-Former) chèn <qformer_soft_token> vào prompt
processor(text, images) → inputs
   ↓
model.generate(..., bad_words_ids=soft_token_bad_words_ids(img_token_id))   ★
   ↓
decode → clean_text()
```

## ★ `bad_words_ids` — chặn model tự sinh soft token
Không có nó, model có thể phát ra ký hiệu `<qformer_soft_token>` **vào giữa báo
cáo**. Nó sẽ được decode thành text rác trong output cuối.

## ★ Cùng `PromptBuilder` với train
`_render_prompt_text` (`:914`) gọi `PromptBuilder(...).build(context).user_text(img_token)`
— **cùng đường** với lúc train. Đây là thứ khiến parity kiểm tra được byte-for-byte.

## Parameters
| Tham số | Ghi chú |
|---|---|
| `record` | dict — cùng hình dạng lúc train |
| `prompt_style` | `"fine"` … |
| `max_new_tokens` | |

## Returns
`str` — báo cáo đã dọn.

## Side effects
Chạy generate trên GPU.

## Called by
`evaluate_variant` (`:1487`) — cho test cohort, **đúng một lần**.

## Tests
`tests/test_stage2_prompts.py:42` (soft token bad words)

## Modification risk
Bỏ `bad_words_ids` → ký hiệu soft token lọt vào báo cáo output.
Dùng đường prompt khác lúc generate → mất parity với train, chất lượng tụt âm thầm.
