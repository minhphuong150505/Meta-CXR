> Source: `training/train_eval_figure9_llm_variants_200.py:1869-1916` (thân: `_generate_texts`, `:1962-2095`)
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

## Stop tokens (2026-09-09)
Giữ `model.generation_config.eos_token_id`, kể cả danh sách nhiều ID; chỉ dùng
tokenizer EOS khi model không cấu hình. MedGemma dùng `[1, 106]`, và target chat
kết thúc bằng 106. Chỉ truyền tokenizer EOS 1 sẽ bỏ mất ranh giới lượt trả lời.
Regression: `tests/test_generation_stop_tokens.py` kiểm tra list/scalar/fallback
với chuỗi sinh tổng hợp, không tải model hay dữ liệu.

## Side effects
Chạy generate trên GPU.

## Called by
`evaluate_variant` (`:1487`) — cho test cohort, **đúng một lần**.

## Tests
`tests/test_stage2_prompts.py:42` (soft token bad words)

## Modification risk
Bỏ `bad_words_ids` → ký hiệu soft token lọt vào báo cáo output.
Dùng đường prompt khác lúc generate → mất parity với train, chất lượng tụt âm thầm.

## Prompt style (2026-10-05, D-027)

`prompt_style` phải bằng `self.legacy_prompt_style` khi không có
`prompt_config` (kiểm tra trong `_chat_texts`). `"paper"` sinh ra prompt
nguyên văn của bài báo META-CXR — xem `build_instruction` trong
[file doc](../../train_eval_figure9_llm_variants_200.py.doc.md).

## Tách thân hàm (2026-10-14)
Thân hàm nằm ở `_generate_texts(..., sampling=None) -> list[str]`, dùng chung với
[`generate_candidates`](generate_candidates.md). `generate` gọi nó với đúng các tham số
cũ và trả `[0]`; khi `sampling is None` kwargs gửi cho `model.generate` giống từng khoá
với trước (`tests/test_beam_generation.py::test_greedy_sends_no_sampling_kwargs`). Số hàng
soft/finding token = `num_beams` khi beam, `num_return_sequences` khi greedy/sampling —
đúng cách HF 4.53 nhân batch.

