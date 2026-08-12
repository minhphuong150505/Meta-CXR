> Source: `stage2/prompts/builder.py` (259 dòng)
> Status: ✅ ACTIVE — ★
> Last verified against source: 2026-08-12

# `stage2/prompts/builder.py`

## Purpose
`PromptBuilder` — **điểm vào prompt duy nhất** cho cả train và inference Stage 2.

## Why it exists
Docstring `:1` ghi rõ: một builder cho cả hai phía, và nó **không đụng model,
tokenizer hay torch**. Nhờ vậy parity train↔inference kiểm tra được **byte-for-byte**.

Prompt lệch giữa train và inference là lỗi im lặng: model học một định dạng rồi
được hỏi bằng định dạng khác, chất lượng tụt mà không có lỗi nào.

## Status
```text
✅ ACTIVE — nhưng OPT-IN qua --prompt-config
```

## Main items
| Tên | Dòng | Vai trò |
|---|---|---|
| `PromptBuilder` | 30 | ★ [`.build(context)`](builder.py.methods/PromptBuilder/build.md), `.build_user_messages(...)` |
| `fit_to_budget(...)` | 230 | ★ [Cắt theo ngân sách do `count_fn` định nghĩa](builder.py.methods/fit_to_budget.md) |

## Đầu ra mang vân tay
Mỗi prompt kèm `prompt_version`, `config_hash`, `template_hash` — ghi vào metadata
artifact. Sau này truy được một kết quả sinh bằng prompt nào.

## Calls / Called by
Gọi: `schemas`, `policies`, `templates`, `ontology`, `validation`. **Chỉ stdlib.**
Được gọi: `fig9:147,928`; `scripts/run_prompt_ablation.py:31`,
`export_stage2_prompt_samples.py:29`, `prompt_length_statistics.py:24`;
`tests/test_stage2_prompts.py:23`.

## Side effects
Không.

## Related tests
`tests/test_stage2_prompts.py` (408 dòng) — parity, chính sách, masking

## Developer notes
1. **Giữ module torch-free.** Thêm import torch phá test CPU và phá parity testing.
2. `fit_to_budget` cắt phần nào trước là quyết định có ảnh hưởng chất lượng — đọc
   code trước khi đổi.

## Source relationships

- **Parent:** [`_index.md`](_index.md)

← [HOME](../../../HOME.md)
