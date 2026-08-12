> Source: `stage2/prompts/schemas.py` (143 dòng)
> Status: ✅ ACTIVE — ★
> Last verified against source: 2026-08-12

# `stage2/prompts/schemas.py`

## Purpose
Kiểu dữ liệu của Prompt v2, quan trọng nhất là **5 visual mode**.

## ★ Năm visual mode và ranh giới label
`VisualMode` (`:13`):

| Mode | Thấy Stage-1 labels? |
|---|---|
| `native_anchor_only` | ❌ |
| `native_anchor_guided` | ✅ |
| `native_multiview` | ❌ ⚠ chưa hoàn chỉnh end-to-end |
| `qformer_visual_only` | ❌ ← **giữ ablation không bị nhiễm** |
| `qformer_guided` | ✅ |

Docstring `:16` ghi việc nêu tường minh các mode này **sửa một `uses_mhcac_prompt`
từng là code chết**.

> `qformer_visual_only` **không nhận label nào**. Nếu nó nhận, so sánh giữa đường
> Q-Former thuần thị giác và đường có label mất hết ý nghĩa.

## Status
```text
✅ ACTIVE
```

## Main items
| Tên | Dòng |
|---|---|
| `VisualMode` (Enum) | 13 |
| `PromptContext` | 59 |
| `PartKind` (Enum) | 96 |
| `PromptPart` | 103 |
| `RenderedPrompt` | 122 |

## Calls / Called by
Gọi: `enum`, `dataclasses`.
Được gọi: `builder.py`, `policies.py`, `templates.py`, `records.py`, `validation.py`;
`tests/test_stage2_prompts.py:38`.

## Side effects
Không.

## Related tests
`tests/test_stage2_prompts.py`

## Developer notes
Thêm visual mode mới **phải** quyết định rõ nó có thấy label không, và ghi vào bảng
trên. Mơ hồ ở đây làm hỏng ablation.

## Source relationships

- **Parent:** [`_index.md`](_index.md)

← [HOME](../../../HOME.md)
