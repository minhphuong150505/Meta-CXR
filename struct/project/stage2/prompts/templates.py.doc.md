> Source: `stage2/prompts/templates.py` (116 dòng)
> Status: ✅ ACTIVE
> Last verified against source: 2026-08-12

# `stage2/prompts/templates.py`

## Purpose
Chuỗi template và **hash của chúng**.

## Why it exists
`template_hash()` (`:114`) khiến một thay đổi template nhỏ nhất cũng đổi hash ghi
vào metadata. Không có nó, hai kết quả sinh bằng hai template khác nhau trông
giống hệt nhau trong file kết quả.

## Status
```text
✅ ACTIVE
```

## Main items
| Hàm | Dòng |
|---|---|
| `sentence_constraint(min_sentences, max_sentences)` | 100 |
| `join_or_none(names)` | 110 |
| `template_hash(visual_mode, length=16)` | 114 |

## Calls / Called by
Gọi: `hashlib`, `schemas.VisualMode`.
Được gọi: `builder.py`; `fig9:151` (`template_hash as _prompt_template_hash`).

## Side effects
Không.

## Related tests
`tests/test_stage2_prompts.py`

## Developer notes
**Đổi template = đổi hash = kết quả cũ không so sánh trực tiếp được với mới.** Đó
là chủ ý. Ghi lại hash trong mọi báo cáo.

## Source relationships

- **Parent:** [`_index.md`](_index.md)

← [HOME](../../../HOME.md)
