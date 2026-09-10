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
## Identity correction (2026-09-08)

`TEMPLATE_VERSION=stage2_prompt_v2_cue_states_v1` ghi nhận thay đổi semantics của builder, dù text fragment cũ vẫn giữ. Không dùng metric của `none` cũ để đại diện cho prompt mới.

## 🧪 `FINDING_TOKEN_HEADER` — 2026-09-10

**Cố ý KHÔNG nằm trong `_HASH_FRAGMENTS`.** Đưa vào đó sẽ đổi `template_hash`
của **mọi** mode, kể cả các arm phải chạy đúng thứ đã tạo ra số đã ghi. Thay vào
đó `template_hash(visual_mode, length, finding_token_count=N)` gộp nó vào payload
**chỉ khi** nhánh bật, nên khi tắt hash giống hệt từng byte.
