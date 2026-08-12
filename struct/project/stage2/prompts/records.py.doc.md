> Source: `stage2/prompts/records.py` (68 dòng)
> Status: ✅ ACTIVE
> Last verified against source: 2026-08-12

# `stage2/prompts/records.py`

## Purpose
`context_from_record` — chuyển một record (native hoặc Q-Former) thành `PromptContext`.

## Status
```text
✅ ACTIVE
```

## Main items
| Hàm | Dòng |
|---|---|
| `context_from_record(...)` | 29 |
| `_as_tuple(value)` | 15 |
| `_as_prob_map(value)` | 23 |

Hai helper `_as_*` chuẩn hóa dữ liệu vào — record đến từ ba nguồn khác nhau
(manifest native, Stage-1 loader, fixture tổng hợp) với kiểu hơi khác nhau.

## Calls / Called by
Gọi: `schemas.PromptContext`. Stdlib.
Được gọi: `builder.py`; `scripts/prompt_length_statistics.py:24`,
`run_prompt_ablation.py`, `export_stage2_prompt_samples.py`.

## Side effects
Không.

## Related tests
`tests/test_stage2_prompts.py`

## Developer notes
Đây là ranh giới giữa "hình dạng dữ liệu" và "hình dạng prompt". Thêm trường mới
vào record phải qua đây mới tới prompt được.

## Source relationships

- **Parent:** [`_index.md`](_index.md)

← [HOME](../../../HOME.md)
