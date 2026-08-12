> Source: `stage2/prompts/ontology.py` (54 dòng)
> Status: ✅ ACTIVE
> Last verified against source: 2026-08-12

# `stage2/prompts/ontology.py`

## Purpose
14 tên bệnh lý (No Finding + 13 findings) — **mirror torch-free** của
`ABNORMALITIES_14` trong `fig9`.

## Why it exists
Docstring `:4` ghi rõ: tên được mirror ở đây để giữ `stage2/prompts/` **không phụ
thuộc torch**. `tests/test_stage2_prompts.py` canh hai nơi khớp nhau.

## Status
```text
✅ ACTIVE
```

## Main items
`validate_critical_findings(names)` (`:48`) — kiểm danh sách "finding quan trọng"
(dùng bởi `NegativePolicy.critical`) chỉ chứa tên hợp lệ.

## Calls / Called by
Gọi: stdlib.
Được gọi: `builder.py`, `policies.py`, `validation.py`.

## Side effects
Không.

## Related tests
`tests/test_stage2_prompts.py`

## Developer notes
⚠ **Ba nơi phải khớp nhau:** file này, `fig9:160 ABNORMALITIES_14`, và
`blip2_qformer.py:43 chexpert_cols`. Lệch một tên là prompt mô tả sai bệnh lý.

## Source relationships

- **Parent:** [`_index.md`](_index.md)

← [HOME](../../../HOME.md)
