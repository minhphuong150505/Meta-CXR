> Source: `scripts/evaluate_stage2.py` (294 dòng)
> Status: ✅ ACTIVE
> Last verified against source: 2026-08-12

# `scripts/evaluate_stage2.py`

## Purpose
Chấm điểm báo cáo sinh ra, từ `.jsonl`.

## Entry point
```bash
python scripts/evaluate_stage2.py --predictions <reports.jsonl> \
    --metrics bleu,rouge,meteor,cider,bertscore \
    --skip-clinical-metrics --output-dir <dir>
```

## Main functions
`parse_args(argv)` (`:68`) · `main(argv) -> int` (`:110`) · `comma_list(value)` (`:64`)

## Calls / Called by
Gọi: `evaluation.bootstrap` (`:34`), `.error_analysis` (`:40`),
`.generation_metrics` (`:41`), `.report_writer` (`:46`), `.schemas` (`:53`),
`.subgroup_analysis` (`:54`), và **`.clinical` import trễ** (`:159`).

## ⚠ Chỉ số lâm sàng
`:172` có comment ghi rõ adapter chưa wire, và kết quả được **báo là gap** chứ
không phải điểm 0.

## Side effects
Ghi markdown + JSON.

## Error / edge cases
Thiếu package METEOR/CIDEr/BERTScore → `MissingMetricDependency` nêu tên package ·
Clinical metric → `unavailable`, **không phải 0**

## Related tests
`tests/test_generation_metrics.py`, `tests/test_clinical_metrics.py`

## Developer notes
⚠ **BLEU/ROUGE không phải độ chính xác lâm sàng.** Đừng trình bày chúng như vậy
trong báo cáo.

← [`_index.md`](_index.md) · [HOME](../../HOME.md)
