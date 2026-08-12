> Source: `scripts/evaluate_stage1.py` (328 dòng)
> Status: ✅ ACTIVE
> Last verified against source: 2026-08-12

# `scripts/evaluate_stage1.py`

## Purpose
Chấm điểm phân loại Stage 1 từ `.npz` — **không model, không GPU, không dataset**.

## Why it exists
Docstring `:4`: mọi thứ tính lại từ `.npz` ghi lúc inference. *"re-running an
evaluation with a different uncertain policy or a new threshold file must not cost
a GPU-hour."*

## Entry point
```bash
python scripts/evaluate_stage1.py --predictions <test.npz> \
    --thresholds <thresholds.json> --output-dir <dir> [--plots]
```

## Main functions
`parse_args(argv)` (`:66`) · `main(argv) -> int` (`:102`) · `_write_plots(...)` (`:293`)

## Calls / Called by
Gọi: `evaluation.baselines` (`:32`), `.bootstrap` (`:33`),
`.classification_metrics` (`:39`), `.report_writer` (`:40`,`:189`),
`.schemas` (`:46`), `.subgroup_analysis` (`:47`), `.threshold_calibration` (`:53`),
`.uncertain_policy` (`:57`), và **`.visualization` import trễ** (`:294`).

## Side effects
Ghi markdown + JSON + CSV (+ PNG nếu `--plots`).

## Error / edge cases
Threshold calibrate trên test → từ chối · Thiếu matplotlib → bỏ plot, giữ metric

## Related tests
`tests/test_evaluation_integration.py`, `tests/test_classification_metrics.py`

## Developer notes
Import `visualization` **trễ ở `:294`** để script chạy được trên máy chỉ có numpy.
Đừng nâng lên đầu file.

← [`_index.md`](_index.md) · [HOME](../../HOME.md)
