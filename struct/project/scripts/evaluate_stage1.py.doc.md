> Source: `scripts/evaluate_stage1.py` (328 dòng)
> Status: ✅ ACTIVE
> Last verified against source: 2026-08-20

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

## `--label-framing` / `--score` (2026-08-20)

Hai flag mới, chuyển tiếp thẳng vào
[`label_framing.apply_framing`](../training/evaluation/label_framing.py.doc.md)
ngay sau khi load, **trước khi** đọc file ngưỡng.

| flag | mặc định | ý nghĩa |
|---|---|---|
| `--label-framing` | `masked_polarity` | `study_presence` mới là framing để trích dẫn F1 |
| `--score` | `conditional_positive` | `marginal_presence` = `mention × q_pos`, cần gate trong `.npz` |

⚠ **`_threshold_framing_mismatch()` từ chối chạy** khi file ngưỡng được calibrate
dưới framing/score khác. Hai framing có tập nhãn khác nhau, nên một ngưỡng fit ở
bên này **vô nghĩa** ở bên kia — và nếu không chặn thì script chỉ in ra số khác,
không báo lỗi, đúng cái bẫy mà `--uncertain-policy` đã có. File ngưỡng cũ (không
có field này) chỉ được cho qua khi request đúng mặc định lịch sử.

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
