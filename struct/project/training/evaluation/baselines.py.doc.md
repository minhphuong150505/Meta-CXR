> Source: `training/evaluation/baselines.py` (204 dòng)
> Status: ✅ ACTIVE
> Last verified against source: 2026-08-12

# `baselines.py`

## Purpose
Baseline để so sánh: đặc biệt là **all-negative**.

## Why it exists
Với dữ liệu mất cân bằng nặng, một model đoán "negative" cho mọi thứ vẫn đạt
accuracy cao. Không có baseline này, một kết quả tầm thường trông như thành công.

## Status
```text
✅ ACTIVE
```

## Main items
| Tên | Dòng | Vai trò |
|---|---|---|
| `compute_baselines(...)` | 80 | ★ Tính các baseline |
| `baseline_table(...)` | 180 | Bảng so sánh |
| `BaselineRow` | 44 | Một dòng kết quả |
| `_row(name, report, description)` | 67 | Helper |

## Calls / Called by
Gọi: `evaluation.classification_metrics`, `schemas`, `uncertain_policy`.
Được gọi: `scripts/evaluate_stage1.py:32,208`; `tests/test_threshold_calibration.py:18`.

## Side effects
Không.

## Related tests
`tests/test_threshold_calibration.py`

## Developer notes
**Luôn báo cáo baseline cùng kết quả model.** Một con số F1 không có baseline
không nói lên điều gì về dữ liệu mất cân bằng.

## Source relationships

- **Parent:** [`training/evaluation/`](_index.md)
- **Related:** [`schemas.py`](schemas.py.doc.md)

← [HOME](../../../HOME.md)
