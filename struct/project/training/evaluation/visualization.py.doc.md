> Source: `training/evaluation/visualization.py` (336 dòng)
> Status: 🟡 CONDITIONAL — cần extra eval-plots
> Last verified against source: 2026-08-12

# `visualization.py`

## Purpose
Vẽ ROC, PR, confusion matrix, biểu đồ per-pathology, so sánh threshold, histogram
xác suất, reliability diagram.

## Why it exists
Docstring `:6` ghi nguyên tắc: *"is never a crash and never an empty axis presented
as a result"* — thiếu matplotlib phải báo rõ, không được trả về một trục rỗng trông
như kết quả.

## Status
```text
🟡 CONDITIONAL — import TRỄ ở `scripts/evaluate_stage1.py:294`
```

## Main items
| Tên | Dòng |
|---|---|
| `plot_roc_curves` | 63 |
| `plot_pr_curves` | 101 |
| `plot_confusion_matrices` | 137 |
| `plot_per_pathology_bars` | 180 |
| `plot_threshold_comparison` | 222 |
| `plot_probability_histogram` | 263 |
| `plot_reliability_diagram` | 298 |
| `_pyplot()` | 27 | ★ Import lazy, raise `PlottingUnavailable` |
| `_roc_points` / `_pr_points` | 40, 52 |
| `PlottingUnavailable` | 23 |

## Calls / Called by
Gọi: `matplotlib` (lazy qua `_pyplot()`), `numpy`.
Được gọi: `scripts/evaluate_stage1.py:294` (trong `_write_plots`, có `--plots`).

## Side effects
Ghi file ảnh.

## Error / edge cases
Thiếu matplotlib → `PlottingUnavailable`, script vẫn hoàn thành phần metric.

## Related tests
Không có test trực tiếp (cần matplotlib).

## Developer notes
`_pyplot()` là lý do `evaluate_stage1.py` chạy được trên máy chỉ có numpy. Đừng
nâng `import matplotlib` lên đầu file.

## Source relationships

- **Parent:** [`training/evaluation/`](_index.md)
- **Related:** [`schemas.py`](schemas.py.doc.md)

← [HOME](../../../HOME.md)
