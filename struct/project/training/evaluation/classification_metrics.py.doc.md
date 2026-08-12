> Source: `training/evaluation/classification_metrics.py` (662 dòng)
> Status: ✅ ACTIVE
> Last verified against source: 2026-08-12

# `classification_metrics.py`

## Purpose

Toàn bộ chỉ số phân loại Stage 1: precision/recall/F1, AUROC, AUPRC, ma trận nhầm
lẫn 3 lớp, và các phép gộp macro/micro/weighted.

**Chỉ cần numpy** — chạy được ở bất cứ đâu test chạy được.

## Status

```text
✅ ACTIVE — file lớn nhất trong evaluation/ (662 dòng)
```

## Main items

| Tên | Dòng | Vai trò |
|---|---|---|
| `evaluate_classification(...)` | 296 | [📄](classification_metrics.py.methods/evaluate_classification.md) ★ Điểm vào chính |
| `PathologyMetrics` | 173 | Chỉ số một bệnh lý |
| `ClassificationReport` | 235 | Kết quả tổng |
| `roc_auc(scores, y_true)` | 79 | ★ Tự implement |
| `average_precision(scores, y_true)` | 110 | ★ AUPRC |
| `binary_confusion(...)` | 66 | TP/FP/FN/TN |
| `three_class_confusion_matrices(...)` | 264 | 14 ma trận 3×3 |
| `apply_thresholds(...)` | 282 | Áp threshold đã calibrate |
| `_three_class_aggregates(...)` | 554 | Gộp 3 lớp |
| `_micro_probability_metrics(...)` | 640 | Micro theo xác suất |
| `_safe_divide`, `_nanmean`, `_harmonic`, `_weighted_by`, `_positive_f1` | — | Helper số học |

`_safe_divide` (`:54`) và `_nanmean` (`:256`) tồn tại vì bệnh lý hiếm có thể cho
mẫu số 0 — trả `nan` có kiểm soát thay vì crash hoặc trả 0 gây hiểu nhầm.

## Calls / Called by

Gọi: `numpy`, `evaluation.schemas`, `evaluation.uncertain_policy`.
Được gọi: `scripts/evaluate_stage1.py:39`; `baselines.py:25`;
`model/lavis/tasks/image_text_pretrain.py:223` (**import trễ, trong hàm**);
`tests/test_classification_metrics.py`, `test_threshold_calibration.py:27`.

## Side effects

Không. Hàm thuần.

## Error / edge cases

Bệnh lý không có mẫu positive → AUPRC/AUROC là `nan`, **không phải 0**. Phân biệt
này quan trọng: 0 nghĩa là "dự đoán sai hoàn toàn", `nan` nghĩa là "không tính được".

## Related tests

`tests/test_classification_metrics.py` (377 dòng)

## Developer notes

1. **`nan` ≠ 0.** Đừng thay `nan` bằng 0 khi tổng hợp — sẽ kéo macro xuống một
   cách sai lệch.
2. `selection_metric: macro_auprc` của Stage 1 đến từ file này.

## Source relationships

- **Parent:** [`training/evaluation/`](_index.md)
- **Related:** [`schemas.py`](schemas.py.doc.md) · [`scripts/_index.md`](../../scripts/_index.md)

← [HOME](../../../HOME.md)
