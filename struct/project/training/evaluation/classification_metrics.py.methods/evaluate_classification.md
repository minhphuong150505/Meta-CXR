# `evaluate_classification(predictions, *, zero_division=1.0)`

> File: [`classification_metrics.py`](../classification_metrics.py.doc.md) · dòng 234

Chấm một split theo giao thức ba lớp của bài báo.

1. Từ chối nếu số lớp khác 3 (`SchemaError`).
2. `decide` = argmax trên xác suất ba lớp.
3. Với từng bệnh: bỏ ô `MISSING` (-1, study không có thông tin CheXpert);
   tính P/R/F1 `weighted` + accuracy (`weighted_prf`), P/R/F1 từng lớp,
   ma trận nhầm lẫn 3×3, AUROC và AUPRC một-lớp-với-phần-còn-lại cho cả ba lớp.
4. Aggregates: trung bình đều trên các bệnh (nanmean), trung bình 5 bệnh, trung
   bình AUROC theo từng lớp.

Trả `ClassificationReport(per_pathology, aggregates, settings)`. Không còn tham số
`thresholds`, `uncertain_policy`, `include_meta_labels` (D-023).
