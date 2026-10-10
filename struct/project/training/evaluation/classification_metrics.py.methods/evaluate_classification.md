# `evaluate_classification(predictions, *, zero_division=1.0, thresholds=None, cutpoints=None)`

> File: [`classification_metrics.py`](../classification_metrics.py.doc.md) · dòng 262

Chấm một split theo ba lớp (Negative / Positive / Uncertain).

1. Từ chối nếu số lớp khác 3 (`SchemaError`); `thresholds` và `cutpoints` cùng
   lúc → `ValueError`.
2. Luật quyết định:
   - mặc định `decide` = argmax trên xác suất ba lớp (giao thức của bài báo);
   - `thresholds` (Eq. 22 theo bệnh × lớp, fit trên val) →
     `decide_with_thresholds` — phân tích bổ sung (2026-10-01);
   - `cutpoints` (`{bệnh: (t1, t2)}` trên `p_pos/(p_pos+p_neg)`, fit trên val) →
     `apply_cutpoints` — **luật cho số chính của dự án từ 2026-10-10** (quyết định
     của user), vẫn không phải giao thức của bài báo.
3. Với từng bệnh: bỏ ô `MISSING` (-1, study không có thông tin CheXpert);
   tính P/R/F1 `weighted` + accuracy (`weighted_prf`), `macro_recall`, P/R/F1 từng lớp,
   ma trận nhầm lẫn 3×3, AUROC và AUPRC một-lớp-với-phần-còn-lại cho cả ba lớp
   (AUROC/AUPRC không phụ thuộc luật quyết định).
4. Aggregates: trung bình đều trên các bệnh (nanmean), kể cả `macro_recall`
   (metric chọn checkpoint Stage 1, D-024), trung bình 5 bệnh, trung bình AUROC
   theo từng lớp.

Trả `ClassificationReport(per_pathology, aggregates, settings)`;
`settings["decision"]` ghi luật đã dùng. Không còn `uncertain_policy`,
`include_meta_labels` (D-023).
