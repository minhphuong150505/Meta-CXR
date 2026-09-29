> Source: `training/evaluation/classification_metrics.py` (347 dòng)
> Status: ✅ ACTIVE
> Last verified against source: 2026-09-29

# `classification_metrics.py`

## Purpose

Chấm Stage 1 **đúng giao thức của bài báo META-CXR**, ba lớp
(Negative / Positive / Uncertain). Viết lại hoàn toàn ngày 2026-09-29
([D-023](../../_meta/DECISIONS.md)): không còn framing nhị phân, không còn
F1 chỉ lớp dương, không còn `uncertain_policy`, không còn loại nhãn "meta".

## Status

```text
✅ ACTIVE — dùng bởi scripts/evaluate_stage1.py, eval hook Stage 1 (mọi epoch được chấm), baselines.py
```

## Main items

| Tên | Dòng | Vai trò |
|---|---|---|
| `evaluate_classification(predictions)` | 234 | [📄](classification_metrics.py.methods/evaluate_classification.md) ★ Điểm vào |
| `weighted_prf(y_true, y_pred)` | 148 | sklearn `average='weighted'`, `zero_division=1` |
| `per_class_prf(matrix)` | 127 | P/R/F1 từng lớp, ngữ nghĩa sklearn |
| `confusion_matrix` | 120 | 3×3, hàng = nhãn thật |
| `decide(probabilities)` | 229 | argmax ba lớp (quyết định của code gốc) |
| `roc_auc`, `average_precision` | 68, 97 | ROC (Mann-Whitney) và AP bậc thang |
| `PAPER_FIVE_FINDINGS` | 56 | 5 bệnh của Bảng 5 và 7 |
| `AGGREGATE_METRICS` | 334 | Tên aggregate hợp lệ (config, selection metric) |

## Aggregates — mapping tới bài báo

| Key | Bài báo |
|---|---|
| `weighted_precision` / `weighted_recall` / `weighted_f1` / `accuracy` | Hình 10, Sec. IV-B-2a (0.87 / 0.78 / 0.73) — trung bình đều 14 bệnh |
| `mean_weighted_f1_5` | Bảng 5, 7 (0.701) |
| `auroc_{negative,positive,uncertain}_mean`, `auroc_mean` | Hình 5 — ROC một-lớp-với-phần-còn-lại |
| `auprc_*_mean` | không có trong bài báo, báo cáo thêm |

⚠ Code gốc (`META-CXR/mhcac/utils.py:compute_metrics_for_tasks`) lấy trung bình
theo batch rồi trung bình các batch; module này tính trên cả split.

## Callers

`scripts/evaluate_stage1.py`, `model/lavis/tasks/image_text_pretrain.py`,
`training/evaluation/baselines.py`, `training/evaluation/config.py`,
`training/evaluation/paper_protocol.py` (dùng `roc_auc`, `PAPER_FIVE_FINDINGS`).

## Tests

`tests/test_classification_metrics.py` (có so khớp sklearn khi host có sklearn),
`tests/test_three_class_only.py`.
