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
| `evaluate_classification(predictions)` | 262 | [📄](classification_metrics.py.methods/evaluate_classification.md) ★ Điểm vào |
| `weighted_prf(y_true, y_pred)` | 159 | sklearn `average='weighted'`, `zero_division=1` |
| `macro_recall(y_true, y_pred)` | 186 | recall ba lớp không trọng số = sklearn `balanced_accuracy_score` (D-024) |
| `per_class_prf(matrix)` | 138 | P/R/F1 từng lớp, ngữ nghĩa sklearn |
| `confusion_matrix` | 131 | 3×3, hàng = nhãn thật |
| `decide(probabilities)` | 257 | argmax ba lớp (quyết định của code gốc) |
| `roc_auc`, `average_precision` | 79, 108 | ROC (Mann-Whitney) và AP bậc thang |
| `PAPER_FIVE_FINDINGS` | 67 | 5 bệnh của Bảng 5 và 7 |
| `AGGREGATE_METRICS` | 365 | Tên aggregate hợp lệ (config, selection metric) |

## Aggregates — mapping tới bài báo

| Key | Bài báo |
|---|---|
| `weighted_precision` / `weighted_recall` / `weighted_f1` / `accuracy` | Hình 10, Sec. IV-B-2a (0.87 / 0.78 / 0.73) — trung bình đều 14 bệnh. `weighted_recall` trùng hoàn toàn với `accuracy` |
| `macro_recall` | **Không có trong paper.** Recall ba lớp không trọng số (sklearn `balanced_accuracy_score`, chỉ các lớp có trong ground truth), trung bình 14 bệnh. Metric chọn checkpoint Stage 1 từ 2026-10-01 (D-024) |
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

`tests/test_classification_metrics.py` (có so khớp sklearn khi host có sklearn,
kể cả `macro_recall` với `balanced_accuracy_score`), `tests/test_selection_metric.py`,
`tests/test_three_class_only.py`.
