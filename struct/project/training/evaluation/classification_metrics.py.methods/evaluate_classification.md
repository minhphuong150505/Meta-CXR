> Source: `training/evaluation/classification_metrics.py:296-551`
> Status: ✅ ACTIVE

# `evaluate_classification(predictions, ...)`

## Purpose

Điểm vào metric Stage 1: áp uncertain policy/threshold, tính per-pathology P/R/F1,
AUROC/AUPRC, confusion 3 lớp và aggregate macro/micro/weighted.

## Inputs / outputs

Nhận `ClassificationPredictions` với probabilities/logits, labels, pathology
names và optional sample mask; trả `ClassificationReport`. Shape trung tâm
`[N,14,3]`, nhưng số pathology lấy từ schema input.

## Edge cases

Class không có positive/negative hợp lệ → probability metric `nan`, không ép 0.
Threshold mapping thiếu pathology dùng contract mặc định của helper; policy
uncertain được validate trước.

## Called by / tests

Stage-1 eval hook, `scripts/evaluate_stage1.py`, baselines;
`tests/test_classification_metrics.py`.

← [`classification_metrics.py`](../classification_metrics.py.doc.md) · [HOME](../../../../HOME.md)
