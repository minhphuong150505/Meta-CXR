> Source: `training/evaluation/threshold_calibration.py` (160 dòng)
> Status: ✅ ACTIVE
> Last verified against source: 2026-09-29

# `threshold_calibration.py`

## Purpose

Ngưỡng **theo từng (bệnh, lớp)** như bài báo (Sec. V-C, Eq. 22, Hình 11): trên ROC
một-lớp-với-phần-còn-lại, chọn ngưỡng gần góc trên-trái nhất
`sqrt((1-TPR)^2 + FPR^2)`. Chỉ fit trên **validation**. Dùng cho prompt Stage 2
(`--cue-rule paper_thresholds`), không dùng cho chỉ số phân loại (luôn argmax).

Thay hoàn toàn bản calibrate ngưỡng dương tính nhị phân cũ (D-023).

## Main items

| Tên | Dòng | Vai trò |
|---|---|---|
| `roc_distance_threshold(scores, y)` | 45 | Eq. 22 cho một lớp |
| `fit_class_thresholds(predictions)` | 70 | ★ `{bệnh: {lớp: ngưỡng}}` |
| `apply_thresholds(probs, thresholds, names)` | 86 | Quyết định theo ngưỡng; lớp vượt ngưỡng xa nhất thắng; không lớp nào → `ABSTAIN` |
| `ThresholdFile.save` / `load_thresholds` | 117 / 138 | File JSON có `format = meta_cxr_per_class_roc_distance_v1`; file định dạng khác bị từ chối |

## Callers

`scripts/calibrate_thresholds.py`; `training/train_eval_figure9_llm_variants_200.load_thresholds`
(Stage 2).
