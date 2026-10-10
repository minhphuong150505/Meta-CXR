> Source: `training/evaluation/threshold_calibration.py` (160 dòng)
> Status: ✅ ACTIVE
> Last verified against source: 2026-10-10

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
| `apply_thresholds(probs, thresholds, names)` | 86 | Quyết định theo ngưỡng; lớp vượt ngưỡng xa nhất thắng; không lớp nào → `ABSTAIN` (prompt Stage 2) |
| `decide_with_thresholds(probs, thresholds, names)` | 116 | Như trên nhưng không lớp nào vượt → argmax; cho `evaluate_classification(..., thresholds=)` và `evaluate_stage1.py --thresholds` (phân tích bổ sung, 2026-10-01) |
| `severity_scores`, `decide_with_cutpoints`, `fit_severity_cutpoints`, `apply_cutpoints` | cuối file | Ngưỡng theo thứ tự trên `p_pos/(p_pos+p_neg)` (2026-10-01). **Từ 2026-10-10 là luật quyết định cho số CHÍNH của dự án** (quyết định của user), vẫn không phải giao thức argmax của bài báo; dùng bởi `scripts/evaluate_stage1_cutpoints.py`, `calibrate_thresholds.py --rule cutpoints`, `evaluate_classification(..., cutpoints=)` |
| `CutpointFile.save` / `load_cutpoint_file` / `load_cutpoints` | cuối file | File JSON `format = meta_cxr_severity_cutpoints_v1`, `{bệnh: [t1, t2]}` + metadata (`split`, `num_samples`, `checkpoint`); từ chối format khác và `t1 > t2`. File của model báo cáo: `configs/stage1_cutpoints/run_20261005_paper.json` |
| `ThresholdFile.save` / `load_thresholds` | 117 / 138 | File JSON có `format = meta_cxr_per_class_roc_distance_v1`; file định dạng khác bị từ chối |

## Callers

`scripts/calibrate_thresholds.py`; `scripts/evaluate_stage1.py` (`--thresholds`,
`--cutpoints`); `training/train_eval_figure9_llm_variants_200.load_thresholds`
(Stage 2). Test: `tests/test_threshold_calibration.py`,
`tests/test_stage1_cutpoint_headline.py`.
