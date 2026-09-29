> Source: `training/evaluation/baselines.py` (127 dòng)
> Status: ✅ ACTIVE
> Last verified against source: 2026-09-29

# `baselines.py`

Baseline tầm thường **ba lớp**, chấm bằng cùng `evaluate_classification`:
`all_negative`, `all_positive`, `all_uncertain`, `majority_class` (lớp phổ biến
nhất của từng bệnh), `prior_random` (rút theo tần suất lớp). Cột bảng:
`BASELINE_COLUMNS` = weighted P/R/F1 + `mean_weighted_f1_5`. Viết lại 2026-09-29
(D-023) — bản cũ tính F1 lớp dương nhị phân.

⚠ `majority_class` và `prior_random` đọc tần suất của chính split được chấm → là
sàn lạc quan.

Caller: `scripts/evaluate_stage1.py`.
