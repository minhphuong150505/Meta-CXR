> Source: `configs/stage1_thresholds_f1_val.json` (184 dòng)
> Status: ✅ ACTIVE — tracked Stage-1 calibration artifact
> Last verified against source: 2026-08-12

# `configs/stage1_thresholds_f1_val.json`

## Purpose

Threshold phân loại theo từng bệnh lý, calibrate trên **validation split** cho
objective F1 với `uncertain_policy=ignore_uncertain` và `min_positive=20`.

File có ba phần: `thresholds` dùng khi chấm test, `details` lưu provenance/chỉ số
từng bệnh lý, và `metadata` xác nhận split `validation` cùng 1.788 sample. Bệnh
lý không đủ positive giữ threshold mặc định `0.5` và ghi lý do.

## Usage and safety

`scripts/evaluate_stage1.py` chỉ đọc file khi caller truyền `--thresholds`; nó
không bị load ngầm. Không calibrate lại trên test. File này khác schema của
[`threshold.json`](../threshold.json.doc.md), artifact lịch sử ở root.

Kết quả Table 5 dùng chính file này cho cả bốn cấu hình để comparison nhất quán.

← [`configs/`](_index.md) · [`results/`](../results/_index.md) · [HOME](../../HOME.md)
