# `scripts/compare_stage1_operating_points.py`

> Thư mục: [`scripts/`](_index.md) · Thêm 2026-11-01 · 🔬 phân tích (CPU, chạy trên máy train)

## Purpose
So các điểm vận hành của CÙNG một mô hình Stage 1 (`docs/handoff/PLAN-2026-11-01-recall-operating-point.md`):
cutpoint headline (weighted F1), cutpoint thiên recall `f<β>` (objective `present_fbeta` của
[`threshold_calibration`](../training/evaluation/threshold_calibration.py.doc.md)) và argmax. Không phải giao thức bài
báo; AUROC/AP không đổi, tăng recall là đổi lấy precision.

## Lệnh
- **`crossfit --val <npz> [--betas 1,1.5,2,3,4] [--repeats 10]`:** chỉ val; 2 nửa × `repeats` (seed 16), fit trên một
  nửa, chấm nửa kia, trung bình. Từ chối npz của test.
- **`paired --test <npz> --rule NAME=<file> [...]`:** file cutpoint fit trên val (từ chối file không phải `split: val`),
  quy tắc đầu là mốc, thêm argmax; bootstrap ghép cặp theo study, 1.000 lần, seed 16.

## Chỉ số
Recall / precision "có bệnh" (micro trên ô có nhãn), weighted F1 trung bình 14 finding, F1-5, macro recall, số finding
gọi "có bệnh" mỗi study (bỏ No Finding), recall "có bệnh" từng finding. Chỉ số tổng hợp.

## Hàm (ghim bởi `tests/test_recall_operating_point.py`)
`summarize`, `rule_decisions`, `crossfit` (không bao giờ chấm trên nửa đã fit), `paired`.
