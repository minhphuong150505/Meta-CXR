# `scripts/evaluate_stage1_cutpoints.py`

> Thư mục: [`scripts/`](_index.md) · Thêm 2026-10-01 · ✅ ACTIVE (phân tích bổ sung)

## Purpose
Phân tích **bổ sung, không phải giao thức của bài báo** (bài báo dùng argmax và
argmax vẫn là kết quả chính). Ca Uncertain nằm giữa ca âm và ca dương trên điểm
`s = p_pos / (p_pos + p_neg)`, nên mỗi bệnh có thể quyết định bằng ngưỡng trên `s`.

## Entry point
```bash
python scripts/evaluate_stage1_cutpoints.py --val <val.npz> --test <test.npz> \
    --output-dir <private dir> [--bootstrap-samples 1000] [--seed 16]
```

## Làm gì
- Fit trên **val**, theo từng bệnh, tối đa weighted F1: (a) một ngưỡng (âm/dương,
  không bao giờ đoán Uncertain), (b) hai ngưỡng (âm / dải Uncertain / dương).
  Hàm: `threshold_calibration.fit_severity_cutpoints`, `apply_cutpoints`.
- Chấm **test** bằng argmax, (a), (b); bootstrap ghép cặp theo study, so với argmax.
- Ghi `cutpoints_report.json` (metric, số lần đoán Uncertain, recall/precision
  Uncertain, ngưỡng, chênh lệch kèm CI 95%). AUROC không phụ thuộc luật quyết
  định nên không báo ở đây.
- Từ chối (exit 2) nếu file fit là split test, hoặc val/test khác danh sách bệnh.

## Kết quả đã đo (run_20260930_3class, test n=3.269)
Xem `docs/handoff/PLAN-2026-09-30-stage1-three-class-retrain.md`: gần như toàn bộ
cải thiện đến từ việc dời ngưỡng âm/dương; dải Uncertain thêm không đáng kể.

## Tests
`tests/test_threshold_calibration.py` (cutpoint, fit, CLI).
