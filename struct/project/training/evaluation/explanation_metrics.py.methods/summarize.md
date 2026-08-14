> Source: `training/evaluation/explanation_metrics.py::summarize`

# `summarize(cams, masks, mask_sources, boxes_by_sample, ...)`

## Contract

Nhận batch NumPy `[N,H,W]`, source `[N]` và box rời tùy chọn; trả hai field bắt
buộc `lung` và `bbox`. Không có field aggregate toàn bộ.

Mỗi metric mang `num_samples`; annotation coverage còn mang `num_boxes`. Nhóm
không có dữ liệu trả `value=null` cùng `unavailable_reason`, không đưa số 0 giả
vào báo cáo.

## Aggregation

- top/all: mean trên các sample có metric định nghĩa;
- coverage: tổng box covered / tổng expert box, có thể khác mean per-study;
- lung coverage: luôn unavailable.

← [`methods`](./_index.md)
