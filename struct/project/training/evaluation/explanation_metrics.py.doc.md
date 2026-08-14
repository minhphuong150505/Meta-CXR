> Source: `training/evaluation/explanation_metrics.py`
> Status: ✅ ACTIVE — lõi NumPy cho đánh giá XAI
> Last verified against source: 2026-08-14

# `explanation_metrics.py`

## Purpose

Cài ba metric ở mục III.C của Faruqui & Dubey, nhận CAM đã min–max về `[0,1]`:

- top saliency precision, Eq. (7): top-k **nhị phân** nằm trong annotation;
- all saliency precision, Eq. (8): phần khối lượng CAM liên tục nằm trong mask;
- annotation coverage, Eq. (9): chấm từng bbox riêng với `tau=0.01`.

File chỉ import stdlib + NumPy. Không import torch, LAVIS hay `mhcac`, nên test
được trên máy dev không có torchvision.

## Population boundary

`summarize` trả `ExplanationSummary(lung=..., bbox=...)` và cố ý không có
`overall`:

| Source | Ý nghĩa | Annotation coverage |
|---|---|---|
| `0 = lung` | anatomical prior: model không nhìn ra ngoài phổi | `None`, unavailable |
| `1 = bbox` | expert MS-CXR annotation: model nhìn vào vùng bệnh | tính trên box rời |

Không được lấy trung bình hai nhóm vì 189 lung/4 bbox trong smoke cache thật có
ý nghĩa lâm sàng hoàn toàn khác nhau.

## Main API

| Function/type | Doc | Contract |
|---|---|---|
| `top_saliency_precision` | [📄](explanation_metrics.py.methods/top_saliency_precision.md) | Eq. (7), giữ đúng `ceil(k·H·W)` pixel |
| `all_saliency_precision` | [📄](explanation_metrics.py.methods/all_saliency_precision.md) | Eq. (8), CAM rỗng → unavailable |
| `annotation_coverage` | [📄](explanation_metrics.py.methods/annotation_coverage.md) | Eq. (9), từng bbox riêng |
| `summarize` | [📄](explanation_metrics.py.methods/summarize.md) | report tách lung/bbox + số mẫu |
| `MetricResult` | — | `value`, `num_samples`, tùy chọn `num_boxes`, reason |
| `PopulationSummary` / `ExplanationSummary` | — | schema không có mixed aggregate |

## Box representation

Nhận một trong hai dạng trên chính lưới CAM:

- `[num_boxes, H, W]` binary masks; hoặc
- `[num_boxes, 4]` tọa độ half-open `(x0,y0,x1,y1)`.

Script GPU dùng dạng mask để tái sử dụng chính xác geometry đã kiểm chứng của
`build_explanation_masks.py` cho từng box.

## Edge cases

- CAM ngoài `[0,1]`, NaN/Inf, mask rỗng hoặc sai shape: fail-closed.
- All-saliency với tổng CAM bằng 0: `None`, không báo 0.
- Lung hoặc bbox list rỗng cho annotation coverage: `None`, không báo 0.
- Tie ở cutoff top-k: flat index chỉ là secondary key ổn định, bảo đảm cardinality
  đúng thay vì `>= quantile` giữ quá nhiều pixel.

## Calls / Called by

Gọi: NumPy. Được gọi bởi `scripts/evaluate_explanation.py` và
`tests/test_explanation_metrics.py`.

## Tests

[`tests/test_explanation_metrics.py`](../../tests/_index.md#nhóm-6--evaluation):
biên 0/1, exact top-k, ngưỡng coverage inclusive, hai bbox riêng, unavailable và
tách source.

← [`_index.md`](_index.md) · [HOME](../../../HOME.md)
