> Source: `training/evaluation/explanation_metrics.py::annotation_coverage`

# `annotation_coverage(cam, boxes, k=0.5, tau=0.01, *, mask_source=1)`

Eq. (9): với từng box `B_k`, tính phần pixel top-k salient trong box; box được
cover khi phần đó `>= tau`. Kết quả là số box cover chia tổng box, không phải
overlap với union mask.

`mask_source=0` trả `None` ngay vì lung mask không chứa expert box. `boxes=None`
hoặc rỗng cũng unavailable, không phải 0. Khi gộp nhiều study, `summarize` cộng
số box cover/tổng box thay vì lấy mean của các tỉ lệ per-study có số box khác
nhau.

← [`methods`](./_index.md)
