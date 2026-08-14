> Source: `training/evaluation/explanation_metrics.py::all_saliency_precision`

# `all_saliency_precision(cam, mask)`

Eq. (8): `sum(cam * mask) / sum(cam)` trên toàn bản đồ liên tục, không threshold.

Nếu CAM không có khối lượng (`sum=0`), kết quả là `None`/unavailable. Trả 0 ở
đây sẽ đánh đồng “không có explanation để chấm” với “mọi saliency nằm ngoài
annotation”.

← [`methods`](./_index.md)
