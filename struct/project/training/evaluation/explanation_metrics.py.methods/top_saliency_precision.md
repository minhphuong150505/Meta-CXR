> Source: `training/evaluation/explanation_metrics.py::top_saliency_precision`

# `top_saliency_precision(cam, mask, k=0.5)`

Eq. (7): tạo mask saliency nhị phân có đúng `ceil(k·H·W)` pixel cao nhất, rồi
trả tỉ lệ số pixel đó nằm trong annotation mask.

Khác `mhcac.explanation.explanation_loss`, đây là metric nên threshold nhị phân
là đúng và không cần khả vi. Tie tại cutoff được giải ổn định theo flat index để
không giữ quá k% pixel như cách so sánh `>= quantile`.

Risk: đổi lại gate mềm sẽ biến metric thành saliency-mass precision và không còn
khớp Eq. (7).

← [`methods`](./_index.md)
