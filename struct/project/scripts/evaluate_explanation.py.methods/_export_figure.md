> Source: `scripts/evaluate_explanation.py::_export_figure`

# `_export_figure(path, image, cams, mask, source, boxes)`

Xuất một PNG gồm ảnh gốc đã transform và một panel heatmap cho mỗi encoder
stream. Lung mask vẽ contour cyan; từng MS-CXR box vẽ riêng màu trắng. Figure,
title và filename không chứa patient/image identifier.

Matplotlib import trễ và chỉ bắt buộc khi `--export-figures > 0`; thiếu package
thì dừng với hướng dẫn extra `eval-plots` thay vì bỏ ảnh âm thầm.

← [`methods`](./_index.md)
