# `scripts/analyze_radiologist_labels.py`

> Thư mục: [`scripts/`](_index.md) · Thêm 2026-10-11 · 🔬 chẩn đoán

## Purpose
Đo **nhiễu nhãn** và chấm Stage 1 theo **nhãn bác sĩ**: MIMIC-CXR-JPG 2.1.0 có
`mimic-cxr-2.1.0-test-set-labeled.csv` (687 study test gán tay; "Airspace Opacity" =
"Lung Opacity"). Cho từng bệnh: độ khớp ba lớp và Cohen's kappa giữa nhãn tự động
CheXpert (thứ mô hình học và được chấm) với nhãn bác sĩ, precision/recall dương tính của
nhãn tự động; với file dự đoán `.npz`: AUROC và weighted F1 tính theo hai bộ nhãn trên
cùng study, bootstrap ghép cặp. `sample_keys` của `.npz` là digest của `image_id` (vị trí
ảnh neo trong annotation), nên `--stage1-config` dựng lại dataset (chỉ đọc CSV) để ghép.

## Kết quả (2026-10-11, 685 study)
Kappa 0,49 (Pleural Other) đến 0,93 (Atelectasis); precision dương tính của nhãn tự động
chỉ 0,42–0,46 ở No Finding, Enlarged Cardiomediastinum, Pneumonia, Pleural Other.
