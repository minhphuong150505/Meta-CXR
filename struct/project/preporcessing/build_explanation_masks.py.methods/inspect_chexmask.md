> Source: `preporcessing/build_explanation_masks.py::inspect_chexmask`
> Status: 🧰 UTILITY

# `inspect_chexmask(csv_path, sample_rows=48)`

Đọc header và một prefix nhỏ, rồi in:

- danh sách tên cột;
- số hàng đã đọc;
- khoảng giá trị `Dice RCA (Mean)`.

Không bao giờ in giá trị `dicom_id`. Chế độ CLI `--inspect` không import
`local_config`, nên chạy được trước khi cấu hình manifest/cache.

Dùng nó trước mỗi lần build: header thật của bản MIMIC (xác minh trên dữ liệu
ngày 2026-08-14) là `dicom_id, Dice RCA (Mean), Dice RCA (Max), Landmarks,
Left Lung, Right Lung, Heart, Height, Width` — khác với DATA_DICTIONARY công bố
trên PhysioNet, vốn viết chung cho cả năm dataset và gọi cột định danh là
`Image ID`. Nếu bản tải về sau này đổi schema, `_validate_columns` phải nổ chứ
không được đoán.

← [`functions`](./_index.md)
