> Source: `scripts/evaluate_explanation.py::_evaluate`

# `_evaluate(args, output_dir)`

Duyệt manifest-selected split theo thứ tự, bỏ study không có classification,
không có mask hoặc không có nhãn positive. Với source bbox, join MS-CXR bằng ID
nhưng không đọc cột split, chèn marker schema tổng hợp cho geometry helper,
transform từng box riêng rồi tính CAM.

Các CAM được tích lũy theo stream nhưng metric gọi `summarize` riêng, giữ hai
population lung/bbox. Hàm trả payload JSON không identifier; tùy flag ghi NPZ và
N figure đầu. Counter ghi rõ số study skip và số box bị crop hết.

Risk: không được thay `boxes_by_sample` bằng union cache mask khi tính annotation
coverage; làm vậy biến Eq. (9) thành metric khác.

← [`methods`](./_index.md)
