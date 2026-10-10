# `training/dataio/oracle_cues.py`

> Thư mục: [`training/dataio/`](_index.md) · Thêm 2026-10-11 · 🧪 thí nghiệm (`--train-cues oracle_mix`)

## Purpose
Cue P/N/U cho record **train** lấy từ nhãn CheXpert THẬT, trừ một tỉ lệ
`model_share` (mặc định 0,2, tất định theo seed + study + bệnh) giữ lớp MHCAC dự đoán.
Lý do: mô hình v3 bỏ qua kênh cue — thay cue bằng nhãn thật không đổi chỉ số nào
(1.000 study test) — nên phân loại tốt hơn không thể cải thiện báo cáo; train trên cue
thường đúng để mô hình học dựa vào cue. Val/test vẫn dùng cue của MHCAC.

## Main items
| Tên | Vai trò |
|---|---|
| `map_chexpert_rows(frame, findings)` | quy tắc D-018 (trống = âm tính nếu study có giá trị nào; không có gì = không biết), viết lại bằng pandas để không import LAVIS |
| `true_label_codes(records, split_csv, chexpert_csv, findings)` | mã lớp thật theo DICOM ảnh neo |
| `mix_oracle_cues(records, codes, findings, model_share, seed)` | ★ thay `pred_groups`, đặt `cue_source="oracle_mix"`, trả số đếm |

`tests/test_oracle_cues.py` ghim ánh xạ khớp `model/lavis/data/chexpert_labels.map_chexpert_labels`.
