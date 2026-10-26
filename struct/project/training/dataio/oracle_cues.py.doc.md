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

## ⚠ "Study" trong lượt bốc là `sample_key` của record (ghi nhận 2026-10-26)

`_uniform(seed, key, finding)` dùng `key = record["sample_key"]`. Với record Stage-1,
`sample_key = stable_fingerprint({cohort_id, dicom_id})`, và `cohort_id` chứa `file_identity()`
(kích thước + `mtime_ns`) của YAML Stage-1, checkpoint và CSV split. Hệ quả: cùng seed, cùng commit
nhưng clone mới (mtime khác) cho lượt bốc khác; hai nhánh có checkpoint / cutpoints /
`--stage1-options` khác nhau cũng có lượt bốc khác nhau. Seed truyền từ
`run_medgemma_qlora.py --seed` (mặc định 16).

**Đã sửa trên branch `fix/oracle-mix-stable-key` (2026-10-26, chưa merge):** `draw_key(record)` trả
về tên file (không đuôi) của ảnh anchor, tức dicom_id, và chỉ dùng `sample_key` khi record không có
`image_path`. Lượt bốc giờ chỉ phụ thuộc seed, study và bệnh; mọi kết quả `oracle_mix` ghi trước
bản sửa không tái lập được trên branch này.

| Hàm | Vai trò |
|---|---|
| `draw_key(record)` | khóa study của lượt bốc: stem của `image_path`, dự phòng `sample_key` |
