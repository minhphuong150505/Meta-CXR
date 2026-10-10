# `scripts/prior_report_baseline.py`

> Thư mục: [`scripts/`](_index.md) · Thêm 2026-10-11 · 🔬 chẩn đoán

## Purpose
Mốc "chép báo cáo lần chụp trước": với mỗi study test, lấy FINDINGS hợp lệ của study
**sớm hơn gần nhất** của cùng bệnh nhân (theo `StudyDate`/`StudyTime` trong
metadata). Ghi tập con có lần trước (`prior_subset.jsonl`), dự đoán của mô hình trên
cùng tập con, và bản trộn "lần trước nếu có, không thì mô hình". Đo giá trị của bối
cảnh lần chụp trước đối với metric NLG. Văn bản báo cáo → chỉ trên máy train.
