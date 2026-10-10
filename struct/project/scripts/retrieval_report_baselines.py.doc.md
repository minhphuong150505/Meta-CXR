# `scripts/retrieval_report_baselines.py`

> Thư mục: [`scripts/`](_index.md) · Thêm 2026-10-11 · 🔬 chẩn đoán

## Purpose
Mốc truy hồi: với mỗi study test, chép FINDINGS của study train gần nhất theo một
"khoá" rồi chấm như báo cáo sinh. Khoá: `random`, `constant` (báo cáo train phổ biến
nhất), `softtok` (cosine trên trung bình 32 soft token), `mhcac` / `mhcac_cutpoints`
(nhãn dự đoán so với nhãn THẬT của train, Hamming), `oracle` (nhãn thật của test),
`npz_<tên>` (embedding ngoài, vd. ảnh MedGemma từ
[`embed_medgemma_images.py`](embed_medgemma_images.py.doc.md)). Đọc cache
`.sensitive_stage1_cache` của Stage 2; ghi JSONL chứa văn bản báo cáo → chỉ trên máy train.
