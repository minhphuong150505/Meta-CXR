# `scripts/probe_image_features.py`

> Thư mục: [`scripts/`](_index.md) · Thêm 2026-10-11 · 🔬 chẩn đoán

## Purpose
Linear probe **cùng giao thức** cho nhiều biểu diễn ảnh, so với chính MHCAC trên cùng
study test và cùng nhãn (D-018): trung bình 32 soft token (từ cache Stage 2) và các
embedding ngoài (`--embedding tên=<train.npz>,<test.npz>`, vd. ảnh MedGemma từ
[`embed_medgemma_images.py`](embed_medgemma_images.py.doc.md)). Fit trên tập train có
mặt ở mọi nguồn, chọn L2 trên 10% giữ lại của TRAIN, chấm test một lần. Chỉ in số tổng hợp.
