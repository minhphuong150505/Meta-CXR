# `scripts/embed_cxr_encoders.py`

> Thư mục: [`scripts/`](_index.md) · Thêm 2026-10-11 · 🔬 chẩn đoán (chạy trên máy train)

## Purpose
Đặc trưng gộp **từng encoder** cho probe tuyến tính so sánh encoder Stage 1 (với
[`probe_image_features.py`](probe_image_features.py.doc.md)): BioViL-T (trung bình patch),
PubMedCLIP (CLS), MedCLIP-Swin (token pooled) và tùy chọn **RAD-DINO** (`--raddino <thư mục
local>`, [CLS ; trung bình patch]) — encoder repo hỗ trợ nhưng đang tắt. Chạy qua dataset
Stage 1 nên mỗi encoder nhận đúng tiền xử lý của nó; dùng cùng tập study con (seed/limit)
với các probe khác. `--checkpoint none` = encoder pretrained.
