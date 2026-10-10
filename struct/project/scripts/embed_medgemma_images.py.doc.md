# `scripts/embed_medgemma_images.py`

> Thư mục: [`scripts/`](_index.md) · Thêm 2026-10-11 · 🔬 chẩn đoán

## Purpose
Embedding ảnh của chính MedGemma (MedSigLIP + projector, 256 token mà LLM đọc trong
`medgemma_direct`), trung bình theo token, cho các study của một cache Stage 2. Chọn
cùng tập con theo `--limit/--seed` như `retrieval_report_baselines.py` để so công bằng
với soft token. Ghi `.npz` (`keys`, `emb` float16) — dữ liệu dẫn xuất, giữ trên máy train.
