# `scripts/score_ce_chexbert.py`

> Thư mục: [`scripts/`](_index.md) · Thêm 2026-10-11 · ✅ ACTIVE (venv riêng `~/.venvs/chexbert`)

## Purpose
**Clinical Efficacy (CE) bằng CheXbert** (`f1chexbert`, chế độ "rrg": Positive hoặc
Uncertain = 1) cho báo cáo sinh và tham chiếu: P/R/F1 micro và macro trên 14 nhãn và
trên 5 bệnh, cộng so sánh ghép cặp với `--baseline` (bootstrap). Thay CE lexicon ở mọi
chỗ cần số CE thật (user duyệt cài 2026-10-11).

Chạy trong venv RIÊNG, torch CPU, để không đụng torch/transformers của venv train:
`uv venv --seed ~/.venvs/chexbert --python 3.11`, cài `torch` (index CPU), `f1chexbert` và
`transformers==4.53.2` (bản 5.x bỏ `encode_plus`). Trọng số `chexbert.pth` (sha256
`6550703c…0fde1`) đặt ở `~/.cache/chexbert/`; CDN HF tới máy train chỉ ~9 KB/s nên tải nơi
khác, kiểm checksum rồi chép sang; chạy với `HF_HUB_OFFLINE=1`. `--label-cache` lưu vector nhãn theo
hash văn bản (không lưu văn bản).
