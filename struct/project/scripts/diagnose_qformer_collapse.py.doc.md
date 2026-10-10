# `scripts/diagnose_qformer_collapse.py`

> Thư mục: [`scripts/`](_index.md) · Thêm 2026-10-11 · 🔬 chẩn đoán (chạy trên máy train)

## Purpose
Đo **ở phase nào và vì sao** 32 query của META-Former (Q-Former) sụp thành một
vector. Nạp một checkpoint Stage 1 (hoặc `none` = khởi tạo BLIP-2), chạy
`forward_image` trên `--limit` study, móc vào `Qformer.bert` để lấy hidden state
mọi tầng và attention của các tầng cross-attention.

## Entry point
```bash
python scripts/diagnose_qformer_collapse.py --checkpoint <ckpt.pth | none> --label <tên> \
    --limit 200 --report <private>/qformer_<tên>.json
```

## Đo gì (trung bình theo study)
- cosine giữa 32 token trong một study và participation ratio (số hướng hiệu dụng)
  sau embedding và sau từng tầng;
- mỗi tầng cross-attention: entropy chuẩn hoá (1 = đều), trọng số lớn nhất, độ giống
  nhau giữa bản đồ attention của các query, khối lượng attention theo encoder, phần
  attention trên token đầu mỗi stream (CLS PubMedCLIP, token pooled Swin), token
  nhận attention nhiều nhất;
- token ảnh: norm và cosine trong study theo stream;
- phản thực tế: thay mọi token ảnh bằng token trung bình của study rồi so đầu ra.

## Kết quả (2026-10-11)
Sụp đổ xảy ra ở **phase 1a** (cosine 0,9989, PR 1,002; khởi tạo BLIP-2 0,445 / 2,94),
attention dồn vào một token (entropy ~0,1); phase 1b làm norm token PubMedCLIP/Swin
tăng 19,6→1.713 / 12→413 khi Q-Former đã đóng băng. Xem
`docs/handoff/PLAN-2026-10-11-soft-token-probe.md` và
`docs/RESEARCH-2026-10-11-stage1-stage2-diagnosis.md`.
