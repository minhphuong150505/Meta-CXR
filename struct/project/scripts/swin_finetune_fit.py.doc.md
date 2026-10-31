# `scripts/swin_finetune_fit.py`

> Thư mục: [`scripts/`](_index.md) · Thêm 2026-10-31 · 🔬 chẩn đoán (chạy trên máy train)

## Purpose
So sánh MedCLIP-Swin đóng băng (F) với mở khóa toàn bộ (T), cùng đầu probe và cùng budget, theo
`docs/handoff/PLAN-2026-10-31-swin-finetune-probe.md`. Đọc cache của
[`swin_finetune_cache.py`](swin_finetune_cache.py.doc.md); đầu probe là `build_attention_probe` của
[`encoder_probe_fit.py`](encoder_probe_fit.py.doc.md) trên 49 token lưới + token pooled của Swin.

## Lệnh
- **`fit --arm frozen|finetune`:** cổng G0 (key/nhãn của cache giống hệt thư mục của phép đo encoder;
  Swin đóng băng trên cache tái tạo đặc trưng Swin đã lưu, cosine ≥ 0,999 trên 64 study val), rồi 8 epoch
  trên 35.654 study fit, batch 64, bf16, AdamW, warmup 5% + cosine, augmentation chỉ lúc train (xoay ±10°,
  phóng 0,9–1,1, tịnh tiến ±5%, sáng/tương phản ×[0,9; 1,1], không lật). Chọn epoch theo macro AP trên
  3.961 study holdout (cùng split với phép đo encoder); val chỉ đọc ở epoch đã chọn. Đầu probe: LR 3e-4,
  weight decay 0,05 trên mọi tham số (như probe cũ). Nhánh T: Swin LR giảm theo tầng 0,7 (stage 4 và
  LayerNorm cuối 5e-5 … patch embedding 1,2e-5), không weight decay trên tensor 1 chiều, drop path của
  Swin bật khi train. **G1:** F đạt val macro AP ≥ 0,4203. **G2:** T đổi trọng số stage 1 và stage 4;
  F giữ mọi tensor Swin giống từng bit. Ghi `val_probs.npy` (chỉ trên máy) và `report.json` (số tổng hợp);
  cổng hỏng → thoát mã khác 0.
- **`compare`:** Δ = T − F macro AP "có bệnh" trên val, bootstrap ghép cặp (1.000, seed 16), kết luận
  `GO_ON` (Δ ≥ +0,02 và cận dưới > 0) / `NOT_ENOUGH` / `STOP` (Δ ≤ 0). `--mhcac NAME=npz` thêm tham chiếu
  mô tả (ghép theo key, kiểm nhãn).

## Hàm thuần (ghim bởi `tests/test_swin_finetune.py`)
`holdout_split`, `swin_stage`, `swin_param_groups`, `warmup_cosine`, `affine_theta`, `augment`,
`to_swin_input`, `verdict`.
