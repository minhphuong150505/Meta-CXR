# `scripts/encoder_probe_fit.py`

> Thư mục: [`scripts/`](_index.md) · Thêm 2026-10-31 · 🔬 chẩn đoán (chạy trên máy train)

## Purpose
Probe trên đặc trưng của [`encoder_probe_features.py`](encoder_probe_features.py.doc.md) và quy tắc
đọc của `docs/handoff/PLAN-2026-10-31-encoder-probe.md`. Fit trên train (10% holdout chọn L2/epoch),
chấm một lần trên val.

- **Tuyến tính:** softmax 3 lớp mỗi finding trên vector toàn cục đã chuẩn hóa; L2 ∈ {1e-4…1e-1}.
- **Attention (chính):** LayerNorm → chiếu 256 + vị trí học được cho từng encoder, vector toàn cục là
  một token thêm, 14 query (một mỗi finding), attention một đầu, đầu ra tuyến tính 3 lớp; AdamW +
  OneCycle, 12 epoch, chọn epoch trên holdout, trung bình xác suất của 3 seed.
- **Tổ hợp:** Swin + từng encoder, bộ ba đang dùng, tất cả.

## Chỉ số và quy tắc
Macro AP "có bệnh" (P hoặc U so với N, trên `p_pos/(p_pos+p_neg)`, 14 finding; `ap_auc` cùng số học
với `recall_metrics.ap_auc_rp` của các pilot). Encoder X là **ứng viên** khi
`attn(X) − attn(swin)` hoặc `attn(swin+X) − attn(swin)` ≥ +0,02 với cận dưới bootstrap ghép cặp
(1.000, seed 16) > 0. **Cổng G0:** macro AP tính lại từ npz val của MHCAC (`--mhcac`) phải khớp
`--expect` trong 5e-5, và nhãn khớp nhãn của probe sau khi ghép theo key.

## Hàm thuần (ghim bởi `tests/test_encoder_probe.py`)
`ap_auc`, `presence_scores`, `per_finding`, `macro_ap`, `paired_bootstrap`, `build_attention_probe`.
