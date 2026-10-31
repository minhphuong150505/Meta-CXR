# `scripts/biovil_prior_fit.py`

> Thư mục: [`scripts/`](_index.md) · Thêm 2026-10-31 · 🔬 chẩn đoán (chạy trên máy train)

## Purpose
Probe attention (`fit_attention` của [`encoder_probe_fit.py`](encoder_probe_fit.py.doc.md), không đổi) trên đặc trưng
của [`biovil_prior_features.py`](biovil_prior_features.py.doc.md): `swin`, `biovil`, `biovilprior`, `swin+biovil` (A),
`swin+biovilprior` (B).

## Cổng và quy tắc
- **G0:** npz val của R1C tính lại đúng 0,4271, nhãn khớp sau khi ghép key.
- **G1:** `attn_swin` / `attn_biovil` / `attn_swin+biovil` lệch ≤ 0,003 so với 0,4263 / 0,4039 / 0,4315 của phép đo encoder.
- **Quyết định:** Δ = B − A macro AP "có bệnh" trên toàn val, bootstrap ghép cặp: `GO_ON` (≥ +0,02, cận dưới > 0) /
  `NOT_ENOUGH` / `STOP` (≤ 0). Mô tả: cùng Δ trên nhóm có ảnh trước, `biovilprior − biovil`, `B − swin`.
