> Source: `vision_encoders/swin/medclip_swin.py`
> Status: ✅ ACTIVE — backend `medclip` của `SwinEncoder`, `ReportDataset`
> Last verified against source: 2026-09-24

# `vision_encoders/swin/medclip_swin.py`

← [vision_encoders](_index.md) · [D-020](../_meta/DECISIONS.md#d-020--meta-former-3-pha-theo-bài-báo-medclip-swin-mhcac-một-nhánh)

MedCLIP Swin-Tiny (Wang et al., EMNLP 2022) — encoder Swin mà bài META-CXR dùng.

- `medclip_preprocess(PIL "L")` → `[3, 224, 224]`: pad vuông nền đen → resize
  224 bicubic → /255 → (x − 0.5862785803043838) / 0.27950088968644304 → lặp 3
  kênh. Tái hiện `MedCLIPFeatureExtractor` dưới transformers 4.24; ghim với
  fixture `tests/fixtures/medclip_preprocess_reference.npz` (ảnh tổng hợp) do
  `scripts/make_medclip_preprocess_reference.py` sinh. Package `medclip` không
  tiền xử lý được dưới transformers 4.53 (tham số theo vị trí bị lệch).
- `build_medclip_swin(weights_path)`: HF `SwinModel` + `vision_model.model.*`
  của MedCLIP, 0 thiếu / 0 thừa, không thì lỗi. Không dùng `projection_head`
  768→512.
- `join_pooled_and_patches`: `[B, 50, 768]`, pooled ở vị trí 0.

Weights: `model.swin.weights_path` (zip trong `medclip_vit_weight.txt`, giải nén).
