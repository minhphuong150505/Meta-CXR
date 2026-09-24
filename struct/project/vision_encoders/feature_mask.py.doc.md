> Source: `vision_encoders/feature_mask.py`
> Status: ✅ ACTIVE khi `model.feature_mask_ratio > 0` (production 0.1)
> Last verified against source: 2026-09-24

# `vision_encoders/feature_mask.py`

← [vision_encoders](_index.md) · [D-020](../_meta/DECISIONS.md#d-020--meta-former-3-pha-theo-bài-báo-medclip-swin-mhcac-một-nhánh)

Bài báo: "randomly mask 10% of the features from each encoder". Zero
`round(ratio × P)` token ngẫu nhiên của mỗi mẫu, độc lập theo mẫu, trên chuỗi
riêng của từng encoder (anchor và view phụ), chỉ lúc train, trước
StreamAdapter/fusion/projection. Thay `Blip2Qformer._create_mask` (không bao giờ
được gọi, và mask chung vị trí cho cả batch trên chuỗi ghép) — đã xóa.
BioViL 196 → 20 token, PubMedCLIP/Swin 50 → 5.

Test: `tests/test_paper_mode.py`.
