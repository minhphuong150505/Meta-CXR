> Source: `pretraining/configs/ablation/pubmedclip.yaml` (157 dòng)
> Status: ✅ COMPLETED — inference-only, ⚠ zero-F1 result

# `ablation/pubmedclip.yaml`

Giữ `active_encoders: [pubmedclip]`, zero BioViL-T và SwinV2. Không retrain.
Kết quả năm pathology đều F1 0 vì không có positive prediction tại các threshold
validation cố định; artifact ghi nhận nguyên trạng sau kiểm tra probability.

← [`ablation/`](_index.md) · [`results`](../../../results/table5_encoder_ablation.md.doc.md)
