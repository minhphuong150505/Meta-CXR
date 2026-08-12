> Source: `pretraining/configs/ablation/swin.yaml` (157 dòng)
> Status: ✅ COMPLETED — inference-only, ⚠ zero-F1 result

# `ablation/swin.yaml`

Giữ `active_encoders: [swin]`, zero BioViL-T và PubMedCLIP. Không retrain. Năm
pathology đều F1 0 tại threshold validation; probability khác rõ all-three nên
ablation đã active, không phải config bị bỏ qua.

← [`ablation/`](_index.md) · [`results`](../../../results/table5_encoder_ablation.md.doc.md)
