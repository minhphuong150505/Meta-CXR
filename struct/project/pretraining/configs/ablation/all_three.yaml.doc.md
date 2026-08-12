> Source: `pretraining/configs/ablation/all_three.yaml` (157 dòng)
> Status: ✅ COMPLETED — inference-only reference

# `ablation/all_three.yaml`

Giữ `active_encoders: [biovil, pubmedclip, swin]`; resolver trả tuple rỗng nên
đường tensor phải tương đương model gốc. Equivalence gate pass: mean F1 0.524952
so với expected 0.5250, trong tolerance ±0.005.

← [`ablation/`](_index.md) · [`results`](../../../results/table5_encoder_ablation.md.doc.md)
