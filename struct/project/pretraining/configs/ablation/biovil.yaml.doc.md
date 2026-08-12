> Source: `pretraining/configs/ablation/biovil.yaml` (157 dòng)
> Status: ✅ COMPLETED — inference-only

# `ablation/biovil.yaml`

Giữ `active_encoders: [biovil]`, zero PubMedCLIP và SwinV2 sau shared projection.
`run_name: ablation_biovil`; không có train/validation split. Table 5 ghi mean F1
0.5249 trên full test split.

← [`ablation/`](_index.md) · [`blip2_qformer.py`](../../../model/lavis/models/blip2_models/blip2_qformer.py.doc.md)
