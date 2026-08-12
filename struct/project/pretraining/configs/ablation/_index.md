> Source: `pretraining/configs/ablation/`
> Status: ✅ COMPLETED EXPERIMENT — Table 5 inference-only
> Last verified against source: 2026-08-12

# `pretraining/configs/ablation/`

Bốn config cùng dựng checkpoint ba encoder đã train, rồi dùng `active_encoders`
để giữ một hoặc cả ba stream tại inference. Chúng có `train_splits: []`,
`valid_splits: []`, `test_splits: [test]`: đây không phải bốn lần retrain.

| Config | Stream giữ lại | Kết quả |
|---|---|---|
| [`biovil.yaml`](biovil.yaml.doc.md) | BioViL-T | mean F1 0.5249 |
| [`pubmedclip.yaml`](pubmedclip.yaml.doc.md) | PubMedCLIP | mean F1 0.0000 ⚠ |
| [`swin.yaml`](swin.yaml.doc.md) | SwinV2 | mean F1 0.0000 ⚠ |
| [`all_three.yaml`](all_three.yaml.doc.md) | cả ba | mean F1 0.5250; equivalence gate pass |

Tất cả chấm full 3.216-study test split bằng threshold calibrate trên validation.
Hai dòng zero-F1 được giữ nguyên sau khi xác nhận ablation thực sự active.

← [`pretraining/configs/`](../_index.md) · [`results/`](../../../results/_index.md)
