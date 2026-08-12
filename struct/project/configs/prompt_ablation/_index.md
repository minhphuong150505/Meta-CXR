> Source: `configs/prompt_ablation/` (9 file YAML)
> Status: 🧪 ABLATION
> Last verified against source: 2026-08-12

# `configs/prompt_ablation/`

## Purpose
Chín biến thể prompt để đo tác động của từng chính sách.

## Children
| File | Giả thuyết được kiểm |
|---|---|
| [`P1_legacy_style.yaml`](P1_legacy_style.yaml.doc.md) | Prompt legacy làm chuẩn so sánh |
| [`P2_pos_unc_no_neg.yaml`](P2_pos_unc_no_neg.yaml.doc.md) | Bỏ hẳn negative có tốt hơn không |
| [`P3_pos_unc_critical_neg.yaml`](P3_pos_unc_critical_neg.yaml.doc.md) | Chỉ giữ negative quan trọng |
| [`P4_add_views.yaml`](P4_add_views.yaml.doc.md) | Thêm thông tin view có giúp không |
| [`P5_visual_primary.yaml`](P5_visual_primary.yaml.doc.md) | Ưu tiên tín hiệu thị giác hơn label |
| [`P6_qformer_visual_only.yaml`](P6_qformer_visual_only.yaml.doc.md) | ★ **Không label** — ablation sạch |
| [`P7_confidence_bins.yaml`](P7_confidence_bins.yaml.doc.md) | Đưa mức tin cậy vào prompt |
| [`P8_compact_normal.yaml`](P8_compact_normal.yaml.doc.md) | Tóm tắt ngắn khi bình thường |
| [`P9_full_negative_control.yaml`](P9_full_negative_control.yaml.doc.md) | Đưa toàn bộ negative |

## Consumer
```bash
python scripts/run_prompt_ablation.py --prompt-configs configs/prompt_ablation/P5_visual_primary.yaml
```
**Dry run** — không load model, không sinh metric model.

## Status
```text
🧪 ABLATION
```

## Developer notes
P6 là biến thể quan trọng nhất về mặt phương pháp: nó **không nhận Stage-1 label**,
nên so sánh với P5/P7 đo đúng đóng góp của label.

← [`_index.md`](../_index.md) · [HOME](../../../HOME.md)
