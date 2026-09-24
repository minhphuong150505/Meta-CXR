> Source: `scripts/count_chexpert_blank_policy.py`
> Status: ✅ CPU, read-only — không ghi gì ngoài `--json-out`
> Last verified against source: 2026-09-24 — chạy trên máy train với `full_allviews_v2`, exit 0

# `scripts/count_chexpert_blank_policy.py`

← [scripts/_index.md](_index.md) · [DECISIONS D-018](../_meta/DECISIONS.md#d-018--ô-chexpert-trống-là-âm-tính-đảo-ngược-quyết-định-2026-08-14)

## Purpose

Đếm nhãn Stage 1 ở **mức study** trên train/val/test dưới cả hai
`blank_label_policy`, bằng chính các hàm của
[`chexpert_labels.py`](../model/lavis/data/chexpert_labels.py.doc.md), và in
class weight cho `negative` theo công thức đang dùng trong `mimic_cxr_full.yaml`:
`[1.0, n_neg/n_pos, n_neg/n_unc]`, kappa 1, cap 10.

In ra theo từng split × policy: số ô `0/1/2/-100` mỗi nhãn, số ô trống thô, số
study `classification_valid`, và kiểm tra nguồn gốc ô `-100` dưới `negative`
(chỉ được đến từ study không có thông tin CheXpert + cột `excluded_labels`).
Kiểm tra đó fail → exit khác 0.

Chỉ in số đếm — không có ID hay text báo cáo, dán vào handoff được.

## Usage

```bash
python scripts/count_chexpert_blank_policy.py \
    --chexpert-csv <data>/mimic-cxr-2.0.0-chexpert.csv.gz \
    --split-dir <data>/processed/full_allviews_v2 \
    --cfg-path pretraining/configs/mimic_cxr_full.yaml \
    --json-out ~/blank_policy_counts.json
```

`excluded_labels` đọc từ `--cfg-path`.
