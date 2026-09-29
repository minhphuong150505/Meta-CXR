> Source: `scripts/evaluate_clinical_efficacy.py` (76 dòng)
> Status: ✅ ACTIVE — cần nhãn labeler có sẵn
> Last verified against source: 2026-09-29

# `scripts/evaluate_clinical_efficacy.py`

← [scripts](_index.md) · [D-023](../_meta/DECISIONS.md#d-023--ba-lớp-không-bao-giờ-nhị-phân-gỡ-toàn-bộ-khung-nhị-phân)

Bảng 3 CE của bài báo: P/R/F1 (macro + micro) giữa nhãn CheXpert của báo cáo
sinh ra và báo cáo gốc. Bài báo: P 0,411 / R 0,465 / macro F1 0,428 (`PAPER_CE`).

```bash
python scripts/evaluate_clinical_efficacy.py --generated-labels gen.csv \
    --reference-labels ref.csv [--output ce.json]
```

- Hai CSV có `sample_key` + 14 cột CheXpert; join theo `sample_key`.
- Chỉ giá trị labeler `1` là dương (Uncertain không tính là dương). Bệnh không
  xuất hiện ở đâu cả là `nan` và bị loại khỏi macro, không phải 0.
- ⚠ **Repo không có labeler** (CheXbert/CheXpert-labeler); phải chạy bên ngoài.
  Lõi: [`paper_protocol.clinical_efficacy`](../training/evaluation/paper_protocol.py.doc.md).
