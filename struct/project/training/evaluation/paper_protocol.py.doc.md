> Source: `training/evaluation/paper_protocol.py` (139 dòng)
> Status: ✅ ACTIVE — thêm 2026-09-29 (D-023)
> Last verified against source: 2026-09-29

# `paper_protocol.py`

## Purpose

**Hai chỗ duy nhất bài báo tự quy về nhị phân**, và là chỗ duy nhất trong repo
được phép tính nhị phân:

| Hàm | Dòng | Bảng | Cách tính |
|---|---|---|---|
| `chexpert_crossdomain(probs, labels01, names, threshold=0.5)` | 65 | Bảng 4 | `p_final = p1/(p0+p1)` (Eq. 21), AUC + F1 trên 5 bệnh; ngưỡng F1 bài báo không nêu → mặc định 0.5, được ghi lại |
| `clinical_efficacy(gen_labels, ref_labels)` | 108 | Bảng 3 (CE) | nhãn labeler = 1 là "có"; P/R/F1 từng bệnh, macro và micro trên 14 bệnh |

⚠ Không có loader CheXpert val cho Stage 1 và không có labeler (CheXpert/CheXbert)
trong repo — module chỉ chấm điểm dữ liệu đã có.

## Callers

`scripts/evaluate_chexpert_crossdomain.py`, `scripts/evaluate_clinical_efficacy.py`.
Test: `tests/test_paper_protocol.py`.
