> Source: `scripts/evaluate_chexpert_crossdomain.py` (75 dòng)
> Status: ✅ ACTIVE — chưa từng chạy trên dữ liệu CheXpert thật
> Last verified against source: 2026-09-29

# `scripts/evaluate_chexpert_crossdomain.py`

← [scripts](_index.md) · [D-023](../_meta/DECISIONS.md#d-023--ba-lớp-không-bao-giờ-nhị-phân-gỡ-toàn-bộ-khung-nhị-phân)

Bảng 4 của bài báo: CheXpert val, 5 bệnh, `p_final = p1/(p0+p1)` (Eq. 21), AUC
và F1 tại ngưỡng 0,5. In kèm hàng của bài báo (`PAPER_TABLE4`, mean 0,824 / 0,699).

```bash
python scripts/evaluate_chexpert_crossdomain.py --predictions <chexpert_val.npz> \
    [--threshold 0.5] [--output t4.json]
```

- Đây là **một trong hai** tính toán nhị phân được phép, vì bài báo làm vậy.
  Lõi: [`paper_protocol.chexpert_crossdomain`](../training/evaluation/paper_protocol.py.doc.md).
- Nhãn phải là 0/1 (-1 = thiếu); gặp nhãn 2 → exit 2.
- ⚠ **Repo chưa có loader CheXpert val** để sinh `.npz` này. Script chỉ chấm điểm.
