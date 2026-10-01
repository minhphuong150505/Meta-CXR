> Source: `scripts/evaluate_stage1.py` (236 dòng)
> Status: ✅ ACTIVE
> Last verified against source: 2026-09-29

# `scripts/evaluate_stage1.py`

← [scripts](_index.md) · [D-023](../_meta/DECISIONS.md#d-023--ba-lớp-không-bao-giờ-nhị-phân-gỡ-toàn-bộ-khung-nhị-phân)

## Purpose
Chấm phân loại Stage 1 từ `.npz` theo **giao thức ba lớp của bài báo** — không
model, không GPU, không dataset.

## Entry point
```bash
python scripts/evaluate_stage1.py --predictions <test.npz> --output-dir <dir> \
    [--no-bootstrap] [--no-plots] [--no-baselines] [--split test] \
    [--thresholds <val_thresholds_eq22.json>]
```

## Làm gì
- Quyết định = **argmax** trên softmax 3 lớp (Negative / Positive / Uncertain).
  Không có file ngưỡng, không có framing, không có policy uncertain.
- Gọi [`evaluate_classification`](../training/evaluation/classification_metrics.py.doc.md):
  mỗi bệnh sklearn `average='weighted'`, `zero_division=1`, rồi trung bình đều
  14 bệnh → so với Fig 10 của bài báo (0,87 / 0,78 / 0,73).
- `mean_weighted_f1_5` (5 bệnh Bảng 5/7, bài báo 0,701); AUROC one-vs-rest mỗi lớp (Fig 5).
- Bootstrap CI theo study, baseline ba lớp, bảng từng bệnh, và `plots/{cls}_vs_rest`.
- `--thresholds` (thêm 2026-10-01): **phân tích bổ sung, không phải giao thức của bài
  báo.** Thay argmax bằng ngưỡng Eq. 22 theo (bệnh, lớp) fit trên val
  (`decide_with_thresholds`: lớp vượt ngưỡng xa nhất thắng, không lớp nào vượt thì
  dùng argmax). Từ chối file fit trên chính split đang chấm hoặc trên test. AUROC
  không đổi. Khác hẳn cờ `--thresholds` nhị phân đã gỡ ở D-023: `load_thresholds`
  từ chối định dạng cũ.

## Đã gỡ (D-023)
`--thresholds` nhị phân (cờ cùng tên hiện tại là ngưỡng ba lớp, xem trên), `--uncertain-policy`, `--label-framing`, `--score`,
`positive_macro_f1`. Các số cũ trong CLAUDE.md dùng chúng; chấm lại từ `.npz` cũ
bằng lệnh trên để so với bài báo.

## Khác biệt còn lại với code tham chiếu
Code gốc (`META-CXR/mhcac/utils.py compute_metrics_for_tasks`) tính trung bình
**theo batch**; ở đây tính trên toàn split. Ghi rõ khi so sánh.
