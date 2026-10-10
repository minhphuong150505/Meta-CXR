# `scripts/probe_soft_tokens_cached.py`

> Thư mục: [`scripts/`](_index.md) · Thêm 2026-10-11 · 🔬 chẩn đoán

## Purpose
Linear probe 32 soft token Q-Former **từ cache record Stage 2 có sẵn**
(`.sensitive_stage1_cache`: `qformer_embs` [32, 768], `class_logits` [14, 3]),
so với chính MHCAC trên cùng các study test, cùng nhãn (D-018: trống = Negative,
-1 = Uncertain, study không có thông tin CheXpert bị bỏ). Khác
[`probe_soft_tokens.py`](probe_soft_tokens.py.doc.md): không encode lại, fit trên
train (mặc định 50.000 study), chọn L2 trên val, chấm test một lần.

## Entry point
```bash
python scripts/probe_soft_tokens_cached.py --cache-dir <run>/.sensitive_stage1_cache \
    --manifest-dir <split dir> --chexpert <mimic-cxr-2.0.0-chexpert.csv.gz> \
    [--test-cache <file>] [--train-limit 50000] --report <private>/report.json
```

## Làm gì
- Hai bộ đặc trưng: `pooled` (trung bình 32 token) và `tokens` (nối 32 token,
  24.576 chiều); 14 đầu softmax 3 lớp trên GPU, lưới L2 `1e-5..1e-2`.
- In cosine giữa các study và giữa 32 token trong một study (≈1 = suy biến),
  AUROC one-vs-rest theo (bệnh, lớp) như `auroc_mean`, đối chứng nhãn xáo trộn.
- Chỉ in số tổng hợp, không text/ID/đường dẫn. Cache là dữ liệu bệnh nhân:
  chạy trên máy train, report để ngoài repo.

## Kết quả đã đo (2026-10-11, `run_20261005_paper`)
MHCAC 0,745 / pooled 0,713 / tokens 0,653 / xáo trộn 0,561; cosine trong study
**+0,9995** (32 token gần như là một vector). Xem
`docs/handoff/PLAN-2026-10-11-soft-token-probe.md`.
