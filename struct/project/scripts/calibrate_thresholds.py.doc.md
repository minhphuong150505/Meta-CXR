> Source: `scripts/calibrate_thresholds.py` (72 dòng)
> Status: ✅ ACTIVE
> Last verified against source: 2026-10-10

# `scripts/calibrate_thresholds.py`

← [scripts](_index.md) · [D-023](../_meta/DECISIONS.md#d-023--ba-lớp-không-bao-giờ-nhị-phân-gỡ-toàn-bộ-khung-nhị-phân)

Fit ngưỡng theo **(bệnh, lớp)** bằng Eq. 22 / Fig 11 của bài báo: điểm ROC gần
(0,1) nhất, `argmin sqrt((1-TPR)^2 + FPR^2)`, one-vs-rest trên xác suất từng lớp.

```bash
python scripts/calibrate_thresholds.py --predictions <val.npz> --output <thresholds.json>
```

- **Chỉ validation**: `.npz` có metadata split `test` → exit 2.
- File ghi format `meta_cxr_per_class_roc_distance_v1`; mọi format khác (file
  ngưỡng nhị phân cũ) bị [`load_thresholds`](../training/evaluation/threshold_calibration.py.doc.md) từ chối.
- Ngưỡng Eq. 22 chỉ dùng cho **prompt Stage 2** (`--cue-rule paper_thresholds`).

## `--rule cutpoints` (2026-10-10) — luật cho số chính của dự án

```bash
python scripts/calibrate_thresholds.py --rule cutpoints [--one-cutpoint] \
    --predictions <val.npz> --checkpoint <tên> --output configs/stage1_cutpoints/<run>.json
```

Mỗi bệnh: hai ngưỡng `t1 <= t2` trên `p_pos/(p_pos+p_neg)` tối đa weighted F1 ba
lớp của bệnh đó (`fit_severity_cutpoints`): N dưới t1, U trong [t1, t2), P từ t2.
`--one-cutpoint` ép t1 == t2 (không bao giờ U). Ghi `CutpointFile`
(`meta_cxr_severity_cutpoints_v1`). Chấm bằng
[`evaluate_stage1.py --cutpoints`](evaluate_stage1.py.doc.md). Chỉ validation.

Đã gỡ (D-023): `--objective f1`, `--selection plateau`, `--min-positive`,
`--uncertain-policy`, `--label-framing`, `--score`.
