> Source: `training/evaluation/config.py` (238 dòng)
> Status: ❓ UNKNOWN — chỉ test import
> Last verified against source: 2026-08-12

# `config.py`

> ⚠⚠ **D-023 (2026-09-29) — ba lớp, không bao giờ nhị phân.** `CLASSIFICATION_METRICS = AGGREGATE_METRICS` (ba lớp); đã XÓA `uncertain_policy`, `threshold_mode`, `threshold_objective`, `include_meta_labels`; `selection_metric` mặc định `loss`. Xem [D-023](../../_meta/DECISIONS.md#d-023--ba-lớp-không-bao-giờ-nhị-phân-gỡ-toàn-bộ-khung-nhị-phân).


## Purpose
Schema và validate cho một khối config `evaluation:` — hình dung một luồng
evaluation điều khiển bằng config thay vì CLI.

## ⚠ Chưa được nối
Chỉ `tests/test_evaluation_config.py` import. `scripts/evaluate_stage1.py` và
`evaluate_stage2.py` nhận tham số qua `argparse`, không đọc khối này.
[D-001](../../_meta/DECISIONS.md#d-001--hạ-tầng-đã-viết-nhưng-chưa-nối-vào-pipeline).

## Status
```text
❓ UNKNOWN
```

## Main items
| Tên | Dòng |
|---|---|
| `EvaluationConfig` | 117 |
| `BootstrapConfig` | 98 |
| `composite_score(...)` | 220 |
| `EvaluationConfigError` | 93 |

Validate: `evaluation.bootstrap.samples >= 0` (`:107`),
`evaluation.bootstrap.confidence ∈ (0,1)` (`:111`),
`evaluation.clinical_metrics` chỉ nhận tên đã biết (`:184`),
key lạ → raise nêu tên (`:209`).

## Calls / Called by
Gọi: `evaluation.bootstrap`, `generation_metrics.LEXICAL_METRICS`,
`threshold_calibration`, `uncertain_policy`.
Được gọi: **chỉ** `tests/test_evaluation_config.py`.

## Side effects
Không.

## Related tests
`tests/test_evaluation_config.py`

## Developer notes
Nếu chuyển evaluation sang config-driven, **dùng file này** — nó đã validate kỹ và
có test. Nếu xác nhận không dùng, cập nhật D-001.

## Source relationships

- **Parent:** [`training/evaluation/`](_index.md)
- **Related:** [`schemas.py`](schemas.py.doc.md)

← [HOME](../../../HOME.md)
