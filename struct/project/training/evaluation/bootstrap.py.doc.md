> Source: `training/evaluation/bootstrap.py` (221 dòng)
> Status: ✅ ACTIVE
> Last verified against source: 2026-08-12

# `bootstrap.py`

## Purpose
Khoảng tin cậy bằng bootstrap resampling.

## Why it exists
Một điểm F1 đơn lẻ trên 5.082 mẫu test không cho biết độ chắc chắn. Khoảng tin cậy
cho biết khác biệt giữa hai model có ý nghĩa hay chỉ là nhiễu.

## Status
```text
✅ ACTIVE
```

## Main items
| Tên | Dòng | Vai trò |
|---|---|---|
| `bootstrap_metric(...)` | 76 | ★ CI cho một chỉ số |
| `bootstrap_many(...)` | 168 | Nhiều chỉ số cùng lúc |
| `bootstrap_sample_metric(...)` | 195 | CI theo per-sample metric |
| `ConfidenceInterval` | 36 | Dataclass kết quả |
| `_clean(value)` | 68 | ★ Trả `None` thay vì `nan` |
| `BootstrapError` | 31 | |

## Calls / Called by
Gọi: `numpy`.
Được gọi: `scripts/evaluate_stage1.py:33`, `evaluate_stage2.py:34`; `config.py:14`;
`tests/test_threshold_calibration.py:22`.

## Side effects
Không. Có seed để tái lập.

## Error / edge cases
`BootstrapError` khi samples < 1 hoặc confidence ngoài (0,1). `_clean` biến `nan`
thành `None` để JSON không mang `NaN` (không hợp lệ trong JSON chuẩn).

## Related tests
`tests/test_threshold_calibration.py`

## Developer notes
`--bootstrap-samples 1000` là con số thường dùng; nhiều hơn tốn thời gian mà CI
gần như không đổi.

## Source relationships

- **Parent:** [`training/evaluation/`](_index.md)
- **Related:** [`schemas.py`](schemas.py.doc.md)

← [HOME](../../../HOME.md)
