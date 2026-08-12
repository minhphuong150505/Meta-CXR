> Source: `training/evaluation/uncertain_policy.py` (135 dòng)
> Status: ✅ ACTIVE
> Last verified against source: 2026-08-12

# `uncertain_policy.py`

## Purpose

Quyết định lớp **Uncertain** được xử lý thế nào khi chấm điểm.

## Why it exists

Báo cáo X-quang thật đầy câu kiểu "không loại trừ khả năng viêm phổi". Ép về nhị
phân là mất thông tin lâm sàng. Nhưng khi tính precision/recall nhị phân, phải
quyết định uncertain thuộc về đâu — và quyết định đó **thay đổi kết quả đáng kể**.

Module này làm lựa chọn đó **tường minh và có tên**, thay vì chôn trong một dòng code.

## Status

```text
✅ ACTIVE — production dùng `ignore_uncertain`
```

## Main items

| Tên | Dòng | Vai trò |
|---|---|---|
| `POLICIES` | — | Bảng policy |
| `DEFAULT_POLICY` | — | |
| `validate_policy(policy)` | 54 | Raise nếu lạ |
| `binarize_labels(...)` | 63 | ★ Nhãn 3 lớp → nhị phân theo policy |
| `describe_policy(policy)` | 113 | Mô tả cho báo cáo |
| `label_counts(labels)` | 127 | Đếm theo lớp |
| `UnknownPolicyError` | 50 | |

## Calls / Called by

Gọi: `numpy`, `evaluation.schemas` (`MISSING`, `POSITIVE`, `UNCERTAIN`).
Được gọi: `classification_metrics.py:38`, `threshold_calibration.py:41`,
`baselines.py:30`, `config.py:24`; `scripts/calibrate_thresholds.py:37`,
`evaluate_stage1.py:57`; `tests/test_classification_metrics.py:33`.

## Side effects

Không.

## Error / edge cases

Policy lạ → `UnknownPolicyError`.

## Related tests

`tests/test_classification_metrics.py`

## Developer notes

**Policy phải giống nhau giữa calibrate và evaluate.** Calibrate với
`ignore_uncertain` rồi chấm với policy khác cho kết quả không so sánh được.
`describe_policy()` nên xuất hiện trong mọi báo cáo để người đọc biết đã dùng gì.

## Source relationships

- **Parent:** [`training/evaluation/`](_index.md)
- **Related:** [`schemas.py`](schemas.py.doc.md) · [`scripts/_index.md`](../../scripts/_index.md)

← [HOME](../../../HOME.md)
