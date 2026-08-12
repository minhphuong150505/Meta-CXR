> Source: `training/evaluation/threshold_calibration.py` (345 dòng)
> Status: ✅ ACTIVE
> Last verified against source: 2026-08-12

# `threshold_calibration.py`

## Purpose

Tìm threshold tối ưu **cho từng bệnh lý**, và bảo vệ khỏi hai lỗi phổ biến:
calibrate trên test, và calibrate trên quá ít mẫu.

## Status

```text
✅ ACTIVE
```

## Main items

| Tên | Dòng | Vai trò |
|---|---|---|
| `calibrate_thresholds(...)` | 198 | [📄](threshold_calibration.py.methods/calibrate_thresholds.md) ★ Điểm vào |
| `calibrate_one(...)` | 156 | Một bệnh lý |
| `PathologyThreshold` | 68 | Kết quả một bệnh lý |
| `CalibrationResult` | 97 | Kết quả tổng |
| `_objective_score(...)` | 122 | Hàm mục tiêu (f1, …) |
| `_score_at(...)` | 299 | Điểm tại một threshold |
| `load_thresholds(path, *, allow_test_split=False)` | 316 | ★ **Chốt chặn** |
| `CalibrationError` | 63 | |

## Hai chốt chặn

### `allow_test_split=False` mặc định
`load_thresholds` **từ chối** file threshold được calibrate trên test, trừ khi
người gọi nói rõ. Calibrate trên test là rò rỉ test set — nó làm mọi con số sau đó
lạc quan giả.

### `--min-positive 20`
Bệnh lý dưới 20 mẫu positive **giữ nguyên threshold 0.5**. Tối ưu trên 3 mẫu
positive chỉ là overfit vào nhiễu.

## Calls / Called by

Gọi: `numpy`, `evaluation.schemas`, `evaluation.uncertain_policy`.
Được gọi: `scripts/calibrate_thresholds.py:31`, `evaluate_stage1.py:53`,
`config.py:20`; `tests/test_threshold_calibration.py:30`.

## Side effects

Ghi `thresholds.json` (qua script gọi nó).

## Error / edge cases

`CalibrationError` khi split sai · Bệnh lý thiếu positive → giữ 0.5, **ghi lại lý do**

## Related tests

`tests/test_threshold_calibration.py` (256 dòng)

## Developer notes

**Thứ tự bắt buộc:** calibrate trên validation → rồi mới evaluate trên test.
Không có bước nào ngược lại được coi là hợp lệ.

## Source relationships

- **Parent:** [`training/evaluation/`](_index.md)
- **Related:** [`schemas.py`](schemas.py.doc.md) · [`scripts/_index.md`](../../scripts/_index.md)

← [HOME](../../../HOME.md)
