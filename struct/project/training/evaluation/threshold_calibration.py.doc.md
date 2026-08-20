> Source: `training/evaluation/threshold_calibration.py` (401 dòng)
> Status: ✅ ACTIVE
> Last verified against source: 2026-08-20

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

## `--selection plateau` (2026-08-20)

Chọn *đứng ở đâu* trên đường cong mục tiêu — câu hỏi tách biệt với *tối ưu mục tiêu nào*.

| | |
|---|---|
| `argmax` | đỉnh chính xác. Mặc định, giữ nguyên hành vi cũ |
| `plateau` | trung vị của mọi ngưỡng đạt ≥ `plateau_fraction` × đỉnh |

Trên val 1,808 study, đỉnh đường cong F1 là một điểm may mắn phụ thuộc study nào rơi
vào val; tâm vùng gần tối ưu chuyển giao sang test tốt hơn.

**Đo bằng CV 5-fold × 10 lần *bên trong* val** (`study_presence` + `marginal_presence`):
plateau@0.95 đạt **0.3246** macro F1 so với argmax **0.3202**. Trên test held-out, cùng
lựa chọn đó cho **0.3397 so với 0.3224** — bootstrap ghép cặp 2,000 lần:
ΔF1 **+0.0174** [+0.0102, +0.0243], Δrecall **+0.0747**, Δprecision **−0.0211**.
Cả ba đều có ý nghĩa; đây là **đánh đổi thật** nghiêng 3.5:1 về recall.

⚠ Kết hợp với `--min-positive 5`. Ở mặc định cũ (20), `Pleural Other` (15 dương trên val)
và `Fracture` (18) rơi về ngưỡng 0.5 — `Fracture` **không bao giờ được dự đoán dương**,
F1 = 0.0000. Hai nhãn đó chiếm **59%** toàn bộ khoảng cách tới trần oracle. Khi được
calibrate thật, chúng ra 0.231 / 0.244, sát ngưỡng oracle trên test (0.250 / 0.247).

`selection` và `plateau_fraction` được ghi vào `metadata` của file ngưỡng.

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
