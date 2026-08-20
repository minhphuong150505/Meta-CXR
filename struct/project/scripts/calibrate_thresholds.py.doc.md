> Source: `scripts/calibrate_thresholds.py` (141 dòng)
> Status: ✅ ACTIVE
> Last verified against source: 2026-08-20

# `scripts/calibrate_thresholds.py`

## Purpose
Calibrate threshold per-pathology — **chỉ trên validation**.

## Entry point
```bash
python scripts/calibrate_thresholds.py \
    --predictions <val.npz> --objective f1 \
    --uncertain-policy ignore_uncertain --min-positive 20 \
    --output <thresholds.json>
```

## Main functions
`parse_args(argv)` (`:42`) · `main(argv) -> int` (`:72`)

## Inputs / Outputs
Vào: `.npz` validation. Ra: `thresholds.json`.

## `--label-framing` / `--score` (2026-08-20)

Giống `evaluate_stage1.py`. Framing được áp **trước** khi calibrate, và được ghi
vào `metadata.source_metadata` của file JSON kết quả — đó chính là thứ
`evaluate_stage1.py` đọc lại để từ chối cặp không khớp.

⚠ Calibrate và evaluate **phải dùng cùng một cặp** `--label-framing` / `--score`.

## `--selection` / `--plateau-fraction` (2026-08-20)

Chuyển tiếp vào `calibrate_one`. Xem
[`threshold_calibration.py`](../training/evaluation/threshold_calibration.py.doc.md).

Cấu hình được đo là tốt nhất (chọn bằng CV trong val, không nhìn test):

```
--selection plateau --plateau-fraction 0.95 --min-positive 5
```

## Calls / Called by
Gọi: `training.evaluation.schemas` (`:30`), `.threshold_calibration` (`:31`),
`.uncertain_policy` (`:37`).
Được gọi: người dùng, sau Stage 1.

## Side effects
Ghi JSON. **Không GPU.**

## Related tests
`tests/test_threshold_calibration.py`

## Developer notes
⚠ **Chạy trên validation, không phải test.** `load_thresholds(allow_test_split=False)`
ở phía evaluate sẽ từ chối file calibrate trên test.
`--min-positive 20`: bệnh lý ít positive giữ 0.5 thay vì overfit.

← [`_index.md`](_index.md) · [HOME](../../HOME.md)
