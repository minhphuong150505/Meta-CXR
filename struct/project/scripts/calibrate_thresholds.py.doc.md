> Source: `scripts/calibrate_thresholds.py` (141 dòng)
> Status: ✅ ACTIVE
> Last verified against source: 2026-08-12

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
