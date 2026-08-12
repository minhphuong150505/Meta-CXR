> Source: `training/evaluation/threshold_calibration.py:198-296`
> Status: ✅ ACTIVE

# `calibrate_thresholds(predictions, *, split, ...)`

## Purpose

Fit một threshold positive cho từng pathology trên validation-like split, kèm
metadata/provenance và fallback 0.5 cho class không đủ support.

## Fail-closed behavior

`split` ngoài `CALIBRATION_SPLITS`—đặc biệt test—raise `CalibrationError`.
Objective/policy lạ cũng raise. Nếu positive `< min_positive` hoặc không có
negative, pathology được đánh dấu `calibrated=False`, score `nan`, reason rõ và
threshold mặc định; không giả metric 0.

## Returns

`CalibrationResult` gồm list `PathologyThreshold` và metadata: split, objective,
policy, constraint, sample count, created UTC, source metadata.

## Tests

`tests/test_threshold_calibration.py`.

← [`threshold_calibration.py`](../threshold_calibration.py.doc.md) · [HOME](../../../../HOME.md)
