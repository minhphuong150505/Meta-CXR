> Source: `medgemma_inference/runner.py:128-240`
> Status: ✅ ACTIVE

# `run_findings_inference(...)`

## Purpose

Sinh FINDINGS có resume, privacy-safe writer và budget wall-clock; trả summary
aggregate, không trả report text.

## Execution flow

```text
assert_impression_disabled()                    ← trước model/download
  → build RunIdentity + ProgressFile.open()     ← mismatch thì dừng
  → PredictionWriter → lọc sample_key đã xong
  → BudgetState(prior_elapsed_seconds)
  → không còn pending? ghi finished, return      ← không load model
  → lazy _build_reporter()
  → mỗi record:
       assert_within_budget trước sample
       load_image → reporter.generate
       FindingsPrediction.to_dict → writer.write + fsync
       định kỳ progress.write
  → progress cuối + cost estimate → RunSummary
```

## Important contracts

`records` phải có `sample_key`, `image_path`; `dataset_fingerprint` phải ổn định.
Budget check nằm giữa sample để không ghi nửa prediction. `reporter_factory`,
`image_loader`, `clock` được inject cho test CPU.

## Tests / risk

`tests/test_pretrained_findings.py`, `test_inference_only_invariants.py`.
Không thêm optimizer, `.train()` hoặc gradient vào inference-only path.

← [`runner.py`](../runner.py.doc.md) · [HOME](../../../HOME.md)
