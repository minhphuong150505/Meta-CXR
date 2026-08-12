> Source: `medgemma_inference/run_pretrained_findings.py:137-237`
> Status: ✅ ACTIVE

# `main(argv=None)`

## Purpose

CLI orchestration cho external Findings baseline: validate config/manifest, chọn
pilot hay full cohort, rồi gọi runner có resume/budget.

## Execution flow

```text
parse_args → load_config
  → resolve split CSV + assert_columns(FINDINGS_ONLY)
  → fingerprint schema/cohort
  → build_records(..., vis_root)
  → max_samples
       "all" mà thiếu --confirm-full-run → return 2
  → run_findings_inference(...)
  → in summary/cost → return 0
```

## Returns / errors

Trả exit code `0`; config/CSV/manifest/argument invalid trả `2`; Impression phase
bị chặn trả `3`. Không bắt lỗi load model/generation ngoài guard có chủ đích, nên
failure bất ngờ vẫn fail rõ.

## Side effects

Đọc CSV/ảnh, có thể tải checkpoint/cấp GPU, ghi prediction/progress/cost files.

## Tests / risk

`tests/test_pretrained_findings.py`. Đừng bỏ `--confirm-full-run`: nó ngăn chạy
toàn split trước khi xem pilot cost estimate.

← [`run_pretrained_findings.py`](../run_pretrained_findings.py.doc.md) · [HOME](../../../HOME.md)
