> Source: `model/pretrained_medgemma/findings_reporter.py` (116 dòng)
> Status: ✅ ACTIVE
> Last verified against source: 2026-08-12

# `model/pretrained_medgemma/findings_reporter.py`

## Purpose
Sinh FINDINGS. Nếu model tự thêm IMPRESSION, **bỏ đi và ghi cảnh báo**.

## Why
Docstring `:3`: Phase 1 không bao giờ giữ text impression, vì **không có đánh giá
impression nào được lên ngân sách hay đã chạy**. Giữ lại sẽ tạo ra dữ liệu mà
không ai chấm được nhưng dễ bị trích dẫn.

## Status
```text
✅ ACTIVE
```

## Main items
| Tên | Dòng |
|---|---|
| [`PretrainedFindingsReporter`](findings_reporter.py.methods/PretrainedFindingsReporter/_index.md) | 53 |
| `GenerationSettings` | 28 |
| `FindingsGeneration` | 45 |

## Calls / Called by
Gọi: `transformers` generate, `.output_schema.postprocess_findings`.
Được gọi: `medgemma_inference/runner.py`.

## Side effects
Chạy generate trên GPU.

## Related tests
`tests/test_pretrained_findings.py`

## Developer notes
`GenerationSettings` đi vào run identity — đổi nó làm resume bị từ chối, **có chủ đích**.

← [`_index.md`](_index.md) · [HOME](../../../HOME.md)
