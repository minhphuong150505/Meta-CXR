> Source: `model/pretrained_medgemma/impression_reporter.py` (58 dòng)
> Status: ⛔ DISABLED có chủ đích
> Last verified against source: 2026-08-12

# `model/pretrained_medgemma/impression_reporter.py`

## ⛔ Module này cố ý TRƠ

Docstring `:3`: *"This module is intentionally inert. Importing it must not
download a checkpoint, construct a processor, allocate VRAM, or import
transformers. It exists so the Phase-2 interface is settled and reviewable, not so
it can run."*

Phase 2 (Impression) **chưa được duyệt ngân sách**. Interface được chốt trước để
review được, nhưng không có đường chạy.

## Status
```text
⛔ DISABLED
```

## Main items
| Tên | Dòng | Vai trò |
|---|---|---|
| `assert_impression_disabled(...)` | 26 | ★ Guard — chạy **trước mọi thứ** trong runner |
| `PretrainedImpressionReporter` | 42 | Interface, không có implementation chạy được |

`ImpressionPhaseDisabledError` ở `errors.py:28`.

## Calls / Called by
Gọi: `.errors`. **Không** import transformers.
Được gọi: `medgemma_inference/runner.py` — guard chạy đầu tiên.

## Side effects
Không. Đó là điểm mấu chốt.

## Related tests
`tests/test_pretrained_findings.py` — kiểm import không tải gì

## Developer notes
**Đừng "bật" module này** mà không có quyết định ngân sách rõ ràng. Guard tồn tại
để chi phí không tăng gấp đôi một cách âm thầm.

← [`_index.md`](_index.md) · [HOME](../../../HOME.md)
