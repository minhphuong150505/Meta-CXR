> Source: `model/pretrained_medgemma/output_schema.py` (74 dòng)
> Status: ✅ ACTIVE
> Last verified against source: 2026-08-12

# `model/pretrained_medgemma/output_schema.py`

## Purpose
Record prediction Phase 1 + hậu xử lý output.

## ★ Cờ provenance
Docstring `:3`: record **cố ý** mang `external_checkpoint` và
`fine_tuned_by_this_project` để một file kết quả **không thể** bị nhầm là output
của model project này train.

Và nó **không mang định danh MIMIC**: `sample_key` là digest **có salt**.

## Status
```text
✅ ACTIVE
```

## Main items
| Tên | Dòng |
|---|---|
| `FindingsPrediction` | 44 |
| `postprocess_findings(raw)` | 27 | → `(findings, warnings)` |

## Calls / Called by
Gọi: `training.dataio.manifest.split_generated_report` (`:15`).
Được gọi: `findings_reporter.py`; `medgemma_inference/prediction_writer.py` (gián tiếp).

## Side effects
Không.

## Related tests
`tests/test_pretrained_findings.py`

## Developer notes
Đừng bỏ cờ provenance khỏi record. Đó là ranh giới học thuật giữa "kết quả của
chúng tôi" và "kết quả của checkpoint người khác".

← [`_index.md`](_index.md) · [HOME](../../../HOME.md)
