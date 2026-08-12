> Source: `safety/verifiers.py` (252 dòng)
> Status: ❓ UNKNOWN — chỉ pipeline + test
> Last verified against source: 2026-08-12

# `safety/verifiers.py`

## Purpose
Protocol và implementation cho việc verify claim.

## Status
```text
❓ UNKNOWN
```

## Main items
| Tên | Dòng | Vai trò |
|---|---|---|
| `VerificationResult` | 42 | |
| `AbnormalityVerifier` (Protocol) | 60 | |
| `PhraseGroundingVerifier` (Protocol) | 65 | ★ Chỗ cắm model thật |
| `MeasurementChecker` (Protocol) | 70 | |
| `ClassifierAbnormalityVerifier` | 75 | ★ Dùng logits MHCAC verify claim |
| `UnavailablePhraseGrounding` | 131 | ★ **Báo unavailable, không giả vờ verify** |
| `RegexMeasurementChecker` | 150 | Kiểm số đo trong text |
| `UncertaintyEstimator` (Protocol) | 212 | |
| `EntropyUncertaintyEstimator` | 217 | Entropy của phân phối lớp |

## ★ `UnavailablePhraseGrounding`
Chưa có model phrase-grounding → trả **"unavailable"**, không trả "verified" cũng
không trả "failed". Cùng nguyên tắc với `training/evaluation/clinical.py`: thiếu
công cụ đo thì nói là thiếu, không bịa kết quả.

## `ClassifierAbnormalityVerifier`
Dùng chính output MHCAC làm nguồn verify độc lập cho claim trong text. Nếu báo cáo
nói "pneumonia" mà classifier nói negative với độ tin cậy cao → cờ đỏ.

## Calls / Called by
Gọi: `safety.claims` (`:25`), `numpy`/`math`.
Được gọi: `safety/pipeline.py:30`, `safety/reconciler.py:41`; `tests/test_safety_pipeline.py:25`.

## Side effects
Không.

## Related tests
`tests/test_safety_pipeline.py`

## Developer notes
Cắm model phrase-grounding thật = implement `PhraseGroundingVerifier` và thay
`UnavailablePhraseGrounding`. Không cần đổi `pipeline.py`.

## Source relationships

- **Parent:** [`_index.md`](_index.md)

← [HOME](../../HOME.md)
