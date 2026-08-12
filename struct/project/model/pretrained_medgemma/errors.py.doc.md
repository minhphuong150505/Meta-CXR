> Source: `model/pretrained_medgemma/errors.py` (34 dòng)
> Status: ✅ ACTIVE
> Last verified against source: 2026-08-12

# `model/pretrained_medgemma/errors.py`

## Purpose
Bốn failure mode của checkpoint ngoài — và một nguyên tắc.

## ★ Nguyên tắc: raise, không hạ cấp
Docstring `:3`: *"Every error here is raised instead of degrading to a weaker
pipeline. A loader that cannot produce a multimodal model must not fall back to a
text-only one: the resulting reports would look plausible while never having seen
a pixel, which is the single most expensive failure this project can make."*

## Status
```text
✅ ACTIVE
```

## Main items
| Exception | Dòng |
|---|---|
| `PretrainedMedGemmaError` | 12 |
| `FindingsModelLoadError` | 16 |
| `NotMultimodalError` | 20 |
| `ImpressionPhaseDisabledError` | 28 |

## Calls / Called by
Được gọi: `findings_loader.py`, `impression_reporter.py`, `runner.py`.

## Side effects
Không.

## Developer notes
**Đừng bắt những exception này rồi tiếp tục.** Fail-closed là toàn bộ giá trị.

← [`_index.md`](_index.md) · [HOME](../../../HOME.md)
