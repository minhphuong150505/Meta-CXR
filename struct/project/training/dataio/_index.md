> Source: `training/dataio/`
> Status: ✅ ACTIVE
> Last verified against source: 2026-08-12

# `training/dataio/`

## Purpose

Đọc split CSV thành record cho Stage 2 — **chỉ bằng pandas**, không import LAVIS,
không import torch model. Đây là điều kiện để `medgemma_direct` thực sự độc lập
với Stage 1.

## Role in project

```text
train/val/test.csv ──► manifest.build_records() ──► list[dict] ──► PromptBuilder ──► MedGemma
```

## Parent

[`training/`](../_index.md)

## Children

| File | LOC | Doc | Vai trò |
|---|---|---|---|
| `manifest.py` | 255 | [📄](manifest.py.doc.md) | `build_records`, `assert_no_leakage`, `assert_columns`, `split_generated_report`, `SECTION_MODES` |
| `validate_manifest.py` | — | [📄](validate_manifest.py.doc.md) | CLI kiểm tra invariant |

## Main responsibilities

1. **Dựng record** từ CSV theo `--section-mode`.
2. **Chặn leakage** — `assert_no_leakage` kiểm tra split disjoint theo patient/study.
3. **Kiểm tra cột** — `assert_columns` fail và **nêu tên cột thiếu**, không fail mơ hồ.
4. **Tách section** — `split_generated_report` chia output thành FINDINGS / IMPRESSION.

## Entry points

```bash
python -m training.dataio.validate_manifest --section-mode findings_and_impression
```

## Dependencies

`pandas` · `training/stage2_utils.stable_fingerprint` (qua dual-import shim).
**Không gì khác.** Đó là điểm mấu chốt.

## Used by

`training/run_medgemma_qlora.py:27` · `train_eval_figure9…:116/122` ·
`medgemma_inference/run_pretrained_findings.py:33` ·
`model/pretrained_medgemma/output_schema.py:15` ·
`tests/test_manifest.py`, `tests/test_section_metrics.py`

## Execution flow

```text
build_records(split, section_mode)
   ↓
pd.read_csv  →  assert_columns()  →  assert_no_leakage()
   ↓
lọc theo section_mode (findings_only | impression_only | findings_and_impression)
   ↓
list[dict]  {image_path, ref, views, pred_groups=None, prior_available, …}
```

## Important configurations

| Key | Giá trị |
|---|---|
| `SECTION_MODES` | `findings_only`, `impression_only`, `findings_and_impression` |
| `DEFAULT_SECTION_MODE` | `findings_and_impression` |
| Cột bắt buộc | `image_path`, `findings`, `target_valid`, và với impression: `impression_clean`, `impression_valid`, `impression_token_count` |

## Status

```text
✅ ACTIVE
```

## Notes

- ⚠ **Manifest dựng trước 2026-07-21 thiếu cột impression** và **không phục vụ
  được** `--section-mode findings_and_impression` (mặc định). `assert_columns`
  fail nêu đúng tên cột. Phải chạy lại `preporcessing/preprocess_mimic_cxr.py`.

- `split_generated_report` đã được nối vào production qua
  `fig9.compute_sectioned_nlg` → `evaluate_variant:1570`. Với target hai section,
  output có cả metric full-report và metric FINDINGS/IMPRESSION riêng.

- **FINDINGS và IMPRESSION có giới hạn độ dài riêng**, dẫn xuất từ train split, và
  **không bao giờ thay thế cho nhau**.

## Related documentation

[DATA_FLOW.md §4](../../_meta/DATA_FLOW.md#4-stage-2--csv--tensor) ·
[`preporcessing/_index.md`](../../preporcessing/_index.md) ·
[`tests/_index.md`](../../tests/_index.md)

← [`training/`](../_index.md) · [HOME](../../../HOME.md)
