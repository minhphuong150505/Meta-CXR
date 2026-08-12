> Source: `scripts/_stage2_fixtures.py` (72 dòng)
> Status: 🧪 — ⚠ KHÔNG phải MIMIC
> Last verified against source: 2026-08-12

# `scripts/_stage2_fixtures.py`

## Purpose
Record tổng hợp để các script prompt chạy được **không cần dữ liệu, không cần GPU**.

## ⚠ Cảnh báo — docstring `:1`
*"These are NOT MIMIC-CXR. They carry the same record shape the real pipeline emits
(`pred_groups`, views, prior flags, a `ref` target) so scripts are exercisable end
to end. Any numbers produced from them are illustrative, never model results."*

## Status
```text
🧪 — dữ liệu tổng hợp
```

## Main functions
`synthetic_records(n=50, seed=16)` (`:30`)

## Calls / Called by
Gọi: stdlib.
Được gọi: `scripts/run_prompt_ablation.py`, `export_stage2_prompt_samples.py`,
`prompt_length_statistics.py` (khi không có dữ liệu thật).

## Side effects
Không.

## Developer notes
Giữ **hình dạng record** khớp với `training/dataio/manifest.build_records`. Lệch đi
thì script chạy được với fixture nhưng vỡ với dữ liệu thật.

← [`_index.md`](_index.md) · [HOME](../../HOME.md)
