> Source: `training/run_medgemma_qlora.py:364-403`
> Status: ✅ ACTIVE

# `build_native_records(...)`

## Located in

[`run_medgemma_qlora.py`](../run_medgemma_qlora.py.doc.md)

## Purpose
Dựng record cho đường `medgemma_direct` — **không đụng LAVIS**.

## ★ Đây là một nửa của ranh giới độc lập
Hàm này chỉ dùng `training/dataio/manifest.py` (pandas thuần). Hàm chị em
`build_stage1_records` mới gọi `training/stage1/lavis_loader`.

`tests/test_native_independence.py:181` kiểm: *"The medgemma_direct data path reads
split CSVs, not ReportDataset."*

## Execution flow
```text
load_split_frame(split, cache_dir)      → pandas DataFrame
   ↓
assert_columns(frame, section_mode, source)   → nêu cột thiếu
assert_no_leakage({train, val, test})         → nêu trùng
   ↓
build_records(frame, section_mode)
   ↓
deterministic_subset(records, limit, seed)    ← khi có --train/val/test-limit
```

## Returns
`list[dict]` — `{image_path, ref, views, pred_groups=None, prior_available, …}`

⚠ `pred_groups=None`: đường native **không có Stage-1 label**. Đó là điều khiến
ablation sạch.

## Side effects
Đọc CSV vào RAM.

## Error handling
`assert_columns` → nêu tên cột thiếu (đặc biệt cột impression với manifest cũ) ·
`assert_no_leakage` → raise

## Tests
`tests/test_manifest.py` · `tests/test_native_independence.py`

## Modification risk
⚠ Thêm bất kỳ import LAVIS/torch nào vào đường này sẽ làm test fail và phá tính
độc lập của ablation.
