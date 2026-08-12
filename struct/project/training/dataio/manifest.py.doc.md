> Source: `training/dataio/manifest.py` (255 dòng)
> Status: ✅ ACTIVE — ★ ranh giới độc lập
> Last verified against source: 2026-08-12

# `training/dataio/manifest.py`

## Purpose

Đọc split CSV thành record Stage 2 — **chỉ bằng pandas**. Cộng ba chốt chặn:
kiểm cột, kiểm leakage, và tách section.

## Why it exists

Nếu đường `medgemma_direct` phải đi qua `MIMIC_CXR_Dataset` (LAVIS), nó sẽ kéo
theo toàn bộ stack Stage 1 — và ablation không còn sạch. File này là lớp đọc dữ
liệu **thay thế**, không phụ thuộc gì ngoài pandas.

## Status

```text
✅ ACTIVE
```

## Main functions

| Hàm | Dòng | Vai trò |
|---|---|---|
| `build_records(...)` | 196 | ★ CSV → `list[dict]` |
| `assert_columns(frame, section_mode, source)` | 120 | ★ **Fail nêu tên cột thiếu** |
| `assert_no_leakage(frames)` | 153 | ★ Split disjoint theo patient VÀ study |
| `select_anchor_rows(...)` | 132 | Chọn anchor cho mỗi study |
| `row_target(row, section_mode)` | 99 | Target theo section |
| `format_report(findings, impression, section_mode)` | 57 | Ghép hai section |
| `split_generated_report(text)` | 76 | ★ Tách output → (findings, impression) |
| `deterministic_subset(records, limit, seed)` | 189 | Lấy mẫu con tái lập |
| `ManifestError` | 53 | Exception riêng |

`SECTION_MODES` = `findings_only`, `impression_only`, `findings_and_impression`
`DEFAULT_SECTION_MODE` = `findings_and_impression`

## Ba chốt chặn

### `assert_columns` — lỗi phải nói rõ thiếu gì
Manifest dựng trước 2026-07-21 thiếu `impression_clean` / `impression_valid` /
`impression_token_count`. Hàm này **nêu đúng tên cột**, không fail mơ hồ.

### `assert_no_leakage` — split phải disjoint
Kiểm cả `subject_id` và `study_id`. Một bệnh nhân xuất hiện ở cả train và test làm
mọi metric vô nghĩa, và không có gì khác trong pipeline phát hiện ra.

### `split_generated_report` — tách để chấm riêng
`fig9.evaluate_variant` gọi `compute_sectioned_nlg` tại `:1570`.
Với `findings_and_impression`, hàm này gọi `split_generated_report` cho prediction
và reference, rồi ghi metric full-report, từng section và omission rate.

## Calls / Called by

Gọi: `pandas`, `training.stage2_utils.stable_fingerprint` (`:21`, dual-import shim).
Được gọi: `run_medgemma_qlora.py:27` · `fig9:116,122` ·
`medgemma_inference/run_pretrained_findings.py:33` ·
`model/pretrained_medgemma/output_schema.py:15` ·
`dataio/validate_manifest.py:23,33` · `tests/test_manifest.py`, `test_section_metrics.py`

## Side effects

Đọc CSV vào RAM. Không ghi.

## Error / edge cases

`ManifestError` / raise từ `assert_columns` (nêu tên cột), `assert_no_leakage`
(nêu ID trùng ⚠ **cẩn thận: đừng log ID ra nơi công khai**), `section_mode` lạ.

## Related tests

`tests/test_manifest.py` (185) · `tests/test_section_metrics.py`

## Developer notes

1. **Đừng import LAVIS hay torch vào file này.** Toàn bộ lý do nó tồn tại là để
   không có import đó. `tests/test_native_independence.py` canh.
2. Mang dual-import shim.
3. FINDINGS và IMPRESSION có giới hạn độ dài **riêng**, dẫn xuất từ train split, và
   không bao giờ thay cho nhau.

← [`dataio/`](_index.md) · [HOME](../../../HOME.md)
