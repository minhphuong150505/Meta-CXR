> Source: `training/evaluation/report_writer.py` (423 dòng)
> Status: ✅ ACTIVE
> Last verified against source: 2026-08-12

# `report_writer.py`

## Purpose
Xuất kết quả ra markdown + JSON + CSV, **kèm metadata tái lập**.

## Why it exists
Một bảng số không có commit hash, version package và device là số không tái lập
được. `ExperimentMetadata` gắn những thứ đó vào mọi báo cáo.

## Status
```text
✅ ACTIVE
```

## Main items
| Tên | Dòng | Vai trò |
|---|---|---|
| `build_markdown_report(...)` | 250 | ★ |
| `write_json` / `write_csv` / `write_jsonl` | 167, 174, 202 | |
| `ExperimentMetadata` | 133 | ★ commit, dirty flag, versions, device |
| `_git_commit()` / `_git_dirty()` | 70, 86 | ★ Ghi cả trạng thái **dirty** |
| `_package_versions()` | 100 | |
| `_device()` | 121 | |
| `_json_safe(value)` | 52 | `nan`/`inf` → JSON hợp lệ |
| `_metric_table` / `_format_interval` / `_fmt` | 241, 222, 212 | |
| `_classification_section` / `_generation_section` | 317, 375 | |
| `HEADLINE_CLASSIFICATION_METRICS` | — | Chỉ số đưa lên đầu |

## Calls / Called by
Gọi: `subprocess` (git), `json`, `csv`, `importlib.metadata`.
Được gọi: `scripts/evaluate_stage1.py:40,189`, `evaluate_stage2.py:46`.

## Side effects
Ghi file; chạy `git` subprocess.

## Related tests
`tests/test_evaluation_integration.py`

## Developer notes
`_git_dirty()` quan trọng: kết quả sinh từ working tree bẩn **không tái lập được**,
và báo cáo phải nói ra điều đó thay vì im lặng.

## Source relationships

- **Parent:** [`training/evaluation/`](_index.md)
- **Related:** [`schemas.py`](schemas.py.doc.md)

← [HOME](../../../HOME.md)
