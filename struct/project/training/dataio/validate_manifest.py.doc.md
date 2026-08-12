> Source: `training/dataio/validate_manifest.py` (159 dòng)
> Status: 🧰 UTILITY — CLI
> Last verified against source: 2026-08-12

# `training/dataio/validate_manifest.py`

## Purpose

CLI kiểm tra bất biến của split manifest **trước** khi tốn GPU-hour: cột bắt buộc,
leakage giữa split, và section target.

## Status

```text
🧰 UTILITY — ACTIVE
```

## Entry point

```bash
python -m training.dataio.validate_manifest --section-mode findings_and_impression
```

Trả `int` exit code (`main() -> int`).

## Main functions

| Hàm | Dòng |
|---|---|
| `parse_args()` | 44 |
| `resolve_paths(args)` | 66 |
| `main()` | 84 |

## Execution flow

```text
parse_args → resolve_paths (từ configs/env_config.yaml)
   ↓
đọc train/val/test CSV bằng pandas
   ↓
assert_columns(mỗi frame, section_mode)     → nêu cột thiếu
assert_no_leakage({train, val, test})       → nêu trùng
   ↓
exit 0 hoặc != 0
```

## Calls / Called by

Gọi: `training.dataio.manifest` (`:23`/`:33` dual-import), `pandas`, `local_config`.
Được gọi: người dùng; nên chạy sau `preporcessing/preprocess_mimic_cxr.py`.

## Side effects

Chỉ đọc. Không ghi, không GPU.

## Error / edge cases

Manifest cũ thiếu cột impression → fail nêu tên cột. Đây là **cách nhanh nhất**
phát hiện manifest lỗi thời.

## Related tests

`tests/test_manifest.py`

## Developer notes

Chạy nó **mỗi lần** dựng lại split. Nó rẻ và bắt được lỗi mà training sẽ chỉ phát
hiện sau nhiều giờ.

← [`dataio/`](_index.md) · [HOME](../../../HOME.md)
