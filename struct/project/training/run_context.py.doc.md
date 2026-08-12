> Source: `training/run_context.py` (81 dòng)
> Status: ✅ ACTIVE
> Last verified against source: 2026-08-12

# `training/run_context.py`

## Purpose

`Stage1Context` — gói mọi thứ cần để định vị và tái tạo một Stage-1 run: tên run,
đường config override, đường checkpoint override và threshold.

## Why it exists

Trước đây các giá trị này là **global mutable** trong `fig9`. Code hiện tại đã
hoàn tất migration: `run_medgemma_qlora.main()` tạo một context tại `:418-423`
và truyền tường minh vào các hàm build/evaluate. `fingerprint_payload()` đưa đúng
identity đó vào fingerprint artifact.

## Status

```text
✅ ACTIVE
```

## Main class

`Stage1Context` (`:26`) — dataclass

| Method | Dòng | Vai trò |
|---|---|---|
| `__post_init__` | 38 | Validate + đóng băng mapping threshold |
| `resolve_config_path(default)` | 57 | Đường config, có override |
| `resolve_checkpoint_path(root)` | 61 | `<root>/<run>/checkpoint_best.pth` |
| `threshold_for(abnormality)` | 67 | Threshold một bệnh lý |
| `fingerprint_payload()` | 70 | ★ Vân tay ghi vào metadata |

## Calls / Called by

Gọi: stdlib.
Được gọi: `run_medgemma_qlora.py:42`; `fig9:101,112` (dual-import);
`tests/test_run_context.py`.

## Side effects

Không.

## Related tests

`tests/test_run_context.py`

## Developer notes

`fingerprint_payload()` là thứ cho phép truy vết một kết quả về đúng Stage-1 run
sinh ra nó. `thresholds` phải tiếp tục được copy vào mapping chỉ đọc; nếu giữ lại
dict từ caller, frozen dataclass vẫn có thể bị mutate gián tiếp.

← [training/](_index.md) · [HOME](../../HOME.md)
