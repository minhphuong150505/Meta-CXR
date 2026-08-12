> Source: `runtime/device.py` (105 dòng)
> Status: ✅ ACTIVE — chỉ P8 dùng
> Last verified against source: 2026-08-12

# `runtime/device.py`

## Purpose
Resolve device và dtype từ config hoặc từ máy. **Không hardcode `cuda:0`.**

## Status
```text
✅ ACTIVE — chỉ P8
```

## Main items
| Tên | Dòng | Vai trò |
|---|---|---|
| `plan_device(...)` | 94 | ★ Điểm vào |
| `DevicePlan` | 18 | Kết quả: device + dtype |
| `resolve_device(requested="auto")` | 37 | |
| `resolve_dtype(device, requested="auto")` | 60 | |
| `check_4bit_available(device)` | 80 | ★ Kiểm bitsandbytes dùng được |
| `_torch()` | 30 | ★ Import lazy |
| `QuantizationUnavailable` | 13 | |

`_torch()` import lazy → module này import được ở môi trường không có torch.

`check_4bit_available` raise **trước** khi tải checkpoint, thay vì để lỗi nổ giữa
chừng sau khi đã tải 4 GB.

## Calls / Called by
Gọi: `torch` (lazy).
Được gọi: `model/pretrained_medgemma/findings_loader.py:24`.

## Side effects
Không.

## Error / edge cases
4-bit không dùng được → `QuantizationUnavailable`.

## Related tests
`tests/test_pretrained_findings.py`

## Developer notes
⚠ Chỗ duy nhất còn hardcode GPU 0 trong repo là `inference.py:312`
([I4](../_meta/LEGACY_AND_OPTIONAL.md#-potential-issues--ghi-nhận-không-sửa)) — nó
không dùng module này.

## Source relationships

- **Parent:** [`_index.md`](_index.md)

← [HOME](../../HOME.md)
