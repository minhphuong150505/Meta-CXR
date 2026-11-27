> Source: `runtime/` (3 file)
> Status: ✅ ACTIVE
> Last verified against source: 2026-08-12

# `runtime/`

## Purpose

Hai mối quan tâm vận hành, cả hai **stdlib-only**: ngân sách chi phí và chọn
device/dtype.

## Parent

[`struct/project/`](../../HOME.md#source-code-tree)

## Children

| File | Doc | Vai trò |
|---|---|---|
| `budget.py` | [📄](budget.py.doc.md) | `BudgetState`, `BudgetExceeded` |
| `device.py` | [📄](device.py.doc.md) | `DevicePlan`, `plan_device` |
| `__init__.py` | — | |

## `budget.py` — hai quyết định đáng chú ý

1. **Tính theo wall-clock, không theo throughput.** Một GPU bị treo tốn đúng bằng
   một GPU đang bận. Tính theo số mẫu đã xử lý sẽ bỏ sót đúng trường hợp tốn kém nhất.

2. **`prior_elapsed_seconds` khiến resume không reset trần chi phí.** Không có nó,
   một run bị restart nhiều lần có thể vượt ngân sách vô hạn mà mỗi lần đều "hợp lệ".

> **Nó chỉ ever dừng một run.** Không bao giờ tự hạ cấp model hay bật thêm section
> để "tiết kiệm". Giới hạn ngân sách không được phép âm thầm đổi thí nghiệm.

## `device.py`

Resolve device và dtype từ config hoặc từ máy. **Không chỗ nào hardcode `cuda:0`.**

(Ngoại lệ cũ `inference.py:312` `device_map={"": 0}` đã hết: file đã xóa 2026-11-27.)
— xem [I4](../_meta/LEGACY_AND_OPTIONAL.md#-potential-issues--ghi-nhận-không-sửa).

## Main responsibilities

1. Theo dõi và cưỡng chế trần chi phí.
2. Chọn device/dtype không hardcode.

## Entry points

Không có. Thư viện.

## Dependencies

Chỉ stdlib (`device.py` có thể chạm `torch` để dò GPU ⚠ cần runtime verification).

## Used by

⚠ **Không còn caller nào từ 2026-11-27.** Người dùng duy nhất là pipeline P8 (`medgemma_inference/`, `model/pretrained_medgemma/`, `tests/test_pretrained_findings.py`), đã xóa theo D-029. Thư mục được giữ lại; xóa hay không là quyết định riêng.

## Status

```text
✅ ACTIVE — nhưng chỉ trong phạm vi P8
```

## Notes

Nếu bạn thêm ngân sách cho Stage 1/Stage 2, hãy dùng lại `budget.py` thay vì viết
mới — đặc biệt là ngữ nghĩa `prior_elapsed_seconds`.

## Related documentation

`medgemma_inference/` (đã xóa 2026-11-27) ·
[ARCHITECTURE.md §5](../_meta/ARCHITECTURE.md#5-hai-khối-stdlib-only)

← [Về HOME](../../HOME.md)
