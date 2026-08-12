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

⚠ Ngoại lệ duy nhất còn lại trong repo: `inference.py:312` (`device_map={"": 0}`)
— xem [I4](../_meta/LEGACY_AND_OPTIONAL.md#-potential-issues--ghi-nhận-không-sửa).

## Main responsibilities

1. Theo dõi và cưỡng chế trần chi phí.
2. Chọn device/dtype không hardcode.

## Entry points

Không có. Thư viện.

## Dependencies

Chỉ stdlib (`device.py` có thể chạm `torch` để dò GPU ⚠ cần runtime verification).

## Used by

`medgemma_inference/runner.py:23` → `budget.BudgetExceeded`, `BudgetState`
`medgemma_inference/config.py:230`, `run_pretrained_findings.py:122` → `config.runtime.*`
`model/pretrained_medgemma/findings_loader.py:24` → `device.DevicePlan`, `plan_device`
`tests/test_pretrained_findings.py:39`

⚠ **Chỉ pipeline P8 dùng thư mục này.** Stage 1 và Stage 2 không dùng.

## Status

```text
✅ ACTIVE — nhưng chỉ trong phạm vi P8
```

## Notes

Nếu bạn thêm ngân sách cho Stage 1/Stage 2, hãy dùng lại `budget.py` thay vì viết
mới — đặc biệt là ngữ nghĩa `prior_elapsed_seconds`.

## Related documentation

[`medgemma_inference/_index.md`](../medgemma_inference/_index.md) ·
[ARCHITECTURE.md §5](../_meta/ARCHITECTURE.md#5-hai-khối-stdlib-only)

← [Về HOME](../../HOME.md)
