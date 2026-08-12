> Source: `safety/reconciler.py` (328 dòng)
> Status: ❓ UNKNOWN — chỉ pipeline + test
> Last verified against source: 2026-08-12

# `safety/reconciler.py`

## Purpose
Quyết định làm gì với claim **không** verify được: giữ nguyên, hedge (làm mềm), bỏ
số đo, hay xóa.

## Status
```text
❓ UNKNOWN
```

## Main items
| Tên | Dòng | Vai trò |
|---|---|---|
| `RuleBasedClaimReconciler` | 117 | ★ `.reconcile(...)` |
| `ReconciliationOutcome` | 79 | Kết quả + danh sách sửa đổi |
| `ClaimEdit` | 57 | Một sửa đổi |
| `hedge(sentence)` | 96 | ★ "There is X" → "There may be X" |
| `strip_measurements(sentence)` | 111 | ★ Bỏ số đo không verify được |

## Ba mức can thiệp, không phải một
Thay vì chỉ "giữ hoặc xóa", module này có mức trung gian:
- **hedge** — giữ nội dung nhưng hạ mức khẳng định
- **strip_measurements** — bỏ con số cụ thể (thứ dễ bịa nhất) nhưng giữ quan sát

Với báo cáo y khoa, xóa cả câu có thể mất thông tin đúng; giữ nguyên có thể lan
truyền số bịa. Mức trung gian là câu trả lời hợp lý.

## `require_grounding=True`
Chế độ nghiêm ngặt: claim không có bằng chứng grounding **không được giữ nguyên**.
`tests/test_safety_pipeline.py:113` kiểm chế độ này.

## Calls / Called by
Gọi: `safety.claims` (`:31`), `safety.verifiers` (`:41`), `re`.
Được gọi: `safety/pipeline.py:29,99`; `tests/test_safety_pipeline.py:23,113,125`.

## Side effects
Không (trả text mới).

## Related tests
`tests/test_safety_pipeline.py`

## Developer notes
`hedge()` và `strip_measurements()` là **can thiệp vào text y khoa**. Bất kỳ thay
đổi nào ở đây nên được bác sĩ review, không chỉ review code.

## Source relationships

- **Parent:** [`_index.md`](_index.md)

← [HOME](../../HOME.md)
