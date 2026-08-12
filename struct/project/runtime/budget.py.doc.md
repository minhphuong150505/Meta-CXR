> Source: `runtime/budget.py` (132 dòng)
> Status: ✅ ACTIVE — chỉ P8 dùng
> Last verified against source: 2026-08-12

# `runtime/budget.py`

## Purpose
Trần chi phí theo **wall-clock**, có khả năng chịu resume.

## Hai quyết định đáng học
### 1. Tính theo wall-clock, không theo throughput
Một GPU bị treo tốn **đúng bằng** một GPU đang bận. Tính theo số mẫu đã xử lý bỏ
sót đúng trường hợp tốn kém nhất.

### 2. `prior_elapsed_seconds` — resume không reset trần
Không có nó, một run bị restart 10 lần có thể tiêu 10× ngân sách mà mỗi lần đều
"hợp lệ".

> **Nó chỉ dừng một run.** Không bao giờ tự hạ cấp model, không bao giờ bật thêm
> section để "tiết kiệm". Giới hạn ngân sách không được phép âm thầm đổi thí nghiệm.

## Status
```text
✅ ACTIVE — nhưng chỉ trong pipeline P8
```

## Main items
| Tên | Dòng |
|---|---|
| [`BudgetState`](budget.py.methods/BudgetState/_index.md) | 26 |
| `BudgetExceeded` | 21 |

## Calls / Called by
Gọi: stdlib (`time`).
Được gọi: `medgemma_inference/runner.py:23,159`; `tests/test_pretrained_findings.py:39`.

## Side effects
Không (đọc đồng hồ).

## Error / edge cases
Vượt trần → `BudgetExceeded` raise, run dừng.

## Related tests
`tests/test_pretrained_findings.py`

## Developer notes
Nếu thêm ngân sách cho Stage 1/Stage 2, **dùng lại module này** — đặc biệt là ngữ
nghĩa `prior_elapsed_seconds`.

## Source relationships

- **Parent:** [`_index.md`](_index.md)

← [HOME](../../HOME.md)
