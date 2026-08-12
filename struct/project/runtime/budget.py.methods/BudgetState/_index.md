> Source: `runtime/budget.py:26-132`
> Status: ✅ ACTIVE

# `BudgetState`

## Responsibility

Đo elapsed wall-clock cộng runtime các lần resume, quy đổi chi phí, enforce budget
và dự phóng full cohort từ throughput đã đo.

## Important state

`hourly_cost_usd`, `budget_limit_usd`, `max_runtime_hours`,
`prior_elapsed_seconds`, `processed_samples`, injected monotonic `clock`.

## Methods

| Method/property | Contract |
|---|---|
| `elapsed_seconds` | session hiện tại + prior, clamp không âm |
| `estimated_cost_usd` | elapsed hours × hourly cost |
| `assert_within_budget` | Raise trước sample mới nếu chạm cost/runtime ceiling |
| `record_samples` | Tăng throughput numerator |
| `project(target_samples)` | Trả zero cho tới khi có rate thật; không bịa estimate |
| `progress_line` | Heartbeat có spent/remaining/projection |

## Modification risk

Không chuyển sang billing theo sample/compute time: GPU stalled vẫn tính tiền.
Không bỏ prior elapsed, nếu không resume có thể vượt trần vô hạn.

← [`budget.py`](../../budget.py.doc.md) · [HOME](../../../../HOME.md)
