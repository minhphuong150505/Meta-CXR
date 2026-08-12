> Source: `medgemma_inference/runner.py` (240 dòng)
> Status: ✅ ACTIVE
> Last verified against source: 2026-08-12

# `medgemma_inference/runner.py`

## Purpose
Điều phối inference: guard, lazy load, vòng lặp, ngân sách, ghi kết quả.

## ★ Thứ tự thao tác — docstring `:3` nói rõ
1. **Guard Impression chạy trước khi nạp bất cứ thứ gì** → cấu hình sai fail trong
   mili-giây, không phải sau khi tải xong checkpoint 4B thứ hai.
2. **Model dựng lazy**, chỉ khi còn việc chưa xong → một run đã resume hoàn toàn
   tốn **0 GPU time**.

Hai điều này nghe nhỏ nhưng tiết kiệm rất nhiều tiền và thời gian trên GPU thuê.

## Status
```text
✅ ACTIVE
```

## Main items
| Tên | Dòng | Vai trò |
|---|---|---|
| [`run_findings_inference(...)`](runner.py.methods/run_findings_inference.md) | 128 | ★ |
| `RunSummary` | 29 | |
| `build_run_identity(...)` | 81 | ★ Vân tay run |
| `write_cost_estimate(...)` | 96 | |
| `_build_reporter(config)` | 57 | |
| `load_image(path)` | 46 | |

## Calls / Called by
Gọi: `runtime.budget.BudgetState/BudgetExceeded` (`:23`, `:159`),
`model.pretrained_medgemma.*`, `.prediction_writer`, `.progress`, `PIL`.
Được gọi: `run_pretrained_findings.py:main`.

## Side effects
Cấp phát GPU (lazy) · Ghi JSONL từng dòng · Ghi cost estimate

## Error / edge cases
`BudgetExceeded` → dừng sạch, giữ nguyên kết quả đã ghi · `ResumeMismatch` →
từ chối append

## Related tests
`tests/test_pretrained_findings.py`, `tests/test_inference_only_invariants.py`

## Developer notes
⚠ **Không được dựng optimizer / tính gradient / gọi `model.train()`** ở đây.
`tests/test_inference_only_invariants.py` canh.

← [`_index.md`](_index.md) · [HOME](../../HOME.md)
