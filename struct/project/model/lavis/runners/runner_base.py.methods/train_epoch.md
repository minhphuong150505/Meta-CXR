> Source: `model/lavis/runners/runner_base.py:840-858`
> Status: ✅ ACTIVE

# `RunnerBase.train_epoch(epoch)`

## Purpose

Chuẩn bị model cho một epoch rồi ủy quyền vòng microbatch/optimizer cho task.

## Execution flow

```text
IF self._model có set_epoch → self._model.set_epoch(epoch)
  ↓
self.model.train()             ← model có thể đã được bọc DDP
  ↓
self.task.train_epoch(...)
```

Setter được gọi trên `_model` gốc nên hoạt động giống nhau khi có hoặc không có
DDP. `hasattr` giữ nguyên hành vi cho model khác không cần biết epoch.

## Why it changed

`Blip2Qformer` cần epoch index để tính warmup của `lambda_explanation`; scheduler
LR vẫn giữ nguyên và tiếp tục đếm optimizer update, không dùng hook này.
