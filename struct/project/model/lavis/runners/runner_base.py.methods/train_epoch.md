> Source: `model/lavis/runners/runner_base.py:860-892`
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
self.task.train_epoch(..., on_sync_step=<closure hoặc None>)
```

## ★ `on_sync_step` — checkpoint giữa epoch (2026-08-18)

Khi `run.save_every_iters > 0`, dựng một closure và truyền xuống vòng lặp trong
`base_task._train_inner_loop`. Vòng lặp gọi nó với `i + 1` **chỉ ở sync step,
sau `optimizer.zero_grad()`** — gọi ở chỗ khác sẽ chụp optimizer đang ôm một
window tích lũy dở, mà gradient đó bị vứt khi resume, nên checkpoint mã hóa một
update chưa từng được áp dụng ở effective batch sai.

Closure bắt `best_agg_metric` / `best_epoch` qua
`self._mid_epoch_best_agg_metric` / `_mid_epoch_best_epoch`, được `train()` gán
ngay trước mỗi epoch. Không có chúng thì resume từ checkpoint giữa epoch sẽ reset
best-tracking và ghi đè `checkpoint_best` bằng điểm tệ hơn.

Lỗi trong closure được bắt và ghi log chứ không ném lên: bỏ lỡ một checkpoint
tốn 17 phút, ném exception tốn cả epoch.

Setter được gọi trên `_model` gốc nên hoạt động giống nhau khi có hoặc không có
DDP. `hasattr` giữ nguyên hành vi cho model khác không cần biết epoch.

## Why it changed

`Blip2Qformer` cần epoch index để tính warmup của `lambda_explanation`; scheduler
LR vẫn giữ nguyên và tiếp tục đếm optimizer update, không dùng hook này.
