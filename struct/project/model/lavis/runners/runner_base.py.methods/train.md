> Source: `model/lavis/runners/runner_base.py:692-826`
> Status: ✅ ACTIVE

# `RunnerBase.train(wandb_run)`

## Located in

[`runner_base.py`](../runner_base.py.doc.md)

## Purpose
Vòng epoch: train → validate → chọn checkpoint → early stop → test một lần.

## Execution flow
```text
resume_ckpt_path có → _load_checkpoint()
   ↓
FOR epoch in range(max_epoch):
   train_epoch(epoch) → model.set_epoch(epoch), nếu có
                      → task.train_epoch → _train_inner_loop
   validate(epoch, best, best_epoch, wandb_run)
   _metric_improved(value, best)?
      ✓ → _save_checkpoint(is_best=True)  → checkpoint_best.pth ; reset counter
      ✗ → early_stop_counter += 1
   _save_checkpoint(is_last=True)         → checkpoint_last.pth
   epoch % save_freq == 0 → _save_checkpoint(epoch)
   counter >= early_stop_patience → break
   ↓ sau vòng lặp
evaluate(cur_epoch="best")   → _reload_best_model → eval_epoch("test")
```

## Parameters
`wandb_run` — run thật ở rank 0, disabled ở rank khác.

## ★ Test chạy đúng MỘT LẦN
Ngoài vòng lặp, sau khi reload `checkpoint_best`. Test **không tham gia** chọn
checkpoint ở bất kỳ bước nào.

## Config dependencies
`run.max_epoch` · `selection_metric` (**`macro_auprc`**) · `selection_mode` ·
`early_stop_patience` (5) · `early_stop_min_delta` · `save_freq` (5) ·
`resume_ckpt_path` · `accum_grad_iters` · `max_grad_norm`

## Side effects
Ghi checkpoint · Log W&B · Có thể **xóa** checkpoint resume cục bộ sau khi nạp (`:1129`)

## Important conditions
```python
save_freq == 0 → chỉ giữ checkpoint_best + checkpoint_last
```

## Tests
`tests/test_training_core.py` (scheduler, đuôi accumulation)

## Modification risk
Đưa test vào vòng chọn checkpoint = **rò rỉ test set**, làm mọi số báo cáo vô nghĩa.
