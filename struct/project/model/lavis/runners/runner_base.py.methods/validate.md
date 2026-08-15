> Source: `model/lavis/runners/runner_base.py:571-690`
> Status: ✅ ACTIVE

# `RunnerBase.validate(cur_epoch, best_agg_metric, best_epoch, wandb_run)`

## Located in

[`runner_base.py`](../runner_base.py.doc.md)

## Purpose
Chạy eval trên **toàn bộ** validation split, gộp qua rank, so với best.

⚠ **Bỏ qua eval khi `cur_epoch < run.eval_start_epoch`** (chỉ với run train; run
`evaluate_only` truyền `cur_epoch="provided"` nên không bao giờ bị bỏ). Phần lưu
`checkpoint_last` / `save_freq` ở cuối hàm **vẫn chạy**, nên mốc resume không mất.
Vì `checkpoint_best` chỉ ghi trong nhánh eval, epoch bị bỏ cũng tự động không thể
được tuyển.

## Execution flow
```text
cur_epoch < eval_start_epoch ?  → log + bỏ qua eval, nhảy xuống save checkpoint_last
   ↓ (ngược lại)
eval_epoch("val", cur_epoch)  → ImageTextPretrainTask.evaluation
   ↓
_reduce_eval_stats(eval_stats)          ← gộp qua các rank DDP
   ↓
lấy giá trị theo selection_metric
_metric_improved(value, best_agg_metric)  ← có min_delta, theo selection_mode
   ↓
log_stats(..., "val")  → W&B
return (agg_metric, best_epoch)
```

## ★ Toàn bộ split, không phải subset
Validation chạy trên **tất cả** mẫu val. Với DDP, `_reduce_eval_stats` gộp lại
trước khi so sánh — nếu không, mỗi rank sẽ chọn checkpoint khác nhau.

## Config dependencies
`run.selection_metric` = `loss` (mode suy ra `min`) ·
`early_stop_min_delta` = 1e-4 · `save_predictions`

## Side effects
Model sang `.eval()` rồi trả về `.train()` · Ghi `.npz` (qua task) · Log W&B

## Error handling
⚠ Hành vi khi `selection_metric` không có trong `eval_stats` — cần runtime verification.

## Tests
`tests/test_stage1_eval_hook.py` ⚠ cần `model.lavis`

## Modification risk
`min_delta` quá lớn → early stop quá sớm. Quá nhỏ → chạy hết 20 epoch dù đã hội tụ.
