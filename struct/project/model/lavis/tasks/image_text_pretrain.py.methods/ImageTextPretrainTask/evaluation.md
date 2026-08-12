> Source: `model/lavis/tasks/image_text_pretrain.py:53-254`
> Status: ✅ ACTIVE

# `ImageTextPretrainTask.evaluation(model, data_loader, cuda_enabled=True)`

## Purpose

Chạy validation/test Stage 1, gom logits/labels/key, tính metric classification và
tùy config lưu prediction NPZ để calibrate offline.

## Flow

```text
model.eval + no_grad
  → mỗi batch: prepare_sample → model(samples)
  → lấy classification_logits + labels + masks + sample keys
  → gom chunk CPU
  → _build_predictions → ClassificationPredictions
  → import trễ training.evaluation.*
  → evaluate_classification(...)
  → nếu save_predictions: _save_predictions
  → dict metric cho RunnerBase
```

## Important conditions

`generation_mask` và `classification_mask` giữ semantics khác nhau. Import
evaluation nằm trong method để LAVIS không kéo dependency ngược ở module scope.
NPZ có derived patient data nên không commit.

## Called by / tests

`RunnerBase.eval_epoch`; `tests/test_stage1_eval_hook.py`.

← [`image_text_pretrain.py`](../../image_text_pretrain.py.doc.md) · [HOME](../../../../../../HOME.md)
