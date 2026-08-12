> Source: `training/train_eval_figure9_llm_variants_200.py:1130-1306`
> Status: ✅ ACTIVE

# `VariantLLM.train_fine(...)`

## Located in

[`training/train_eval_figure9_llm_variants_200.py`](../../train_eval_figure9_llm_variants_200.py.doc.md)

## Purpose
Vòng huấn luyện Stage 2: gradient accumulation, validation CE, early stopping,
lưu adapter.

## Execution flow
```text
optimizer + scheduler
   ↓
FOR epoch:
   FOR batch in DataLoader(RecordDataset, collate_fn=collate_train):
      _forward_batch(batch) → loss
      loss / accumulation_window_size(...)      ← ★ đuôi accumulation
      backward
      mỗi grad_accum: clip → step → zero_grad
   ↓
   evaluate_loss(val_records)    ← ★ chọn checkpoint theo VALIDATION CE
   save_adapter(checkpoints/last, status=resumable)  ← MỖI EPOCH
   cải thiện → save_adapter(output root, status=complete)
   ↓
early stopping theo patience
```

## Resume hiện tại

`resume_state` có thể là file `trainer_state.pt` hoặc thư mục chứa file đó. Code
khôi phục optimizer, scheduler, epoch kế tiếp, `global_step`, best validation
loss, bad-epoch count, DataLoader generator và RNG CPU/CUDA. Adapter `last` được
ghi sau từng epoch, nên crash giữa epoch mất tối đa phần epoch đang chạy; crash
sau epoch có checkpoint resumable.

## ★ `accumulation_window_size` — đuôi epoch
`training/stage2_utils.py:159`. Batch cuối thường không đủ `grad_accum` bước; chia
loss cho **số microbatch thực có**, không cho hằng số. Nếu không, gradient của
batch cuối bị scale sai.

## Parameters
`train_records`, `val_records`, epochs, batch size, `max_length` (mặc định 768),
grad accum, lr.

## Side effects
Cấp phát GPU · Ghi `checkpoints/last` mỗi epoch · promote best adapter · Log

## Config dependencies
`--train-limit`, `--val-limit`, và các hyperparameter trong `parse_args`.

## Tests
`tests/test_training_core.py` (đuôi accumulation, ở mức helper)

## Modification risk
Nếu thêm checkpoint giữa **microbatch** hoặc resume chính xác giữa epoch, cân nhắc
`training/trainer/CheckpointManager`; module hiện tại resume ở ranh giới epoch.
