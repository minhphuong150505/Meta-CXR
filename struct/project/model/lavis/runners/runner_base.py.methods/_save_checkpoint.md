> Source: `model/lavis/runners/runner_base.py:939-988`
> Status: ✅ ACTIVE

# `RunnerBase._save_checkpoint(cur_epoch, is_best=False, is_last=False, ...)`

## Located in

[`runner_base.py`](../runner_base.py.doc.md)

## Purpose
Ghi checkpoint, **lọc bỏ tham số đóng băng**.

## Execution flow
```text
unwrap_dist_model(model)          ← bỏ lớp DDP
   ↓
lọc state_dict: bỏ param có requires_grad=False   ← encoder đóng băng KHÔNG vào checkpoint
   ↓
payload = {model, optimizer, config, scaler, epoch}
   ↓
is_best  → checkpoint_best.pth
is_last  → checkpoint_last.pth
ngược lại → checkpoint_{epoch}.pth
```

## ★ Encoder đóng băng không vào checkpoint
Ba encoder chiếm phần lớn tham số nhưng **không đổi**. Bỏ chúng làm checkpoint nhỏ
đi rất nhiều.

⚠ Hệ quả: nạp checkpoint **cần dựng lại encoder từ pretrained weights trước**. Đó
là lý do `from_config` gọi `load_checkpoint_from_config` rồi `lavis_loader` mới nạp
checkpoint Stage 1 lên trên.

## ⚠ Buffer ITC không vào checkpoint
`register_buffer(..., persistent=False)` → queue rỗng sau resume.

## Config dependencies
`run.output_dir`, `run.run_name`, `run.save_freq`

## Side effects
Ghi file `.pth` (có thể vài GB).

## Modification risk
Bỏ bộ lọc `requires_grad` → checkpoint phình lên gấp nhiều lần và mang trọng số
encoder trùng lặp.
