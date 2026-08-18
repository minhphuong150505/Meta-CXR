> Source: `model/lavis/runners/runner_base.py:1020-1105`
> Status: ✅ ACTIVE

# `RunnerBase._save_checkpoint(cur_epoch, is_best=False, is_last=False, mid_epoch=False, iters_done=None, ...)`

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
payload = {model, optimizer, config, scaler, epoch,
           best_agg_metric, best_epoch,
           mid_epoch, iters_done}           ← dùng khi resume
   ↓
is_best  → checkpoint_best.pth
is_last  → checkpoint_last.pth
ngược lại → checkpoint_{epoch}.pth
   ↓
torch.save(payload, save_to + ".tmp")  →  os.replace(tmp, save_to)   ← ATOMIC
```

## ★ Ghi atomic — thêm 2026-08-18
Trước đây gọi thẳng `torch.save(save_obj, save_to)`, **không** atomic. File
~3,8 GB (319M tham số huấn luyện được + hai moment Adam), nên crash giữa lúc ghi
để lại `checkpoint_last.pth` cụt — phá đúng thứ nó sinh ra để bảo vệ, và phá
đúng lúc dễ crash nhất. Giờ ghi ra `.tmp` cạnh đích rồi `os.replace`; temp file
cố ý đặt cùng thư mục để chắc chắn cùng filesystem (`os.replace` chỉ atomic
trong một filesystem). Khối `except BaseException` bắt cả `KeyboardInterrupt`
để không bỏ lại `.tmp` mồ côi — một file như vậy dễ bị nhầm là checkpoint dùng
được. Ghim bởi `tests/test_mid_epoch_checkpoint.py`.

## ★ `mid_epoch` / `iters_done` — thêm 2026-08-18
Đánh dấu checkpoint ghi **giữa** epoch. Xem
[`run.save_every_iters`](../../../../pretraining/configs/mimic_cxr_full.yaml.doc.md)
và [`_load_checkpoint`](_load_checkpoint.md) — cờ này quyết định resume vào lại
epoch đó hay nhảy sang epoch kế.

## ⚠ `best_agg_metric` được ghi kể cả khi chưa chấm epoch nào
`train()` khởi tạo nó là `+inf` (mode `min`) hoặc `-inf` (mode `max`) tại `:694`,
và `_save_checkpoint` ghi nguyên giá trị đó xuống đĩa. Với `eval_start_epoch: 5`,
mọi `checkpoint_last.pth` của epoch [0]–[4] mang **`best_agg_metric: inf`** — một
giá trị vô nghĩa, không phải điểm số thật.

Nó chỉ vô hại khi resume **cùng** `selection_metric`. Đổi metric lúc resume thì
đây là bẫy — xem [`_load_checkpoint`](_load_checkpoint.md).

## ★ Encoder đóng băng không vào checkpoint
Ba encoder chiếm phần lớn tham số nhưng **không đổi**. Bỏ chúng làm checkpoint nhỏ
đi rất nhiều.

⚠ Hệ quả: nạp checkpoint **cần dựng lại encoder từ pretrained weights trước**. Đó
là lý do `from_config` gọi `load_checkpoint_from_config` rồi `lavis_loader` mới nạp
checkpoint Stage 1 lên trên.

## ⚠ Buffer ITC không vào checkpoint
`register_buffer(..., persistent=False)` → queue rỗng sau resume.

## Config dependencies
`run.output_dir`, `run.run_name`, `run.save_freq`, `run.save_every_iters`

## Side effects
Ghi file `.pth` (có thể vài GB).

## Modification risk
Bỏ bộ lọc `requires_grad` → checkpoint phình lên gấp nhiều lần và mang trọng số
encoder trùng lặp.

Quay lại `torch.save` thẳng vào đích → mất tính atomic, và vì giờ ghi mỗi 1.000
iter chứ không phải mỗi epoch, xác suất crash trúng lúc ghi tăng lên rõ rệt.
