> Source: `model/lavis/runners/runner_base.py:1010-1135`
> Status: ✅ ACTIVE

# `RunnerBase._load_checkpoint(url_or_filename)`

## Located in

[`runner_base.py`](../runner_base.py.doc.md)

## Purpose
Resume: nạp model + optimizer + scaler + epoch.

## Execution flow
```text
url_or_filename là URL → tải về; ngược lại đọc file
   ↓
torch.load(map_location=...)
   ↓
model.load_state_dict(checkpoint["model"], strict=False)   ← ⚠ strict=False
   ↓
optimizer.load_state_dict, scaler.load_state_dict (nếu có)
start_epoch = checkpoint["epoch"] + 1
   ↓
logic đóng băng: :1050 `for param in self.model.mhcac.parameters()` …
   ↓
_delete_local_resume_checkpoint_after_load(path)   ← :1129, có thể XÓA file
```

## ⚠ `strict=False`
Cho phép nạp checkpoint thiếu key encoder đóng băng (xem
[`_save_checkpoint`](_save_checkpoint.md)). Nhưng cũng nghĩa là **key sai tên sẽ bị
bỏ qua lặng lẽ** — kiểm log nếu model có vẻ chưa được nạp.

## ⚠ Freeze-list mồ côi
```python
for token in ("mhcac", "aggregator", "cls_loss_fn")    # :189
```
`aggregator` **không còn tồn tại** trên `Blip2Qformer`.
[I2](../../../../_meta/LEGACY_AND_OPTIONAL.md#-potential-issues--ghi-nhận-không-sửa).

## ⚠ Có thể xóa checkpoint
`_delete_local_resume_checkpoint_after_load` (`:1129`) — đọc kỹ điều kiện trước khi
resume từ file bạn muốn giữ.

## Config dependencies
`run.resume_ckpt_path`

## Side effects
Đọc file (có thể tải về từ URL) · Nạp state vào model/optimizer/scaler ·
⚠ **Có thể xóa file checkpoint cục bộ**

## Modification risk
Đổi `strict=False` thành `True` sẽ làm mọi checkpoint hiện có không nạp được (thiếu
key encoder).
