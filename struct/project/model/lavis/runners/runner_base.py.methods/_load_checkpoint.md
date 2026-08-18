> Source: `model/lavis/runners/runner_base.py:1127-1250`
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
_resumed_best_metric = checkpoint.get("best_agg_metric")   ← :1176
_resumed_best_epoch  = checkpoint.get("best_epoch")
   → train() khôi phục vào best_agg_metric nếu **không phải None** (:719)
   ↓
logic đóng băng: :1050 `for param in self.model.mhcac.parameters()` …
   ↓
_delete_local_resume_checkpoint_after_load(path)   ← :1129, có thể XÓA file
```

## ⚠ `strict=False`
Cho phép nạp checkpoint thiếu key encoder đóng băng (xem
[`_save_checkpoint`](_save_checkpoint.md)). Nhưng cũng nghĩa là **key sai tên sẽ bị
bỏ qua lặng lẽ** — kiểm log nếu model có vẻ chưa được nạp.

## ⚠⚠ Đổi `selection_metric` khi resume làm `checkpoint_best` không bao giờ được ghi
Hỏng **hoàn toàn im lặng**: training chạy bình thường, log sạch, chỉ là không có
file best nào xuất hiện.

Cơ chế: một run `selection_metric: loss` (mode `min`) chưa chấm epoch nào ghi
`best_agg_metric: inf` xuống checkpoint (xem [`_save_checkpoint`](_save_checkpoint.md)).
Resume checkpoint đó dưới một metric F1/AUPRC (mode `max`) thì `:719` khôi phục lại
`inf`, sau đó [`validate`](validate.md) gọi `_metric_improved(value, inf)` — tức
`value > inf + min_delta` — và kết quả **luôn là False**.

Đã xác minh trên checkpoint thật 2026-08-15: `epoch=4, best_agg_metric=inf,
best_epoch=0`.

Cách xử lý — đặt key về `None`, vì `:719` chặn bằng `if resumed_metric is not None`
nên `None` giữ nguyên `-inf` vừa khởi tạo:

```python
ck = torch.load(p, map_location="cpu", weights_only=False)
assert ck["epoch"] == <epoch mong đợi>      # từ chối vá nhầm checkpoint
ck["best_agg_metric"] = None; ck["best_epoch"] = None
torch.save(ck, p)
```

Kiểm chứng bằng dòng log
`Resume checkpoint from ... (best_agg_metric=None, best_epoch=None)` (`:1179`).

`selection_mode` không cần đụng tới: nó được suy ra là `max` cho mọi metric có tên
**không chứa** chuỗi `loss` (`:501`).

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

## ★ `mid_epoch` quyết định epoch bắt đầu — thêm 2026-08-18

```python
if checkpoint.get("mid_epoch", False):
    self.start_epoch = checkpoint["epoch"]        # vào LẠI epoch đó
else:
    self.start_epoch = checkpoint["epoch"] + 1    # hành vi cũ
```

Không có nhánh này thì checkpoint ghi giữa epoch sẽ làm resume **nhảy cóc mất
nguyên một epoch**. Checkpoint viết trước 2026-08-18 không có khóa `mid_epoch`,
`.get(..., False)` đưa chúng về đúng hành vi cũ.

⚠ Resume từ checkpoint giữa epoch **không** tua lại vị trí trong dữ liệu: giữ
trọng số và moment Adam, mất vị trí, nên vài study được thấy hai lần trong epoch
đó. Bỏ qua các batch đã tiêu thụ đồng nghĩa giải nén lại chúng cho không — chính
là chi phí mà tính năng này muốn tránh. Ghim bởi
`tests/test_mid_epoch_checkpoint.py`.
