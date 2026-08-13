> Source: `mhcac/explanation.py:147-160`
> Status: 🟡 CONDITIONAL

# `explanation_lambda(epoch, lambda_max, warmup_start_epoch, warmup_epochs)`

## Purpose

Trả trọng số explanation loss theo bảng warmup đã duyệt.

Với `lambda_max=0.25`, `warmup_start_epoch=2`, `warmup_epochs=2`:

| Epoch | 0 | 1 | 2 | 3 | 4 | 9 |
|---|---:|---:|---:|---:|---:|---:|
| Lambda | 0.0 | 0.0 | 0.125 | 0.1875 | 0.25 | 0.25 |

Trong ramp, progress tăng tuyến tính từ 0.5 tại epoch bắt đầu đến 1.0 tại
`warmup_start_epoch + warmup_epochs`. `warmup_epochs <= 0` bật đầy đủ ngay tại
epoch bắt đầu.

## Called by

`Blip2Qformer.forward`, dùng `current_epoch` do `RunnerBase.train_epoch` truyền qua
`set_epoch`.
