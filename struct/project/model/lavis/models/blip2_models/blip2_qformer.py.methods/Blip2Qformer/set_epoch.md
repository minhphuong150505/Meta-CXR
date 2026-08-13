> Source: `model/lavis/models/blip2_models/blip2_qformer.py:431-432`
> Status: ✅ ACTIVE

# `Blip2Qformer.set_epoch(epoch)`

## Purpose

Lưu epoch index vào `current_epoch` để `forward` tính `lambda_eff` theo lịch
`explanation_lambda`.

## Called by

`RunnerBase.train_epoch` gọi trên model gốc trước `model.train()` và trước khi ủy
quyền cho task. Hook dùng `hasattr`, nên model LAVIS khác không có setter vẫn giữ
hành vi cũ.

## Side effects

Chỉ gán một integer; không thay optimizer, scheduler hay checkpoint state dict.
