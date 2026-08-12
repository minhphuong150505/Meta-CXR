> Source: `model/lavis/models/blip2_models/blip2_qformer.py:52-65`
> Status: ✅ ACTIVE — inference-only ablation

# `_resolve_encoder_ablation(stream_names, active_encoders)`

## Purpose

Chuyển danh sách encoder cần **giữ** từ YAML thành tuple encoder cần zero. Nếu
`active_encoders` rỗng/không có, trả `()` và giữ nguyên đường model đã train.

## Contract

- Mọi tên active phải tồn tại trong `shared_visual_projector.stream_names`.
- Tên lạ gây `ValueError` kèm danh sách stream model đã dựng; không fallback.
- Thứ tự output theo thứ tự stream đã build, giúp ablation tái lập ổn định.

Được `from_config` dùng để đặt `model.ablate_encoders`. Xem
[`_apply_encoder_ablation`](Blip2Qformer/_apply_encoder_ablation.md).

← [`blip2_qformer.py`](../blip2_qformer.py.doc.md) · [HOME](../../../../../../HOME.md)
