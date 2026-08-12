> Source: `model/lavis/models/blip2_models/blip2_qformer.py:490-496`
> Status: ✅ ACTIVE — inference-only ablation

# `Blip2Qformer._apply_encoder_ablation(shared)`

## Purpose

Zero đúng span của các encoder bị loại sau `SharedVisualTokenProjector`, giữ
nguyên shape, vị trí token, expert tokens và mọi trọng số học được. Đây là cơ chế
đã dùng để tạo Table 5 từ cùng một checkpoint ba encoder.

## Safety contract

- `ablate_encoders == ()` trả lại chính object `shared`, không đổi đường gốc.
- Nếu `self.training` là `True`, raise `RuntimeError`; ablation không được dùng
  để giả thành một recipe training encoder đơn.
- Việc zero gọi `SharedVisualTokens.without(...)`, không dựng lại model và không
  thay layout checkpoint.

## Called by

`_encode_image_streams` ngay sau shared projection. Được test trực tiếp bởi
`tests/test_encoder_ablation.py`.

← [`Blip2Qformer`](_index.md) · [`blip2_qformer.py`](../../blip2_qformer.py.doc.md)
