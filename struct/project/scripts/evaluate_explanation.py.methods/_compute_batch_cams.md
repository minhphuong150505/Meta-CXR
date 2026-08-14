> Source: `scripts/evaluate_explanation.py::_compute_batch_cams`

# `_compute_batch_cams(model, batch, device, amp_context)`

## Contract

Chạy đường image-only giống `Blip2Qformer.forward`: `_encode_image_streams`, bật
`mhcac.capture_streams`, gọi MHCAC student, lấy `_last_cam_streams`, rồi luôn reset
hai field trong cleanup.

Score dùng `logit_difference_squared(logits, labels, valid)`; CAM dùng trực tiếp
`mhcac.explanation.grad_cam`. Không copy công thức Grad-CAM. CAM native 14²/7²
được bilinear về 112² và min–max từng sample trước metric.

## Gradient invariant

Hàm mở `torch.enable_grad()` rõ ràng và không được đặt dưới `no_grad`. Default
`create_graph=True` của helper giữ shared graph qua nhiều stream; kết quả detach
sang NumPy ngay sau đó. `model.zero_grad(set_to_none=True)` chỉ dọn state, không
có backward/optimizer step.

← [`methods`](./_index.md)
