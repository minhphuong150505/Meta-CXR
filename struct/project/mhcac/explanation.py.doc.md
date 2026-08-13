> Source: `mhcac/explanation.py` (160 dòng)
> Status: 🟡 CONDITIONAL — được wire vào Stage 1, chỉ chạy khi `lambda_explanation > 0`
> Last verified against source: 2026-08-13

# `mhcac/explanation.py`

## Purpose

Cài explanation-aware loss theo Grad-CAM cho nhánh phân loại Stage 1, với score
**Logit Difference Squared** và mask không gian theo từng encoder stream.

## Role in architecture

```text
student logits [B,14,3] + projected MHCAC streams + labels + mask
        │
        ├─ s = Σ_positive (z_pos - z_neg)²
        ├─ một autograd.grad cho tất cả stream
        ├─ Grad-CAM FP32 ở lưới riêng 14×14 / 7×7
        └─ top-k mềm + precision trong mask → L_exp
```

Module chỉ import `torch`, `torch.nn`, `torch.nn.functional`; không import LAVIS,
torchvision hay transformers nên test được trên CPU độc lập với training stack.

## Status

```text
🟡 CONDITIONAL — import luôn có, nhưng ExplanationLoss chỉ được dựng và gọi khi
model.loss.lambda_explanation > 0.
```

## Main functions / classes

| Tên | Doc | Vai trò |
|---|---|---|
| `logit_difference_squared` | [📄](explanation.py.methods/logit_difference_squared.md) | Score theo các bệnh có nhãn Positive + mask study hợp lệ |
| `grad_cam` | [📄](explanation.py.methods/grad_cam.md) | Eq. (3)–(4), arithmetic FP32 |
| `explanation_loss` | [📄](explanation.py.methods/explanation_loss.md) | Top-k mềm + Eq. (6) |
| `resize_mask_to_grid` | [📄](explanation.py.methods/resize_mask_to_grid.md) | Mask độ phân giải cao → lưới stream |
| `ExplanationLoss` | [📄](explanation.py.methods/ExplanationLoss/_index.md) | Một score, một `autograd.grad`, trung bình các stream |
| `explanation_lambda` | [📄](explanation.py.methods/explanation_lambda.md) | Lịch warmup đã duyệt |

`_cam_from_gradients` là helper nội bộ dùng chung cho `grad_cam` và
`ExplanationLoss.forward`; nó không gọi autograd lần thứ hai.

## Invariants

1. Study không có nhãn `Positive` không đóng góp và được đánh dấu `valid=False`.
2. Score chỉ dùng class index `1=Positive` và `0=Negative`; class Uncertain không
   tham gia hiệu logit.
3. CAM và gradient được cast sang FP32 trước phép trung bình/nhân/cộng.
4. Ngưỡng quantile được `detach`; giá trị giữ lại là saliency mềm
   `H_hat * 1[H_hat >= theta]`, không phải mask nhị phân.
5. `ExplanationLoss.forward` gọi `torch.autograd.grad` đúng một lần với danh sách
   mọi activation stream và `create_graph=True`.
6. Không có sample hợp lệ → zero nối với graph qua `logits.sum() * 0.0`.

## Inputs / outputs

`ExplanationLoss.forward` nhận:

- `logits [B,14,3]`, `labels [B,14]`;
- `streams: dict[name, (activation [B,N,C], (h,w))]`;
- `mask [B,H,W]` và `valid_mask [B]`.

Trả scalar trung bình giữa stream và dict scalar theo tên stream để caller có thể
log riêng. Mỗi mask được pool độc lập về đúng lưới của stream.

## Calls / called by

Được `Blip2Qformer.forward` gọi sau lời gọi MHCAC student. Tensor activation đến từ
`AbnormalityClassificationModel._last_cam_streams`; teacher và anchor-only không
bật capture.

## Configuration

`model.loss.lambda_explanation` và `model.explanation.{top_k,
warmup_start_epoch,warmup_epochs,streams}` được đọc tại
`Blip2Qformer.from_config`. Giai đoạn 1 không sửa config production nên default
`lambda_explanation=0.0` giữ đường cũ.

## Tests

`tests/test_explanation_loss.py`: score/validity, giới hạn loss trong/ngoài mask,
top-k mềm có gradient, Grad-CAM, double backprop tới module nhỏ, bảng warmup,
resize mask và vòng đời capture của MHCAC.

## Source relationships

- **Parent:** [`mhcac/_index.md`](_index.md)
- **Methods:** [`explanation.py.methods/`](explanation.py.methods/)
- **Caller:** [`blip2_qformer.py`](../model/lavis/models/blip2_models/blip2_qformer.py.doc.md)
- **Related:** [`mhcac_12.py`](mhcac_12.py.doc.md) · [`loss.py`](loss.py.doc.md)

← [HOME](../../HOME.md)
