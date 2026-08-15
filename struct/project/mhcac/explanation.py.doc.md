> Source: `mhcac/explanation.py`
> Status: 🟡 CONDITIONAL — được wire vào Stage 1, chỉ chạy khi
> `lambda_explanation > 0` **hoặc** `lambda_explanation_strong > 0`
> Last verified against source: 2026-08-16

# `mhcac/explanation.py`

## Purpose

Cài explanation-aware loss theo Grad-CAM cho nhánh phân loại Stage 1, với score
**Logit Difference Squared** và mask không gian theo từng encoder stream.

## Role in architecture

**HAI term, tách 2026-08-16.** Trước đó chỉ có một, và nó gộp mọi bệnh dương
thành một score duy nhất — nghĩa là CAM trả lời "vùng nào giải thích *bất kỳ*
bệnh nào của study này", không phải "vùng nào giải thích bệnh X".

```text
student logits [B,14,3] + projected MHCAC streams + labels + masks
        │
        ├─ WEAK   (CheXmask lung, ~93% study, mask giải phẫu dùng chung 14 nhãn)
        │     s = Σ_positive (z_pos - z_neg)²        ← gộp, như cũ
        │     một autograd.grad cho tất cả stream
        │     top_k 0.2
        │
        └─ STRONG (MS-CXR box, 823 study train, mask RIÊNG theo bệnh)
              với mỗi bệnh L có box trong batch:
                 s_L = Σ_{study có box L} (z_pos,L - z_neg,L)²
                 một autograd.grad  → CAM riêng cho bệnh L
              top_k 0.5, trung bình có trọng số theo số cặp (study, bệnh)
```

**Chi phí của strong term là số *bệnh phân biệt* có box trong batch, không phải
14.** Thường là 1–2. Study không có box không tốn gì.

Hai term **không bao giờ được cộng ở đây** — chúng được trả riêng để
`Blip2Qformer` áp hai lambda khác nhau. Gộp lại sẽ để tín hiệu weak dồi dào nhấn
chìm tín hiệu strong khan hiếm mà nhìn vẫn như đang học localization.

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
| `logit_difference_squared` | [📄](explanation.py.methods/logit_difference_squared.md) | Score GỘP mọi bệnh Positive + mask study hợp lệ (dùng cho weak) |
| `single_label_score` | [📄](explanation.py.methods/single_label_score.md) | Score của MỘT bệnh trên các study được chọn (dùng cho strong) |
| `grad_cam` | [📄](explanation.py.methods/grad_cam.md) | Eq. (3)–(4), arithmetic FP32 |
| `explanation_loss` | [📄](explanation.py.methods/explanation_loss.md) | Top-k mềm + Eq. (6) |
| `resize_mask_to_grid` | [📄](explanation.py.methods/resize_mask_to_grid.md) | Mask độ phân giải cao → lưới stream |
| `ExplanationLoss` | [📄](explanation.py.methods/ExplanationLoss/_index.md) | Trả `(weak, strong, per_stream_weak)` — xem lưu ý arity dưới |
| `explanation_lambda` | [📄](explanation.py.methods/explanation_lambda.md) | Lịch warmup đã duyệt |

`_cam_from_gradients` là helper nội bộ dùng chung cho `grad_cam` và
`ExplanationLoss.forward`; nó không gọi autograd lần thứ hai.

## ⚠ `forward` trả về BA giá trị

Đổi 2026-08-16 từ `(total, per_stream)` sang `(weak, strong, per_stream)`.
Caller unpack hai giá trị sẽ vỡ ngay, đó là chủ đích — im lặng bỏ mất term
strong sẽ tệ hơn nhiều.

## ⚠ Grad-CAM cần ít nhất 2 channel mới định vị được

`_cam_from_gradients` lấy trung bình gradient theo **token** để ra trọng số
kênh, nên activation một kênh chỉ cho CAM phẳng, không phân biệt vị trí. Điều
này đã làm hỏng lần dựng test đầu tiên cho strong term; test hiện dùng hai kênh
tách trái/phải (`tests/test_explanation_loss.py::_TwoLabelModule`).

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
