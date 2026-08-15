> Source: `mhcac/explanation.py`
> Status: 🟡 CONDITIONAL
> Last verified against source: 2026-08-16

# `ExplanationLoss.forward(logits, labels, streams, mask, valid_mask, bbox_masks=None, bbox_valid=None)`

## Purpose

Tính **hai** explanation loss có mức độ chứng cứ khác nhau, trả về riêng biệt.

## ⚠ Đổi arity 2026-08-16

Trả về `(weak, strong, per_stream_weak)` — trước đó là `(total, per_stream)`.
Caller cũ vỡ ngay lập tức, và đó là chủ đích: nuốt mất term strong trong im lặng
tệ hơn nhiều so với một `ValueError` khi khởi động.

## Execution flow

```text
không có stream → (0, 0, {})
  │
  ├─ WEAK  _weak_term(...)
  │    logit_difference_squared(...) → score gộp, positive_valid
  │    một torch.autograd.grad(score.sum(), [A_1..A_n], create_graph=True)
  │    mỗi stream: Grad-CAM → resize mask → explanation_loss(top_k) → masked mean
  │    → mean qua các stream, kèm dict per-stream
  │
  └─ STRONG _strong_term(...)
       bbox_valid [B,A] → tập BỆNH phân biệt có box trong batch
       với mỗi bệnh L (thường 1–2, không phải 14):
          single_label_score(logits, L, rows) → một autograd.grad
          mỗi stream: Grad-CAM → resize bbox_masks[:,L] → explanation_loss(strong_top_k)
       → trung bình CÓ TRỌNG SỐ theo số study mang bệnh đó
```

## Vì sao trung bình có trọng số

Nếu lấy trung bình đều theo bệnh, một bệnh xuất hiện ở một study sẽ nặng ngang
một bệnh xuất hiện ở mười study. Nhân theo `row_count` rồi chia tổng khiến giá
trị là trung bình trên **cặp (study, bệnh)**, đúng đơn vị mà box được vẽ.

## Parameters

| Tên | Nghĩa |
|---|---|
| `mask` | `[B,H,W]` mask phổi CheXmask (weak) |
| `valid_mask` | `[B]` — study có mask hợp lệ; giao thêm điều kiện có bệnh Positive bên trong `logit_difference_squared` |
| `bbox_masks` | `[B,A,H,W]` box MS-CXR theo bệnh; `None` → strong = 0 |
| `bbox_valid` | `[B,A]` bool — cặp (study, bệnh) nào có box |

## Error / edge cases

| Tình huống | Hành vi |
|---|---|
| `bbox_masks`/`bbox_valid` là `None` | strong = 0, graph-connected |
| `bbox_valid` không phải `[B,A]` | `ValueError` nêu rõ hai shape |
| `bbox_masks` và `bbox_valid` lệch `[B,A]` | `ValueError` |
| Không cặp nào hợp lệ | strong = 0 |
| Chạy dưới `torch.no_grad()` | `RuntimeError` — xem gate `torch.is_grad_enabled()` trong `Blip2Qformer.forward` |

## Modification risk

Gọi `grad_cam` riêng trong vòng lặp cho weak term sẽ backward lặp lại cho từng
encoder. Giữ lời gọi `autograd.grad` gom danh sách activation.

Với strong term, gộp mọi bệnh vào một `autograd.grad` sẽ **xoá bỏ toàn bộ mục
đích** của nó: CAM lại trở về dạng gộp và box theo bệnh trở thành vô nghĩa.
Pinned bởi `tests/test_explanation_loss.py::test_strong_term_is_disease_specific`.
