> Source: `mhcac/explanation.py:105-144`
> Status: 🟡 CONDITIONAL

# `ExplanationLoss.forward(logits, labels, streams, mask, valid_mask)`

## Purpose

Tính explanation loss chung cho mọi stream mà chỉ dựng một đồ thị đạo hàm bậc hai.

## Execution flow

```text
logit_difference_squared(...) → score, positive_valid
  ↓ không valid / không stream: graph-connected zero
torch.autograd.grad(score.sum(), inputs=[A_1, ..., A_n], create_graph=True)  ← một lần
  ↓ mỗi stream
Grad-CAM → resize mask về grid_hw → loss vector → masked mean
  ↓
mean(per_stream scalars), dict[name, scalar]
```

`valid_mask` được giao với điều kiện study có bệnh Positive bên trong
`logit_difference_squared`. Vì vậy mask tồn tại nhưng ảnh âm tính vẫn không đóng góp.

## Modification risk

Gọi `grad_cam` riêng trong vòng lặp sẽ backward lặp lại cho từng encoder, tăng chi
phí và có thể giải phóng graph sai thời điểm. Giữ lời gọi `autograd.grad` gom danh
sách activation.
