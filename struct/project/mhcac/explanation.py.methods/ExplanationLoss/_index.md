> Source: `mhcac/explanation.py:98-144`
> Status: 🟡 CONDITIONAL

# `class ExplanationLoss(nn.Module)`

## Responsibility

Gộp score Logit Difference Squared, Grad-CAM nhiều stream và spatial explanation
loss thành một module khả vi bậc hai.

## Constructor

`ExplanationLoss(top_k=0.5)` lưu tỉ lệ pixel cần giữ; giá trị ngoài `(0,1]` bị từ chối.

## Public method

[`forward`](forward.md) — trả scalar trung bình và dict loss theo stream.

## State

Chỉ có `top_k`; không có parameter, buffer hay cache activation.
