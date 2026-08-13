> Source: `mhcac/explanation.py:65-84`
> Status: 🟡 CONDITIONAL

# `explanation_loss(cam, mask, top_k=0.5, eps=1e-6)`

## Purpose

Đo phần khối lượng saliency top-k nằm ngoài mask mục tiêu, riêng cho từng sample.

## Execution flow

```text
CAM → min-max normalize theo sample → H_hat
theta = quantile(H_hat, 1-top_k).detach()
H_positive = H_hat * 1[H_hat >= theta]
L = 1 - sum(H_positive * mask) / (sum(H_positive) + eps)
```

## Critical invariant

`H_positive` **giữ giá trị mềm**. Phép so sánh chỉ làm gate; không thay saliency
bằng tensor 0/1. Nếu nhị phân hóa toàn bộ Eq. (5), gradient của loss theo CAM sẽ
bằng 0 và training hỏng im lặng.

## Inputs / outputs

`cam` và `mask` phải cùng shape `[B,H,W]`; trả vector loss `[B]`. `top_k` phải nằm
trong `(0,1]`, `eps > 0`.
