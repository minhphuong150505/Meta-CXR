> Source: `mhcac/explanation.py:32-62`
> Status: 🟡 CONDITIONAL

# `grad_cam(score, activations, grid_hw, create_graph=True)`

## Purpose

Cài Eq. (3)–(4): lấy gradient của tổng score theo activation, trung bình gradient
theo N vị trí để có trọng số kênh, rồi `ReLU(Σ_c alpha_c A_c)` và reshape về lưới.

## Data flow

```text
score [B] + A [B,N,C]
  → autograd.grad(score.sum(), A, create_graph=...)
  → alpha [B,1,C] = mean_N(gradient)
  → ReLU(sum_C(alpha * A))
  → CAM [B,h,w]
```

Arithmetic CAM dùng `activations.float()` và `gradients.float()`. `N` bắt buộc
bằng `h*w`; hàm không suy đoán hay nội suy một chuỗi không gian không hợp lệ.

`ExplanationLoss.forward` dùng helper `_cam_from_gradients` vì nó cần gom mọi
stream vào **một** lời gọi autograd thay vì gọi hàm public này nhiều lần.
