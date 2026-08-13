> Source: `mhcac/explanation.py:87-95`
> Status: 🟡 CONDITIONAL

# `resize_mask_to_grid(mask, grid_hw)`

## Purpose

Đưa mask nhị phân `[B,H,W]` về lưới CAM `(h,w)` bằng
`adaptive_avg_pool2d`, sau đó ngưỡng `> 0` và trả float 0/1.

Ngưỡng sau average-pool mang ngữ nghĩa union: ô lưới được bật nếu bất kỳ phần nào
của vùng mask độ phân giải cao rơi vào ô đó.

## Output

Tensor float32 `[B,h,w]`, chỉ chứa 0 hoặc 1.
