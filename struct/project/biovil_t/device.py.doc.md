> Source: `biovil_t/device.py` (14 dòng)
> Status: ✅ ACTIVE — utility
> Last verified against source: 2026-08-12

# `biovil_t/device.py`

## Purpose

`get_module_device(module)` trả device của parameter đầu tiên, hoặc CPU nếu module
không có parameter. BioViL dùng helper này khi cần tạo tensor cùng device với
model mà không giả định `cuda:0`.

## Contract

- Input: `torch.nn.Module`.
- Output: `torch.device`.
- Module rỗng không raise `StopIteration`; trả `torch.device("cpu")`.

## Calls / Called by

Chỉ dùng `torch`. Được gọi trong `biovil_t/encoder.py` và code downstream của
BioViL.

## Modification risk

Đừng đổi fallback thành CUDA: module không parameter phải vẫn chạy được trên CPU.

← [`biovil_t/`](_index.md) · [HOME](../../HOME.md)
