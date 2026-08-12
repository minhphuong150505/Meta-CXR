> Source: `biovil_t/resnet.py` (86 dòng)
> Status: ✅ ACTIVE — backbone implementation
> Last verified against source: 2026-08-12

# `biovil_t/resnet.py`

## Purpose

ResNet đã sửa để trả đồng thời feature map và pooled vector cho BioViL.

## Main items

| Item | Vai trò |
|---|---|
| `ResNetHIML.forward(x, return_intermediate_layers=False)` | Forward backbone; có thể trả intermediate |
| `_resnet(...)` | Factory nội bộ |
| `resnet18(...)`, `resnet50(...)` | Constructor backbone |

## Data flow

Feature map không bị bỏ: nó trở thành patch sequence sau flatten/reshape ở tầng
encoder/model. Vì vậy đổi stride/dilation làm thay đổi `P` trong `[B,P,D]` và ảnh
hưởng view fusion/Q-Former.

## Dependencies

`torchvision.models.resnet`; đây là một lý do import stack Stage 1 cần torchvision.

← [`biovil_t/`](_index.md) · [HOME](../../HOME.md)
