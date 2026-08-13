> Source: `preporcessing/build_explanation_masks.py::transform_mask_geometry`
> Status: 🟡 CONDITIONAL

# `transform_mask_geometry(mask, ...)`

## Contract

Input binary `[H,W]` ở kích thước gốc → output `uint8[112,112]` `{0,255}`.

## Geometry invariant

`Resize(512)` là resize **một số nguyên**: cạnh ngắn thành 512 và giữ tỉ lệ. Sau
đó `CenterCrop(448)`, cuối cùng nearest-neighbor xuống 112². Số học resize/crop
khớp đường PIL của torchvision; interpolation nearest giữ mask nhị phân.

## Risk

Resize thẳng `(512,512)` hoặc dùng kích thước đã preprocess sẽ làm mask lệch ảnh
mà loss vẫn có thể giảm. Test geometry ghim invariant này.

← [`functions`](./_index.md)
