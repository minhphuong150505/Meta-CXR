> Source: `preporcessing/build_explanation_masks.py`
> Status: 🟡 CONDITIONAL

# Methods/functions — `build_explanation_masks.py`

| Function | Responsibility |
|---|---|
| [`decode_rle`](decode_rle.md) | Decode RLE C-order, one-based, fail-closed |
| `decode_lung_union` | OR phổi trái/phải, bỏ heart |
| `rasterize_bbox_union` | Union bbox ở kích thước ảnh gốc |
| [`transform_mask_geometry`](transform_mask_geometry.md) | Resize cạnh ngắn → crop tâm → 112² |
| [`inspect_chexmask`](inspect_chexmask.md) | In tên cột, số hàng đọc, khoảng Dice — không in identifier |
| `_load_ms_cxr` | Gán bbox theo split của manifest; bỏ + đếm dòng không có trong manifest |
| [`build_mask_caches`](build_mask_caches.md) | Dựng cache mọi split, thống kê coverage/split mismatch |
| `main` | Parse CLI; `--inspect` thoát trước khi đọc config project |

← [`build_explanation_masks.py`](../build_explanation_masks.py.doc.md)
