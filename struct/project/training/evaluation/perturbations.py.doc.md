> Source: `training/evaluation/perturbations.py` (180 dòng)
> Status: ❓ UNKNOWN — chỉ counterfactual + test dùng
> Last verified against source: 2026-08-12

# `perturbations.py`

## Purpose
Các phép làm hỏng ảnh cho kiểm chứng counterfactual.

## Status
```text
❓ UNKNOWN — chỉ `counterfactual.py` và test dùng
```

## Main items
| Hàm | Dòng | Làm gì |
|---|---|---|
| `blank_image(image)` | 61 | Ảnh trắng/đen |
| `constant_image(image)` | 66 | Giá trị hằng |
| `shuffled_pixels(image, seed)` | 76 | Xáo pixel — giữ histogram, phá cấu trúc |
| `shuffled_patches(image, seed, patch=16)` | 86 | Xáo patch — giữ texture cục bộ |
| `region_occlusion(...)` | 114 | Che một vùng |
| `pick_random_donor(...)` | 146 | Lấy ảnh từ study khác |
| `apply_self_contained(name, image, seed)` | 166 | ★ Áp phép không cần donor |
| `Donor` | 139 | |
| `_check_image(image)` | 53 | Validate shape |

**Bốn mức độ phá khác nhau** cho phép định vị model dựa vào gì: histogram, texture
cục bộ, hay cấu trúc giải phẫu toàn cục.

## Calls / Called by
Gọi: `torch`.
Được gọi: `evaluation/counterfactual.py:32`; `tests/test_counterfactual.py:20`.

## Side effects
Không (trả tensor mới).

## Error / edge cases
`_check_image` raise nếu shape sai.

## Related tests
`tests/test_counterfactual.py`

## Developer notes
`apply_self_contained` tách riêng vì các phép cần **donor** (ảnh study khác) có
rủi ro trộn dữ liệu giữa bệnh nhân — cần cẩn thận hơn.

## Source relationships

- **Parent:** [`training/evaluation/`](_index.md)
- **Related:** [`schemas.py`](schemas.py.doc.md)

← [HOME](../../../HOME.md)
