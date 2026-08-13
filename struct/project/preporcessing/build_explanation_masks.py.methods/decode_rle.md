> Source: `preporcessing/build_explanation_masks.py::decode_rle`
> Status: 🟡 CONDITIONAL

# `decode_rle(rle, height, width)`

## Contract

Trả `np.uint8[height,width]` với giá trị `{0,1}`. Empty RLE → toàn 0.

## Algorithm

1. Parse chuỗi thành các cặp start/length.
2. Đổi start từ one-based sang zero-based.
3. Tô run trên vector phẳng C-order.
4. `reshape(height, width)`.

Đây là đúng convention utility chính thức CheXmask, không phải COCO RLE
column-major. `decode_lung_union` gọi hàm hai lần rồi OR kết quả.

## Failure policy

Sai số token, số phần tử lẻ, run không dương hoặc vượt biên → `ValueError` không
chứa giá trị nguồn.

## Tests

`test_decode_rle_round_trip_matches_known_mask` ·
`test_decode_lung_union_combines_both_lungs`.

← [`functions`](./_index.md)
