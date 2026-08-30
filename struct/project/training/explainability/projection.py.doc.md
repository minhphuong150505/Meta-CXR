> Source: `training/explainability/projection.py` (297 dòng)
> Status: ✅ ACTIVE — hệ toạ độ PubMedCLIP đã xác minh trên máy train 2026-08-30
> Last verified against source: 2026-08-30

# `training/explainability/projection.py`

## Purpose

Chuyển toạ độ lưới đặc trưng sang toạ độ ảnh. Tensor thuần, chỉ import `torch`.

## Hệ toạ độ Stage 1 — và vì sao nó là một assert, không phải comment

Dataset áp `Resize(512)` cạnh ngắn rồi `CenterCrop(448)`, ra **một** tensor
3×448×448 mà **mọi** encoder cùng đọc (`ReportDataset.py:455-457`):

| encoder | lưới | px mỗi ô | phủ |
|---|---|---|---|
| BioViL-T | 14×14 | 32 | 448 |
| PubMedCLIP | 7×7 (+1 CLS) | 64 | 448 |

`Pubmedclip.forward` đưa chính tensor 448×448 đó qua `CLIPImageProcessorFast`;
vì đầu vào **đã vuông**, center-crop 224 là no-op và phép biến đổi thuần tuý là
hạ mẫu. Hai encoder cùng field of view. `14*32 == 7*64 == 448`.

✅ **XÁC MINH trên máy train ngày 2026-08-30.**
`preprocessor_config.json` của `flaviagiammarino/pubmed-clip-vit-base-patch32`:

```json
{ "do_resize": true, "size": 224,
  "do_center_crop": true, "crop_size": 224, "resample": 3 }
```

Đầu vào là tensor **vuông** 448×448, nên resize đưa về 224×224 và center-crop
224 là **no-op** — hạ mẫu thuần tuý của cùng một crop, không có crop thứ hai,
không lệch toạ độ. Đây giờ là bằng chứng, không còn là suy luận.

Dù vậy nó vẫn **không được để nằm trong comment**:
`assert_shared_coordinate_frame(STAGE1_GRIDS)` chạy **lúc import module** và ném
`ValueError` nếu số học sai. Hàm nhận grid tuỳ ý nên `attention_capture.py` gọi
lại được với hình học model thật báo về.

## ⚠ KHÔNG hardcode số patch của MedGemma

Máy dev không load được model; một hằng số sai sẽ reshape sai bản đồ mà không
báo gì. `infer_square_grid(num_tokens)` suy ra lưới từ số token và **từ chối**
nếu không phải số chính phương — thông báo lỗi chỉ thẳng nguyên nhân thường gặp:
còn dính global/CLS token, phải tách bằng `split_global_tokens` trước.

## Main items

| Item | Vai trò |
|---|---|
| `GridSpec(height, width, patch_px)` | Lưới + ô ảnh nó lát. `covered_px`, `num_tokens` |
| `STAGE1_GRIDS` | `biovil` 14×14@32, `pubmedclip` 7×7@64 — kỳ vọng, không phải nguồn sự thật |
| `assert_shared_coordinate_frame(grids, image_size)` | Mọi lưới phải lát **đúng cùng một** ô vuông |
| `infer_square_grid(num_tokens, patch_px)` | Suy lưới vuông, từ chối đoán |
| `split_global_tokens(attribution, n)` | Tách token phi-không-gian ở **đầu** stream, khớp `StreamLayout.num_global_tokens` |
| `assert_spatial_projection_supported(source)` | Từ chối `qformer_soft_token` |
| `normalize_map(x)` | Min-max từng map; map hằng → **0** |
| `project_to_image(attribution, grid, image_size, mode, normalize, source)` | `[N]`/`[B,N]` → `[448,448]`/`[B,448,448]` |
| `SpatialProjectionUnsupported` | Kiểu riêng, để `except RuntimeError` trần không nuốt mất |

## ⚠ Vì sao từ chối chiếu soft token của Q-Former

32 soft token không mang vị trí — xem [`_index.md`](_index.md#-ràng-buộc-số-hai-đâu-là-đường-có-nghĩa-không-gian).
`project_to_image` gọi `assert_spatial_projection_supported` **trước khi** chạm
tới shape, để caller không nhận lỗi shape rồi "sửa" nó thành một bức ảnh vô
nghĩa.

## `normalize_map` — pin vào Stage 1, không được trôi

Hành vi giống hệt `_normalize_cam` trong `scripts/evaluate_explanation.py`, kể cả
quy tắc "map hằng → 0" (map phẳng không mang thứ hạng nào; kéo giãn về [0,1] là
bịa ra một thứ hạng). Cần bản torch để module giữ được tính thuần tensor, nên hai
cái được **pin số học vào nhau** bằng
`test_normalize_map_matches_stage1_normalize_cam`.

`project_to_image` dùng `bilinear` + `align_corners=False`, giống
`evaluate_explanation.py`, để CAM Stage-1 và map Stage-2 resample cùng cách và so
sánh được.

## Error / edge cases

- Lưới không lát đúng ô vuông → `ValueError` nêu số px thực tế.
- Số giá trị ≠ `grid.num_tokens` → `ValueError`. Trường hợp điển hình: truyền 50
  token PubMedCLIP còn dính CLS vào lưới 49 ô.
- Giá trị không hữu hạn → `ValueError`.
- `source="qformer_soft_token"` → `SpatialProjectionUnsupported`.

## Related tests

`tests/explainability/test_projection.py` — 33 test. Test ô vuông tổng hợp chạy
4 góc × 2 lưới, kiểm **cả** khối lượng theo góc phần tư **và** vị trí pixel đỉnh.
Có một test riêng chỉ dùng `top_right`, vì transpose gửi `top_right`→`bottom_left`
nhưng để nguyên `top_left`/`bottom_right` — chỉ góc lệch đường chéo mới bắt được.

Tầng GPU (đẩy ảnh thật có ô vuông tương phản qua encoder thật) **chưa viết**, cần
máy train.
