> Source: `preporcessing/build_explanation_masks.py`
> Status: 🟡 CONDITIONAL — chạy một lần trước recipe explanation-aware
> Last verified against source: 2026-08-13

# `build_explanation_masks.py`

## Purpose

Dựng cache mask giải thích hai tầng cho Stage 1 mà không cần đọc ảnh gốc:

```text
CheXmask RLE phổi trái/phải ─┐
                             ├─ ưu tiên bbox → geometry → uint8 112×112
MS-CXR bbox theo ảnh ────────┘
```

MS-CXR bbox (`mask_source=1`) ghi đè prior giải phẫu phổi
(`mask_source=0`). Mask không đạt `Dice RCA (Mean) >= 0.7` bị loại. Ảnh không có
nguồn hợp lệ không xuất hiện trong JSON index, nên dataset trả `valid=False`.

## Privacy / data governance

Đầu ra là dẫn xuất MIMIC-CXR, chỉ được đặt trên private storage và đã bị
`.gitignore` chặn cho cả `*.npy` lẫn ba tên JSON index theo split. Script không
log identifier, patient/study key, đường dẫn ảnh hay report. `--inspect` chỉ in
tên cột và **hình dạng** identifier.

## Inputs

| Nguồn | Cách đọc | Cột dùng |
|---|---|---|
| CheXmask OriginalResolution | `read_csv(..., chunksize=256)` | `dicom_id`, Dice mean, hai RLE phổi, `Height`, `Width` |
| MS-CXR | DataFrame nhỏ | bbox pixel + kích thước ảnh + split nguồn |
| Ba manifest của project | từ `local_config.py` | identity study/ảnh và `ViewPosition` |

Script chọn đúng một anchor mỗi study với cùng ưu tiên PA → AP → lateral như
`MIMIC_CXR_Dataset`. Split project là nguồn chân lý; cột `split` của MS-CXR chỉ
được dùng để đếm và cảnh báo số dòng lệch, không dùng để định tuyến bbox.

## RLE convention

Decoder bám utility chính thức của CheXmask:

- flatten NumPy C-order (theo hàng);
- start position đánh số từ 1;
- chuỗi là các cặp `(start, run_length)`;
- left/right lung được OR; heart không được dùng.

Run lẻ, không dương hoặc vượt kích thước đều fail-closed mà không echo nội dung
RLE.

## Geometry

Mask được rasterize ở đúng `Height×Width` / `image_height×image_width`, sau đó:

```text
Resize(512 as int: cạnh ngắn=512, giữ tỉ lệ)
  → CenterCrop(448)
  → Resize nearest 112×112
```

`resize_shorter_side` và `center_crop` tái hiện số học torchvision bằng Pillow;
không resize thẳng source mask thành hình vuông.

## Outputs

Mỗi split được ghi atomically:

- `masks_<split>.npy`: memmap-compatible `uint8`, shape `[N_valid,112,112]`,
  giá trị `{0,255}`;
- `index_<split>.json`: mapping identifier → `{"row": int, "mask_source": 0|1}`.

Trong lúc build có một working memmap theo số study; cuối cùng chỉ hàng hợp lệ
được compact sang file công khai của cache.

## CLI

```bash
python preporcessing/build_explanation_masks.py --inspect
python preporcessing/build_explanation_masks.py \
  --split val --limit 200 --output-dir <private-cache-dir>
```

Flags: `--split {train,val,test,all}`, `--chexmask-csv`, `--ms-cxr-csv`,
`--output-dir`, `--limit`, `--dice-threshold`, `--inspect`.

## Main functions

| Function | Doc | Role |
|---|---|---|
| `decode_rle` / `decode_lung_union` | [📄](build_explanation_masks.py.methods/decode_rle.md) | RLE chính thức → binary mask |
| `transform_mask_geometry` | [📄](build_explanation_masks.py.methods/transform_mask_geometry.md) | geometry train → 112×112 |
| `build_mask_caches` | [📄](build_explanation_masks.py.methods/build_mask_caches.md) | orchestration chunked + bbox priority |
| `inspect_chexmask` | [📄](build_explanation_masks.py.methods/inspect_chexmask.md) | schema/shape inspection không lộ ID |

## Tests

`tests/test_explanation_mask_pipeline.py`: RLE round-trip, union phổi, bbox
override, Dice gate, geometry và output cache tổng hợp.

## Dependencies

Chỉ stdlib + `pandas`, `numpy`, `Pillow`. Không import torch/torchvision.

← [`_index.md`](_index.md) · [HOME](../../HOME.md)
