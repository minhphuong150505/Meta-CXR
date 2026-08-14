> Source: `model/lavis/data/ReportDataset.py` (1.335 dòng)
> Status: ✅ ACTIVE — ★
> Last verified against source: 2026-08-13

# `ReportDataset.py`

## Purpose

Dataset và processor cho Stage 1. Quan trọng nhất là `MIMIC_CXR_Dataset` — nơi
**study-level sampling**, **anchor/auxiliary selection**, **mask**, và **collate
ragged** được hiện thực. Khi cache explanation được cấu hình, đây cũng là nơi
mask 112² được căn cùng affine với ảnh anchor.

## Why it exists

Dataset LAVIS gốc coi mỗi ảnh là một mẫu. MIMIC-CXR có nhiều ảnh cho một study và
một báo cáo cho cả study — dùng nguyên bản sẽ lặp báo cáo và đánh trọng số quá cao
cho study nhiều view. File này thay bằng lấy mẫu theo study.

## Role in architecture

```text
split CSV ──► MIMIC_CXR_Dataset.__getitem__ ──► collater ──► Blip2Qformer.forward
```

## Status

```text
✅ ACTIVE — MIMIC_CXR_Dataset
⚠ CheXpertDataset, IU_Xray_Dataset: có trong file nhưng train.py KHÔNG dựng
```

## Used in

Training ✅ · Validation ✅ · Test ✅ · Inference ✅ (transform, qua `inference.py`)

## Entry point

Không.

## Inputs

`configs/env_config.yaml` → `VIS_ROOT`, `PROCESSED_{TRAIN,VAL,TEST}_CSV` ·
`cfg.model_cfg.data.*` (`study_sampling`, `anchor_priority`, `max_aux_views`) ·
`cfg.datasets_cfg` (image_size, resize_size, augmentation) · `run.feature_cache_dir` ·
`model.explanation.mask_cache_dir`

## Outputs

Dict sample — xem [DATA_FLOW.md §2.2](../../../_meta/DATA_FLOW.md#22-__getitem__--một-sample).

## Main classes

| Class | Status | Vai trò |
|---|---|---|
| `MIMIC_CXR_Dataset` (`:221`) | ✅ ★ | Dataset chính |
| `MIMIC_CXR_Builder` (`:1000`) | ✅ | `@registry.register_builder("mimic_cxr")` |
| `MyBlipCaptionProcessor` (`:49`) | ✅ | `@registry` text processor |
| `MyReportProcessor` (`:195`) | ✅ | Processor prompt báo cáo |
| `ExpandChannels` (`:91`) | ✅ | Ảnh 1 kênh → 3 kênh |
| `Conversation`, `SeparatorStyle` (`:119`,`:126`) | 🕰 | Của đường Vicuna; trong `__getitem__` đã **bị comment hết** (`:887-903`) |
| `MIMICEvalCap` (`:1009`) | [📄](ReportDataset.py.methods/MIMICEvalCap/_index.md) ⚠ | Không thấy caller |
| `CheXpertDataset` (`:1091`) | ⚠ | `train.py` không dựng |
| `IU_Xray_Dataset` (`:1207`) | ⚠ | `train.py` không dựng |

## Main functions

| Hàm | Doc | Vai trò |
|---|---|---|
| `create_chest_xray_transform_for_inference` (`:107`) | — | Dùng bởi `inference.py:28` |

## Main methods — `MIMIC_CXR_Dataset`

| Method | Doc | Vai trò |
|---|---|---|
| `__init__` (`:222`) | [📄](ReportDataset.py.methods/MIMIC_CXR_Dataset/__init__.md) | Nạp CSV, dựng transform, index study, feature cache |
| `_init_study_index` (`:498`) | [📄](ReportDataset.py.methods/MIMIC_CXR_Dataset/_init_study_index.md) | ★ Gộp ảnh → study, chọn anchor |
| `_row_visual` (`:788`) | [📄](ReportDataset.py.methods/MIMIC_CXR_Dataset/_row_visual.md) | ★ Ảnh hoặc cached feature; **chốt bảo mật path** |
| `__getitem__` (`:837`) | [📄](ReportDataset.py.methods/MIMIC_CXR_Dataset/__getitem__.md) | ★ Một study → một sample |
| `collater` (`:950`) | [📄](ReportDataset.py.methods/MIMIC_CXR_Dataset/collater.md) | ★ Pad aux view ragged về `N_max` |
| `_init_feature_cache` (`:546`) | — | Mở store feature cache |
| `_init_explanation_mask_cache` (`:592`) | — | Đọc JSON index; chưa mở `.npy` |
| `_get_explanation_mask_memmap` (`:654`) | — | Mở memmap lazy theo PID worker |
| `_read_explanation_mask` (`:679`) | — | Cache row → mask/valid/source |
| `_apply_synced_image_mask_transforms` (`:690`) | — | Một affine cho ảnh bilinear + mask nearest |
| `load_image` (`:837`), `remap_to_uint8` (`:790`) | — | Đọc ảnh + kéo giãn min–max. ★ **`remap_to_uint8` từng là nút cổ chai của toàn bộ training** — xem bên dưới |
| `_view_id` (`:495`) | — | `ViewPosition` → int |
| `_coerce_bool` (`:477`) | — | Ép cột cờ về bool |
| `set_custom_epoch` (`:732`) | — | Đặt epoch cho sampling |
| `__len__` (`:947`) | — | **Số study**, không phải số ảnh |

## Execution flow

```text
__init__  → read_csv → _coerce_bool → split geometric/optical transforms
          → load mask JSON index → _init_feature_cache → _init_study_index
__getitem__(i)
   ├─ studies[i] → anchor row
   ├─ IF mask cache: read cache row (memmap mở lazy trong worker)
   ├─ _row_visual(anchor)        → image (+ synchronized mask) HOẶC <enc>_feat
   ├─ findings, chexpert labels, classification_mask, generation_mask
   ├─ IF mask cache: explanation_mask [112,112], valid, source
   └─ IF multi_view: _row_visual cho từng aux → "aux_image" (ragged)
collater(samples)
   ├─ default collate cho key thường
   └─ pad aux về N_max, dựng aux_mask + aux_view_ids
```

## Calls

`mimic_cxr_utils.build_study_index`, `view_id` · `torchvision.transforms` +
`RandomAffine.get_params`/functional affine ·
`PIL`, `numpy` · `BaseDataset.collater`

## Called by

`pretraining/train.py:118-155` · `training/stage1/lavis_loader.py:31,96` ·
`inference.py:28` (chỉ transform) · `tests/test_mimic_data_pipeline.py`

## Side effects

Đọc CSV + mask JSON index vào RAM · Đọc ảnh từ đĩa mỗi `__getitem__` ·
Mở explanation memmap read-only ở lần worker access đầu tiên · Mở feature cache
store · Không ghi gì.

## Error / edge cases

| Tình huống | Hành vi |
|---|---|
| `image_path` tuyệt đối hoặc `C:/…` | **`ValueError`** (`:803`) — chốt bảo mật |
| `image_path` thoát `vis_root` (`..`) | **`ValueError`** (`:808`) |
| DICOM vắng trong feature cache | **`KeyError`** nêu tên DICOM + gợi ý `study_sampling=false` (`:826`) |
| Cache mask thiếu file/index sai schema/shape | fail-closed trước hoặc ở lần đọc worker đầu |
| Study không có entry mask | mask zero, `explanation_mask_valid=False` |
| Marker Kaggle cũ `/mimic-cxr-jpg-lite/` | Vẫn hỗ trợ, cắt lấy phần sau |
| `N_max == 0` | Tensor rỗng `[B,0,…]`, **không** `None` |

## Related tests

`tests/test_mimic_data_pipeline.py` (study sampling) ·
`tests/test_explanation_mask_pipeline.py` (cache lazy, synchronized affine,
geometry và no-cache regression)

## Related documentation

[DATA_FLOW.md §2](../../../_meta/DATA_FLOW.md#2-stage-1--csv--tensor) ·
[`preporcessing/_index.md`](../../../preporcessing/_index.md)

## Developer notes

1. ⚠ **`.gitignore` chặn `model/lavis/data/`** nhưng file này đã được track.
   `git add .` **không** bắt thay đổi — dùng `git add -f`.
2. ⚠ **`ReportDataset.py:1117`** (trong `CheXpertDataset`) có
   `df[col].fillna(x, inplace=True)` — pandas 3.0 Copy-on-Write khiến nó **âm thầm
   không làm gì**. Không nằm trên đường MIMIC-CXR nên chưa ảnh hưởng.
3. **Augmentation áp một lần trong dataset**, để mọi encoder thấy cùng một ảnh đã
   biến đổi. Đừng thêm augmentation ở phía model.
4. Khi mask cache bật, affine được sample **một lần**: ảnh dùng bilinear, mask
   được upsample 448² rồi dùng nearest và hạ lại 112². Dùng cùng pixel translate
   trực tiếp trên mask 112² sẽ lệch 4 lần.
5. Khi không có `mask_cache_dir`, ba key explanation không được phát và chuỗi
   transform giữ nguyên hành vi cũ.
6. **Đừng đổi `image_path` sang tuyệt đối** — `_row_visual` sẽ raise.
7. Code Vicuna cũ (`Conversation`, prompt) còn nằm dưới dạng comment trong
   `__getitem__` (`:862-903`). Không xóa (ngoài phạm vi task này), nhưng đừng tưởng
   nó đang chạy.

## Source relationships

- **Parent:** [`model/lavis/_index.md`](../_index.md)
- **Methods:** [`ReportDataset.py.methods/`](ReportDataset.py.methods/)
- **Related:** [`blip2_qformer.py`](../models/blip2_models/blip2_qformer.py.doc.md)

← [HOME](../../../../HOME.md)


## Nhãn CheXpert: ô trống bị mask, không phải âm tính

`__init__` ánh xạ export CheXpert thành `0=âm, 1=dương, 2=không chắc,
IGNORE_LABEL(-100)=ô trống`. Ô trống nghĩa là labeler không thấy nhắc tới, chứ
không phải bác sĩ loại trừ; 79,4% ma trận nhãn là ô trống.

Hệ quả đã đo trên `processed/full_allviews_v2` (222.758 study train):

- trung bình chỉ **2,86/14** nhãn còn lại mỗi study; **31%** study còn đúng 1 nhãn
- dương tính thành lớp đa số ở **12/14** nhãn (Atelectasis 44.718 dương / 1.502 âm)
- `No Finding` còn **0** mẫu âm — CheXpert chỉ đánh 1 hoặc để trống — nên nó là
  nhãn một lớp; đã bị loại khỏi macro metric bởi `run.include_meta_labels: false`

Không consumer nào phải sửa: `ClassificationLoss` giữ `labels_i >= 0` và ma trận
nhầm lẫn lúc eval giữ `labels >= 0`, cả hai đã có sẵn từ trước.
`preporcessing/preprocess_mimic_cxr.py::clean_chexpert` áp đúng ánh xạ này để hai
đường không lệch nhau, dù cột nhãn của nó không được ghi vào split CSV.

Ghim bởi `tests/test_blank_label_masking.py`.


## `remap_to_uint8` — nút cổ chai của toàn bộ pipeline (2026-08-14)

Cho tới 2026-08-14, training chạy **đúng bằng tốc độ dataloader**:

| | s/it (batch 16) |
|---|---|
| run thật | 0,6524 |
| dataloader chạy một mình, không GPU | **0,6520** |
| model chạy một mình, batch nạp sẵn | **0,2099** |

GPU nhàn 68% thời gian. ⚠ **`data: 0.0000` trong log KHÔNG loại trừ được điều
này**: `MetricLogger` đo thời gian `next()` bị chặn, nhưng vòng lặp Python chạy
trước CUDA bất đồng bộ nên lúc nó gọi `next()` thì batch đã tới, còn thời gian
chờ thật rơi vào `time:`. Muốn thấy phải đo **loader riêng, không GPU** và
**model riêng, batch nạp sẵn**, rồi so cả hai với run.

Thủ phạm: `remap_to_uint8` làm `array.astype(float)` → **float64** trên ảnh 7 MP
**nguyên bản, trước mọi phép resize**, rồi quét toàn mảng thêm bốn lượt —
khoảng **336 MB lưu lượng DRAM mỗi ảnh**. Nó chiếm **45,7% của 83,6 ms mỗi
study**, nhiều hơn cả giải mã JPEG. Vì nghẽn **băng thông** chứ không phải tính
toán, 12 worker bóp nghẹt lẫn nhau: đạt 24,5 study/s thay vì 143 study/s như chi
phí đơn luồng dự đoán.

Trên đầu vào 8-bit, hàm này là phép ánh xạ tuyến tính từ 256 giá trị sang 256
giá trị — tức **đúng bằng một bảng tra 256 phần tử**. Và trên bộ dữ liệu này
bảng tra là **ma trận đồng nhất**: mọi ảnh đã dùng hết dải [0,255], nên hàm đang
tốn 24 ms/ảnh để trả về chính đầu vào của nó.

Đường nhanh uint8 **bit-identical**, đã kiểm trên ảnh thật và ghim bằng
`tests/test_remap_to_uint8.py` (12 test). Đường float64 tổng quát giữ nguyên cho
đầu vào 16-bit DICOM mà BioViL-T vốn thiết kế cho.

Kết quả:

| | trước | sau |
|---|---|---|
| `__getitem__` | 83,6 ms/study | **42,5 ms** |
| dataloader | 0,6520 s/batch (24,5 study/s) | **0,1722 s/batch (92,9 study/s)** |
| vòng lặp thật | 0,6346 s/it | **0,2347 s/it** |
| `next(it)` chiếm | 26,5% wall | **0,1%** |
| step chiếm | 67,5% wall | **94,1%** |
| 10 epoch | ~25 h | **~9,1 h** |

`CheXpertDataset` và `IU_Xray_Dataset` giữ bản sao riêng của hàm này, **cố ý
không sửa** — không nằm trên đường training.

Thủ phạm tiếp theo nếu muốn nhanh hơn: **giải mã JPEG**, giờ chiếm 66% mỗi item.
`Image.draft()` giải ở 1/4 kích thước trong miền DCT sẽ cắt vài lần, nhưng nó
**đổi chuỗi lấy mẫu tức là đổi pixel** — là thay đổi tiền xử lý thật, không miễn
phí, và chưa làm.
