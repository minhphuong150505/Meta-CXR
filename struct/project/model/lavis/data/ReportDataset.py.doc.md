> Source: `model/lavis/data/ReportDataset.py` (1.130 dòng)
> Status: ✅ ACTIVE — ★
> Last verified against source: 2026-08-12

# `ReportDataset.py`

## Purpose

Dataset và processor cho Stage 1. Quan trọng nhất là `MIMIC_CXR_Dataset` — nơi
**study-level sampling**, **anchor/auxiliary selection**, **mask**, và **collate
ragged** được hiện thực.

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
`cfg.datasets_cfg` (image_size, resize_size, augmentation) · `run.feature_cache_dir`

## Outputs

Dict sample — xem [DATA_FLOW.md §2.2](../../../_meta/DATA_FLOW.md#22-__getitem__--một-sample).

## Main classes

| Class | Status | Vai trò |
|---|---|---|
| `MIMIC_CXR_Dataset` (`:221`) | ✅ ★ | Dataset chính |
| `MIMIC_CXR_Builder` (`:801`) | ✅ | `@registry.register_builder("mimic_cxr")` |
| `MyBlipCaptionProcessor` (`:49`) | ✅ | `@registry` text processor |
| `MyReportProcessor` (`:195`) | ✅ | Processor prompt báo cáo |
| `ExpandChannels` (`:91`) | ✅ | Ảnh 1 kênh → 3 kênh |
| `Conversation`, `SeparatorStyle` (`:119`,`:126`) | 🕰 | Của đường Vicuna; trong `__getitem__` đã **bị comment hết** (`:695-711`) |
| `MIMICEvalCap` (`:810`) | [📄](ReportDataset.py.methods/MIMICEvalCap/_index.md) ⚠ | Không thấy caller |
| `CheXpertDataset` (`:892`) | ⚠ | `train.py` không dựng |
| `IU_Xray_Dataset` (`:1008`) | ⚠ | `train.py` không dựng |

## Main functions

| Hàm | Doc | Vai trò |
|---|---|---|
| `create_chest_xray_transform_for_inference` (`:107`) | — | Dùng bởi `inference.py:28` |

## Main methods — `MIMIC_CXR_Dataset`

| Method | Doc | Vai trò |
|---|---|---|
| `__init__` (`:222`) | [📄](ReportDataset.py.methods/MIMIC_CXR_Dataset/__init__.md) | Nạp CSV, dựng transform, index study, feature cache |
| `_init_study_index` (`:477`) | [📄](ReportDataset.py.methods/MIMIC_CXR_Dataset/_init_study_index.md) | ★ Gộp ảnh → study, chọn anchor |
| `_row_visual` (`:627`) | [📄](ReportDataset.py.methods/MIMIC_CXR_Dataset/_row_visual.md) | ★ Ảnh hoặc cached feature; **chốt bảo mật path** |
| `__getitem__` (`:665`) | [📄](ReportDataset.py.methods/MIMIC_CXR_Dataset/__getitem__.md) | ★ Một study → một sample |
| `collater` (`:751`) | [📄](ReportDataset.py.methods/MIMIC_CXR_Dataset/collater.md) | ★ Pad aux view ragged về `N_max` |
| `_init_feature_cache` (`:525`) | — | Mở store feature cache |
| `load_image` (`:609`), `remap_to_uint8` (`:574`) | — | Đọc + chuẩn hóa percentile |
| `_view_id` (`:474`) | — | `ViewPosition` → int |
| `_coerce_bool` (`:456`) | — | Ép cột cờ về bool |
| `set_custom_epoch` (`:571`) | — | Đặt epoch cho sampling |
| `__len__` (`:748`) | — | **Số study**, không phải số ảnh |

## Execution flow

```text
__init__  → read_csv → _coerce_bool → transform → _init_study_index → _init_feature_cache
__getitem__(i)
   ├─ studies[i] → anchor row
   ├─ _row_visual(anchor)        → "image"  HOẶC  "<enc>_feat"
   ├─ findings, chexpert labels, classification_mask, generation_mask
   └─ IF multi_view: _row_visual cho từng aux → "aux_image" (ragged)
collater(samples)
   ├─ default collate cho key thường
   └─ pad aux về N_max, dựng aux_mask + aux_view_ids
```

## Calls

`mimic_cxr_utils.build_study_index`, `view_id` · `torchvision.transforms` ·
`PIL`, `numpy` · `BaseDataset.collater`

## Called by

`pretraining/train.py:118-155` · `training/stage1/lavis_loader.py:31,96` ·
`inference.py:28` (chỉ transform) · `tests/test_mimic_data_pipeline.py`

## Side effects

Đọc CSV vào RAM (`self.annotation`) · Đọc ảnh từ đĩa mỗi `__getitem__` ·
Mở feature cache store · Không ghi gì.

## Error / edge cases

| Tình huống | Hành vi |
|---|---|
| `image_path` tuyệt đối hoặc `C:/…` | **`ValueError`** (`:637`) — chốt bảo mật |
| `image_path` thoát `vis_root` (`..`) | **`ValueError`** (`:643`) |
| DICOM vắng trong feature cache | **`KeyError`** nêu tên DICOM + gợi ý `study_sampling=false` (`:654`) |
| Marker Kaggle cũ `/mimic-cxr-jpg-lite/` | Vẫn hỗ trợ, cắt lấy phần sau |
| `N_max == 0` | Tensor rỗng `[B,0,…]`, **không** `None` |

## Related tests

`tests/test_mimic_data_pipeline.py` (study sampling — nạp `mimic_cxr_utils.py` theo path)

## Related documentation

[DATA_FLOW.md §2](../../../_meta/DATA_FLOW.md#2-stage-1--csv--tensor) ·
[`preporcessing/_index.md`](../../../preporcessing/_index.md)

## Developer notes

1. ⚠ **`.gitignore` chặn `model/lavis/data/`** nhưng file này đã được track.
   `git add .` **không** bắt thay đổi — dùng `git add -f`.
2. ⚠ **`ReportDataset.py:897`** (trong `CheXpertDataset`) có
   `df[col].fillna(x, inplace=True)` — pandas 3.0 Copy-on-Write khiến nó **âm thầm
   không làm gì**. Không nằm trên đường MIMIC-CXR nên chưa ảnh hưởng.
3. **Augmentation áp một lần trong dataset**, để mọi encoder thấy cùng một ảnh đã
   biến đổi. Đừng thêm augmentation ở phía model.
4. **Đừng đổi `image_path` sang tuyệt đối** — `_row_visual` sẽ raise.
5. Code Vicuna cũ (`Conversation`, prompt) còn nằm dưới dạng comment trong
   `__getitem__` (`:672-711`). Không xóa (ngoài phạm vi task này), nhưng đừng tưởng
   nó đang chạy.

## Source relationships

- **Parent:** [`model/lavis/_index.md`](../_index.md)
- **Methods:** [`ReportDataset.py.methods/`](ReportDataset.py.methods/)
- **Related:** [`blip2_qformer.py`](../models/blip2_models/blip2_qformer.py.doc.md)

← [HOME](../../../../HOME.md)
