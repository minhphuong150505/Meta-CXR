> Source: `biovil_t/`
> Status: ✅ ACTIVE
> Last verified against source: 2026-08-12

# `biovil_t/`

## Purpose

Vision encoder **BioViL-T** (Microsoft) — backbone chính của Stage 1. Chiều ra
**1408**, cũng chính là `VISUAL_DIM` mà mọi encoder khác được chiếu về.

## ⚠ Đây là bản ĐƯỢC DÙNG

Có hai bản sao trong repo. Bản này (`biovil_t/` ở root) là bản **mọi import trỏ
tới**. `vision_encoders/biovil_t/` là bản sao không ai dùng.

```python
# model/lavis/models/blip2_models/blip2.py:20-22
from biovil_t.model import ImageModel
from biovil_t.pretrained import get_biovil_t_image_encoder, _download_biovil_t_image_model_weights
from biovil_t.types import ImageEncoderType
```

## Role in project

```text
image [B,3,448,448] → ImageModel → .projected_patch_embeddings → [B,P,1408]
                                        ↓
                                    ln_vision (LayerNorm, KHÔNG phải chiếu chiều)
                                        ↓
                                  ViewFusionModule["biovil"]
                                        ↓
                                  SharedVisualTokenProjector (Identity cho luồng này)
```

Vì BioViL-T đã ra đúng 1408, `SharedVisualTokenProjector` cấp cho nó một
`Identity` thay vì `Linear` (`blip2_qformer.py:330-331`).

## Parent

[`struct/project/`](../../HOME.md#source-code-tree)

## Children

| File | Vai trò | Status |
|---|---|---|
| [`model.py`](model.py.doc.md) | `ImageModel` — model phân loại/encode cấp cao | ✅ |
| [`encoder.py`](encoder.py.doc.md) | `ImageEncoder`, `MultiImageEncoder` — backbone + pooling | ✅ |
| [`modules.py`](modules.py.doc.md) | `MLP`, `MultiTaskModel` và các head phụ | 🟡 |
| [`resnet.py`](resnet.py.doc.md) | ResNet backbone có dilation | ✅ |
| [`transformer.py`](transformer.py.doc.md) (266) | Transformer cho temporal/multi-image path | 🟡 |
| [`pretrained.py`](pretrained.py.doc.md) | Tải weight `biovil_t_image_model_proj_size_128.pt` | ✅ |
| [`types.py`](types.py.doc.md) | `ImageEncoderType`, kiểu dữ liệu | ✅ |
| [`device.py`](device.py.doc.md) | Helper chọn device | ✅ |
| `__init__.py` | | ✅ |
| `__pycache__/` | ⚠ **bị git track** — build artifact Py3.7/3.8 lọt vào repo | — |

## Main responsibilities

1. Nạp weight BioViL-T đã pretrain (tải về nếu chưa có).
2. Encode ảnh X-quang thành patch embedding 1408 chiều.
3. Cung cấp `projected_patch_embeddings` — thuộc tính mà `blip2_qformer` đọc.

## Entry points

Không có. Thư viện.

## Dependencies

`torch`, `torchvision` (ResNet), và một cơ chế tải weight qua HTTP
(`_download_biovil_t_image_model_weights`).

## Used by

| Ai | Dùng gì |
|---|---|
| `model/lavis/models/blip2_models/blip2.py:20-22, :83` | `ImageModel`, `get_biovil_t_image_encoder`, `ImageEncoderType` |
| `Blip2Qformer.__init__` (gián tiếp) | qua `self.init_vision_encoder("biovil", …)` |
| `Blip2Qformer._encode_image_streams:500` | `self.visual_encoder(image).projected_patch_embeddings` |
| `Blip2Qformer._encode_aux_streams:437` | như trên, dưới `torch.no_grad()` |

## Execution flow

```text
Blip2Qformer.__init__
   ↓
Blip2Base.init_vision_encoder("biovil", img_size, drop_path_rate, ...)
   ↓
biovil_t.pretrained.get_biovil_t_image_encoder()
   ↓
_download_biovil_t_image_model_weights()   → biovil_t_image_model_proj_size_128.pt
   ↓
ImageModel(...)  →  self.visual_encoder
   ↓
freeze_vit=true → requires_grad=False, .eval(), train = disabled_train
```

⚠ `visual_encoder.train` bị thay bằng `disabled_train` — nên `model.train()` ở
cấp trên **không** bật lại chế độ train cho encoder. Đây là chủ ý, và là lý do
encoder chắc chắn đóng băng.

## Important configurations

| Key | Ảnh hưởng |
|---|---|
| `model.encoders.biovil` | Bật/tắt luồng này |
| `model.vit_model: biovil` | Fallback quyết định `use_biovil` khi `encoders` không nêu |
| `model.freeze_vit` (mặc định `true`) | Đóng băng + `disabled_train` |
| `model.image_size: 448` | Kích thước ảnh vào |
| `model.drop_path_rate`, `model.vit_precision` | Truyền vào `init_vision_encoder` |

## Status

```text
✅ ACTIVE — luôn bật ở mọi config production
```

## Notes

- ⚠ **Weight được tải qua mạng lần chạy đầu.** Máy không có internet sẽ hỏng ở
  bước khởi tạo model, không phải lúc train.

- **Chiều 1408 là hằng số neo cả kiến trúc.** `VISUAL_DIM = 1408`
  (`blip2_qformer.py:33`) đến từ đây. Đổi encoder khác chiều sẽ kéo theo
  `SharedVisualTokenProjector`, `MHCAC.visual_dim`, và `init_Qformer(…, vis_num_feat, …)`.

- **`.pyc` bị track** (11 file, Python 3.7/3.8). Ghi nhận là
  [I1](../_meta/LEGACY_AND_OPTIONAL.md#-potential-issues--ghi-nhận-không-sửa), không sửa.

- **Đừng sửa `vision_encoders/biovil_t/`** khi định sửa BioViL-T. Sửa thư mục này.

## Related documentation

- [ARCHITECTURE.md §2.1](../_meta/ARCHITECTURE.md#21-vision-encoders--đóng-băng)
- [`vision_encoders/_index.md`](../vision_encoders/_index.md) — cảnh báo bản sao
- [GLOSSARY: BioViL-T](../_meta/GLOSSARY.md#biovil-t)

← [Về HOME](../../HOME.md)
