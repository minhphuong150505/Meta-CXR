> Source: `model/lavis/models/blip2_models/blip2_qformer.py:120-1705`
> Status: ✅ ACTIVE

# `class Blip2Qformer(Blip2Base)`

## Located in

[`blip2_qformer.py`](../../blip2_qformer.py.doc.md)

## Responsibility
Toàn bộ Stage 1 trong một class: 3 encoder đóng băng, view fusion, shared
projector, MHCAC, Q-Former, và tổng hợp 12 loss.

## Inheritance
`Blip2Base` (`model/lavis/models/blip2_models/blip2.py`) — cung cấp `init_tokenizer`,
`init_Qformer`, `init_vision_encoder`, `load_checkpoint_from_config`, `disabled_train`.

Đăng ký hai tên registry: `"blip2"` và `"blip2_feature_extractor"`.

## Constructor
`__init__` nhận **43 tham số ngoài self**; xem [trang riêng](__init__.md). Dựng bởi
[`from_config`](../from_config.md), không bao giờ gọi trực tiếp.

## Sub-module (thứ trainable)
| Attribute | Loại | Trainable? |
|---|---|---|
| `visual_encoder` + `ln_vision` | BioViL-T | ❌ đóng băng (+ `disabled_train`) |
| `pubmedclip`, `swin`, `raddino` | encoder | ❌ `.eval()` |
| `view_fusion` | `ModuleDict` (1/encoder) | ✅ |
| `shared_visual_projector` | `SharedVisualTokenProjector` | ✅ |
| `mhcac` | `AbnormalityClassificationModel` | ✅ |
| `Qformer` + `query_tokens` | Q-Former | ✅ |
| `vision_proj`, `text_proj`, `itm_head`, `temp` | head | ✅ |
| `cls_loss_fn`, `mpc_loss_fn` | loss | — |
| `explanation_loss_fn` | `ExplanationLoss` hoặc `None` | —; chỉ dựng khi lambda > 0 |

## Buffer (không persistent)
`itc_image_queue [1024,32,256]` fp16 · `itc_text_queue [1024,256]` fp16 ·
`itc_queue_ptr` · `itc_queue_filled`

⚠ `persistent=False` → **không vào `state_dict`**. Checkpoint không mang queue;
resume bắt đầu với queue rỗng.

## Instance state tạm
`_last_prefusion_streams` (reset mỗi `_encode_image_streams`) ·
`_last_raddino_patches` · `_keep_prefusion` (bool cố định lúc `__init__`)

`current_epoch` điều khiển warmup; `explanation_streams` giới hạn stream được giám
sát. Activation CAM chỉ sống trong local của `forward`, không được giữ sau return.

## Lifecycle
```text
from_config(cfg) → __init__ → load_checkpoint_from_config
   ↓
set_epoch(epoch) → current_epoch
   ↓ (mỗi batch)
forward(samples) → BlipOutput
   ↓ (inference/Stage 2)
generate(samples) hoặc lấy Q-Former output
```

## Public methods
[`forward`](forward.md) ★ · [`set_epoch`](set_epoch.md) · `generate` ·
`compute_sim_matrix` · `initialize_expert_tokens`

## Private methods quan trọng
[`_encode_image_streams`](_encode_image_streams.md) ★ ·
[`_native_stream_layouts`](_native_stream_layouts.md) ★ ·
[`_encode_aux_streams`](_encode_aux_streams.md) ·
[`_image_text_contrastive`](_image_text_contrastive.md) ·
[`_image_text_matching`](_image_text_matching.md) ·
[`_language_modeling`](_language_modeling.md) ·
[`_update_itc_queue`](_update_itc_queue.md)

## Dependencies
`mhcac.mhcac_12`, `mhcac.explanation`, `mhcac.loss`, `mhcac.view_fusion`, `vision_encoders.*`,
`biovil_t.*`, `Qformer`, `blip_outputs.BlipOutput`

## Callers
`task.build_model` · `BaseTask.train_step` · `ImageTextPretrainTask.evaluation` ·
`lavis_loader.build_stage1_model` · `inference.py` · `precompute_features.py`
