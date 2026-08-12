> Source: `model/lavis/models/blip2_models/blip2_qformer.py` (1.497 dòng)
> Status: ✅ ACTIVE — ★ TRUNG TÂM STAGE 1
> Last verified against source: 2026-08-12

# `blip2_qformer.py`

## Purpose

**Toàn bộ Stage 1 nằm trong một class: `Blip2Qformer`.** Encoder, view fusion,
projection, MHCAC, Q-Former, và tổng hợp 11 loss — tất cả ở đây.

Nếu bạn chỉ đọc được một file trong repo, đọc file này.

## Why it exists

Đây là bản fork của `blip2_qformer.py` gốc LAVIS, sửa để: (a) dùng nhiều encoder
y khoa đóng băng thay vì một ViT, (b) nối thêm nhánh phân loại MHCAC song song với
Q-Former, (c) hỗ trợ multi-view, (d) thêm ITC negative queue cho microbatch nhỏ,
và (e) ablation encoder **chỉ ở inference** tại ranh giới shared-token.

## Role in architecture

```text
batch ──► Blip2Qformer.forward() ──► BlipOutput(loss, classification_logits, …)
                    │
        ┌───────────┼───────────┐
     encoders   ViewFusion   SharedProjector
                    │
            ┌───────┴───────┐
          MHCAC        Q-Former
```

## Status

```text
✅ ACTIVE
```

## Used in

Training ✅ · Validation ✅ · Test ✅ · Inference ✅ (qua `generate()` và Stage 2 Q-Former)

## Entry point

Không. Được `registry` dựng qua `task.build_model(cfg)`.

## Inputs

`samples` dict từ `MIMIC_CXR_Dataset.collater`:

| Key | Shape | Bắt buộc |
|---|---|---|
| `image` | `[B,3,448,448]` | ✅ (trừ khi có `<enc>_feat`) |
| `text_output` | `list[str]` | ✅ |
| `classification_labels` | `[B,14]` long | ✅ |
| `classification_mask`, `generation_mask` | `[B]` bool | ✅ |
| `aux_image`, `aux_mask`, `aux_view_ids`, `anchor_view_id` | multi-view | 🟡 |
| `<enc>_feat`, `aux_<enc>_feat` | feature cache | 🟡 |

## Outputs

`BlipOutput` với `loss` + 11 thành phần loss + `classification_logits [B,14,3]`
+ `classification_mask`.

## Important imports

```python
from mhcac.mhcac_12 import AbnormalityClassificationModel     # :23
from vision_encoders.pubmedclip.pubmed_clip import Pubmedclip # :26
from vision_encoders.swin.swin_encoder import SwinEncoder     # :27
from vision_encoders.rad_dino.rad_dino_encoder import RadDinoEncoder  # :28
from vision_encoders.shared_visual_tokens import SharedVisualTokenProjector  # :29
# from vision_encoders.medclip.medclip import Medclip         # :30  ← LEGACY, đã comment
from mhcac.loss import (ClassificationLoss, MultiPositiveContrastiveLoss,
                        soft_target_kl_loss, view_consistency_loss)  # :35
from mhcac.view_fusion import ViewFusionModule                # :41
```

`VISUAL_DIM = 1408` (`:33`) — hằng số neo cả kiến trúc.
`chexpert_cols` (`:43`) — 14 tên bệnh lý, khớp `stage2/prompts/ontology.py`.

## Important configuration

Toàn bộ đọc trong [`from_config`](blip2_qformer.py.methods/from_config.md).
Khối `model:` của run YAML: `encoders.*`, `swin.*`, `raddino.*`, `multi_view`,
`view_fusion.*`, `loss.lambda_*`, `loss.itc_queue_size`, `mhcac.*`,
`num_query_token`, `cross_attention_freq`, `max_txt_len`, `freeze_vit`.

⚠ `from_config` **âm thầm bỏ qua block config lạ**. Comment `:1367` ghi rõ: các key
multi-view "phải được đọc tường minh để có hiệu lực". Thêm key mới mà quên thêm
dòng đọc ở đây = key đó không làm gì.

## Main classes

### `Blip2Qformer(Blip2Base)` — [📄 chi tiết](blip2_qformer.py.methods/Blip2Qformer/_index.md)

Đăng ký hai tên: `@registry.register_model("blip2")` và `"blip2_feature_extractor"`.

## Main functions

| Hàm | Doc | Vai trò |
|---|---|---|
| `_resolve_encoder_ablation` | [📄](blip2_qformer.py.methods/_resolve_encoder_ablation.md) | `active_encoders` → các stream cần zero; fail nếu tên lạ |
| `_hard_negative_sampling_weights` | [📄](blip2_qformer.py.methods/_hard_negative_sampling_weights.md) | ★ Trọng số hard negative, FP32, loại positive **trước** softmax |

## Main methods

| Method | Doc | Vai trò |
|---|---|---|
| `__init__` | [📄](blip2_qformer.py.methods/Blip2Qformer/__init__.md) | Dựng 6 khối con + đăng ký ITC queue |
| `forward` | [📄](blip2_qformer.py.methods/Blip2Qformer/forward.md) | ★ Trái tim Stage 1 |
| `_encode_image_streams` | [📄](blip2_qformer.py.methods/Blip2Qformer/_encode_image_streams.md) | ★ Encoder → fusion → shared projector |
| `_encode_aux_streams` | [📄](blip2_qformer.py.methods/Blip2Qformer/_encode_aux_streams.md) | Encode auxiliary view, batched, no_grad |
| `_apply_encoder_ablation` | [📄](blip2_qformer.py.methods/Blip2Qformer/_apply_encoder_ablation.md) | Zero span encoder sau projection; chỉ hợp lệ khi `eval()` |
| `_image_text_contrastive` | [📄](blip2_qformer.py.methods/Blip2Qformer/_image_text_contrastive.md) | ITC + queue |
| `_image_text_matching` | [📄](blip2_qformer.py.methods/Blip2Qformer/_image_text_matching.md) | ITM + hard negative |
| `_language_modeling` | [📄](blip2_qformer.py.methods/Blip2Qformer/_language_modeling.md) | LM |
| `_update_itc_queue` | [📄](blip2_qformer.py.methods/Blip2Qformer/_update_itc_queue.md) | Cập nhật ring buffer |
| `from_config` | [📄](blip2_qformer.py.methods/from_config.md) | ★ YAML → tham số |
| `generate` | [📄](blip2_qformer.py.methods/Blip2Qformer/generate.md) | Sinh caption (không dùng ở Stage 1 training) |
| `forward_image` | [📄](blip2_qformer.py.methods/Blip2Qformer/forward_image.md) | Trích Q-Former image embeddings cho downstream |

### Trivial helpers

| Method | Vai trò |
|---|---|
| `_create_mask` | Mask ngẫu nhiên patch — ⚠ không thấy caller trong forward |
| `_stash_prefusion` | Giữ tensor pre-fusion khi loss phụ cần |
| `_fuse` | Wrapper gọi `ViewFusionModule[name]` |
| `_gather_with_local_grad` | All-gather giữ grad ở rank cục bộ |
| `_batch_mask` | Đọc mask từ samples, có fallback |
| `initialize_expert_tokens` | Khởi tạo expert token từ embedding tên bệnh lý |
| `compute_sim_matrix` | Ma trận similarity i2t/t2i |

## Execution flow

Xem [CALL_GRAPH.md §2](../../../../_meta/CALL_GRAPH.md#2-blip2qformerforward--trái-tim-stage-1).

## Calls

`mhcac.mhcac_12.AbnormalityClassificationModel` (×3 mỗi forward) ·
`mhcac.loss.*` · `mhcac.view_fusion.ViewFusionModule` ·
`vision_encoders.*` · `biovil_t.*` (qua `Blip2Base.init_vision_encoder`) ·
`Qformer.bert` · `concat_all_gather` (`models/base_model.py`)

## Called by

`model/lavis/tasks/base_task.py:train_step` → `model(samples)` ·
`ImageTextPretrainTask.evaluation` · `training/stage1/lavis_loader.build_stage1_model` ·
`inference.py` (qua `init_blip`) · `pretraining/precompute_features.py`

## Data flow

Xem [DATA_FLOW.md §2.4](../../../../_meta/DATA_FLOW.md#24-trong-model).

## Side effects

| | |
|---|---|
| Buffer | `itc_image_queue`, `itc_text_queue`, `itc_queue_ptr`, `itc_queue_filled` — **mutate mỗi forward khi training** |
| Instance state | `_last_prefusion_streams`, `_last_raddino_patches` — reset đầu mỗi `_encode_image_streams` |
| Mạng | Tải BLIP-2 pretrained + weight encoder lần đầu |
| GPU | Toàn bộ model |

⚠ Buffer ITC đăng ký `persistent=False` → **không** vào `state_dict`, nên checkpoint
không mang queue. Resume bắt đầu với queue rỗng.

## Error / edge cases

| Tình huống | Hành vi |
|---|---|
| Không encoder nào bật | `ValueError` (`:174`) |
| `active_encoders` chứa stream checkpoint không có | `ValueError`; không bỏ qua âm thầm |
| Bật ablation khi model đang training | `RuntimeError`; đường này inference-only |
| `raw_streams` rỗng | `ValueError("No image encoder stream is enabled.")` (`:539`) |
| `aux_image=None` mà cần encode aux | `ValueError` nêu tên stream thiếu (`:428`) |
| `itc_queue_size < 0` | `ValueError` (`:229`) |
| Hàng hard-negative không còn candidate | `ValueError` (`:87`) |
| `similarities` không phải 2-D | `ValueError` (`:69`) |
| Batch không có mẫu hợp lệ | Trả loss `0.0` giữ đồ thị (`zero = x.sum()*0.0`), **không** crash |

Pattern `zero = tensor.sum() * 0.0` xuất hiện nhiều lần — nó tạo một loss bằng 0
**vẫn nối vào đồ thị autograd**, để DDP không treo vì tham số không nhận gradient.

## Related tests

`tests/test_stage1_objectives.py` (teacher-only, shape student) ·
`tests/test_encoder_ablation.py` (span, fail-closed, cấm trong train) ·
`tests/test_blip2_negative_sampling.py` ⚠ cần torchvision ·
`tests/test_shared_visual_tokens.py` · `tests/test_view_fusion.py`

## Related documentation

[ARCHITECTURE.md §2](../../../../_meta/ARCHITECTURE.md#2-stage-1--chi-tiết-từng-khối) ·
[`mhcac/_index.md`](../../../../mhcac/_index.md) ·
[`vision_encoders/_index.md`](../../../../vision_encoders/_index.md)

## Developer notes

1. **MHCAC chạy 3 lần mỗi forward** (student `:907`, teacher `:925`, anchor `:965`).
   Đây là chi phí lớn nhất sau encoder.
2. **`from_config` bỏ qua key lạ.** Thêm config mới phải thêm dòng đọc tường minh.
3. **`ln_vision` chỉ áp cho biovil** và nằm phía encoder của merge — nó là chuẩn
   hóa, không phải chiếu chiều.
4. **PubMedCLIP dựng với `project=False`** — projector chung sở hữu phép chiếu.
5. `:202` re-link `Qformer.cls.predictions.bias` sang `decoder.bias` — vá cho
   transformers ≥4.50 thay decoder khi resize mà để lại bias alias sai vocab size.
6. **Sửa file này ảnh hưởng:** mọi Stage 1 run, Stage 2 mode Q-Former, `inference.py`,
   `precompute_features.py`.

## Source relationships

- **Parent:** [`model/lavis/_index.md`](../../_index.md)
- **Methods:** [`blip2_qformer.py.methods/`](blip2_qformer.py.methods/)
- **Related:** [`Qformer.py`](Qformer.py.doc.md) · [`ReportDataset.py`](../../data/ReportDataset.py.doc.md) · [`runner_base.py`](../../runners/runner_base.py.doc.md)

← [HOME](../../../../../HOME.md)
