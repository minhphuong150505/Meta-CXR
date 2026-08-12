> Source: `model/lavis/models/blip2_models/blip2_qformer.py:137-404`
> Status: ✅ ACTIVE

# `Blip2Qformer.__init__(...)`

## Located in

[`blip2_qformer.py`](../../blip2_qformer.py.doc.md)

## Purpose
Dựng 6 khối con theo đúng thứ tự phụ thuộc, và đăng ký ITC queue.

## Signature
44 tham số keyword. Không gọi trực tiếp — dùng [`from_config`](../from_config.md).

## Parameters đáng chú ý
| Tham số | Default | Ghi chú |
|---|---|---|
| `use_biovil` / `use_pubmedclip` | `True` | |
| `use_swin` / `use_raddino` | `False` | ⚠ prod bật swin, tắt raddino |
| `num_query_token` | 32 | |
| `cross_attention_freq` | 2 | |
| `embed_dim` | 256 | Chiều ITC projection |
| `max_txt_len` | 32 | ⚠ prod đặt 256 |
| `freeze_vit` | `True` | |
| `multi_view` | `False` | ⚠ prod `True` |
| `lambda_*` (11 cái) | xem code | |
| `itc_queue_size` | 1024 | |
| `class_weights` | `None` → default 14×3 | `[]` → tắt weighting |
| `uncertain_policy` | `"three_class"` | ⚠ prod `ignore_uncertain` |

## Execution flow — thứ tự có ý nghĩa
```text
1  init_tokenizer()
2  validate: ít nhất 1 encoder bật              → ValueError (:174)
3  init_vision_encoder("biovil") → vis_num_feat
4  freeze_vit → requires_grad=False, .eval(), train = disabled_train   (:186-191)
5  init_Qformer(num_query_token, vis_num_feat, cross_attention_freq)
6  Qformer.resize_token_embeddings(len(tokenizer), mean_resizing=False)
7  ★ cls.predictions.bias = cls.predictions.decoder.bias    (:202)  ← vá transformers≥4.50
8  copy trọng số "_query" từ tên gốc                        (:204-207)
9  vision_proj / text_proj / itm_head / temp
10 validate itc_queue_size >= 0                             → ValueError (:229)
11 register_buffer × 4  (persistent=False)
12 Pubmedclip(project=False) / SwinEncoder / RadDinoEncoder  (.eval())
13 IF multi_view: ViewFusionModule cho từng encoder + MultiPositiveContrastiveLoss
14 SharedVisualTokenProjector(shared_stream_dims, VISUAL_DIM)
15 AbnormalityClassificationModel(embed_dim=768, ..., visual_dim=1408)
16 ClassificationLoss(class_weights, label_smoothing, uncertain_policy)
```

## Detailed logic
**Bước 4 — `train = disabled_train`.** Thay method, không chỉ đặt `.eval()`. Nên
`model.train()` ở cấp trên **không** bật lại encoder. Đây là bảo đảm cứng rằng
encoder đóng băng.

**Bước 7 — vá bias.** Comment `:198-201`: transformers ≥4.50 thay decoder khi
resize nhưng để lại bias alias ở vocab size cũ. Không vá → checkpoint BLIP-2
30.523 token load lỗi shape.

**Bước 8 — trọng số `_query`.** Nhánh query khởi tạo từ trọng số text rồi tách ra
học riêng.

**Bước 11 — `persistent=False`.** Queue ~16 MB, không vào checkpoint.

**Bước 13 — một `ViewFusionModule` cho mỗi encoder**, vì mỗi encoder có `D` khác.
`vf_cfg.pop("dim_source")` (`:305`) bỏ key không phải tham số constructor.

## Side effects
Cấp phát toàn bộ model; tải weight encoder (mạng lần đầu).

## Error handling
Không encoder nào bật → `ValueError` (`:174`) · `itc_queue_size < 0` → `ValueError` (`:229`)

## Modification risk
| Sửa | Ảnh hưởng |
|---|---|
| Bỏ vá bias `:202` | Checkpoint BLIP-2 không load |
| Bỏ `disabled_train` | `model.train()` bật lại encoder → encoder học, phá giả định |
| Đổi `VISUAL_DIM` | Kéo theo projector, MHCAC, Q-Former, và mọi checkpoint cũ |
| Đổi `default_class_weights` | Đổi cân bằng loss; giá trị hiện tại tính từ cohort train thật |
