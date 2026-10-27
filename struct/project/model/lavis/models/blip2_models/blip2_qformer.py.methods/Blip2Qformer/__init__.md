> Source: `model/lavis/models/blip2_models/blip2_qformer.py:138-429`
> Status: ✅ ACTIVE

# `Blip2Qformer.__init__(...)`

> ⚠⚠ **D-023 (2026-09-29) — ba lớp, không bao giờ nhị phân.** Các tham số gate / mention-conditioned / `uncertain_policy` đã bị XÓA; `cls_loss_fn = ClassificationLoss(class_weights=..., label_smoothing=...)`. Xem [D-023](../../../../../../_meta/DECISIONS.md#d-023--ba-lớp-không-bao-giờ-nhị-phân-gỡ-toàn-bộ-khung-nhị-phân).


## Located in

[`blip2_qformer.py`](../../blip2_qformer.py.doc.md)

## Purpose
Dựng 6 khối con theo đúng thứ tự phụ thuộc, và đăng ký ITC queue.

## Signature
43 tham số ngoài `self`. Không gọi trực tiếp — dùng [`from_config`](../from_config.md).

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
| `lambda_*` (11 objective cũ) | xem code | |
| `qformer_input_norm` | `None` | 2026-10-15: `"per_stream"` dựng `QformerInputNorm` NGAY SAU `shared_visual_projector` |
| `lambda_query_diversity` | `0.0` | 2026-10-15: > 0 cần một trong `lambda_itc/itm/lm` > 0, và (lần 2) `qformer_dropout: 0.0` + `feature_mask_ratio: 0.0` (`assert_noise_free_query_path`), không thì lỗi |
| `eval_mode_modules` | `()` | 2026-10-15: kiểm ở cuối `__init__` (mỗi tên phải là submodule); `train()` giữ chúng ở eval |
| `qformer_dropout` | `None` | 2026-10-15 (lần 2): `None` = dropout 0,1 của BERT; số trong [0, 1) đặt `p` cho mọi lớp dropout của Q-Former ngay sau khi dựng, module vẫn ở train (giữ gradient checkpointing) |
| `lambda_explanation` | `0.0` | Gate cứng: zero không dựng `ExplanationLoss` |
| `explanation_cfg` | `None` | `top_k`, warmup và danh sách stream |
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
15 AbnormalityClassificationModel(embed_dim=768, ..., num_commmon_tokens=mhcac_num_common_tokens, visual_dim=1408)
   # mhcac_num_common_tokens <- model.mhcac.num_common_tokens (from_config, mặc định 14, prod 8; D-026)
16 ClassificationLoss(class_weights, label_smoothing, uncertain_policy)
17 IF lambda_explanation > 0: ExplanationLoss(top_k); current_epoch = 0
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

**Bước 17 — gate cứng.** `explanation_loss_fn=None` khi lambda bằng 0. Config
explanation vẫn có default an toàn nhưng không kéo theo capture hoặc autograd.

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

## `view_fusion.mode` — 2026-10-26

`resolve_view_fusion_mode` (mhcac/late_view_pool.py) đọc `view_fusion_cfg["mode"]`
(vắng = `early`) và từ chối `late` khi `multi_view: false`, `lambda_mpc > 0`,
`lambda_view_consistency > 0` hoặc explanation loss bật. `late` dựng
`self.late_view_pool = LateViewPooling(14, 3, num_view_types)` và KHÔNG dựng
`view_fusion`, `stream_adapters`, `mpc_heads`. `early` dựng như trước, không tiêu thêm
RNG. `late_anchor_weight` (0.5) cũng được pop khỏi `view_fusion_cfg`, và
`late_detach_aux` (vắng = `False`, 2026-10-26) — bật mà không ở `late` thì `ValueError`
(early fusion có `detach_aux` riêng). Plan: `docs/handoff/PLAN-2026-10-26-late-fusion-detach-aux.md`.
`late_aux_head` (vắng = `False`, 2026-10-27) — bật mà không ở `late`, hoặc cùng `late_detach_aux`, thì
`ValueError`; bật thì ngay sau `self.mhcac` dựng `self.mhcac_aux = copy.deepcopy(self.mhcac)` (tắt: `None`,
không tiêu RNG, state dict không đổi). Plan: `docs/handoff/PLAN-2026-10-27-late-aux-head.md`.
`view_fusion.mode: tokens` (2026-10-27): `self.aux_tokens_on = multi_view and mode == "tokens"`; không dựng
module fusion nào (nhánh `elif self.aux_tokens_on:` chỉ log), và MHCAC được dựng với
`aux_view_tokens=self.aux_tokens_on, num_view_types=view_fusion.num_view_types` — tắt thì MHCAC không
có tham số mới và không tiêu RNG. `late_detach_aux` / `late_aux_head` vẫn đòi `late`, nên bị từ chối ở
`tokens`. Plan: `docs/handoff/PLAN-2026-10-27-aux-tokens-mhcac.md`.
