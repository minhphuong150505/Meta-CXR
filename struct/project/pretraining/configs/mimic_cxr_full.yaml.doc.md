> Source: `pretraining/configs/mimic_cxr_full.yaml` (config)
> Status: ✅ ACTIVE — ★ PRODUCTION, recipe Stage-1 duy nhất
> Last verified against source: 2026-08-17

# `pretraining/configs/mimic_cxr_full.yaml`

## Purpose
**Recipe Stage 1 production**: full MIMIC-CXR p10–p19, một GPU; recipe
classification thuần.

⚠ **Explanation-aware loss đã TẮT từ 2026-08-17** — `lambda_explanation` và
`lambda_explanation_strong` đều `0.0`, theo quyết định của user sau A/B có kiểm
soát (xem [D-017](../../_meta/DECISIONS.md#d-017--dừng-explanation-aware-loss-trong-production)).
Trước đó tài liệu này ghi "explanation-aware loss hai tầng được bật từ Giai đoạn
2"; câu đó không còn đúng.

## ⚠ Default trong file không phải setting đã chạy
Đổi tên từ `mimic_cxr_full_l4.yaml` ngày 2026-08-13 (tên cũ nói NVIDIA L4 nhưng
con số memory trong file đến từ RTX 5060 Ti — máy train thật).

File ghi `batch_size_train: 8` / `accum_grad_iters: 8`, đã verify vừa 16 GB
(*"batch 8 peaked at 12,755 MB"*). Nhưng run 10-epoch tạo ra `checkpoint_best`
hiện tại chạy với override CLI `run.batch_size_train=6 run.batch_size_eval=6
run.accum_grad_iters=11`. Khi tái lập kết quả phải truyền lại ba override đó.

## Khối `model:`
| Key | Giá trị | Ghi chú |
|---|---|---|
| `arch` / `model_type` | `blip2` / `pretrain` | Registry lookup |
| `pretrained` | URL BLIP-2 | Tải lần đầu |
| `freeze_vit` | `true` | + `disabled_train` |
| `encoders` | biovil ✅ pubmedclip ✅ **swin ❌** raddino ❌ | Swin tắt 2026-08-14. Đổi **kiến trúc**: projector nhận 2 stream, MHCAC thấy 246 token (biovil 196 + pubmedclip 1 CLS + 49). Bật lại swin còn kéo MHCAC về nhánh legacy resample-về-49, vì `_native_stream_layouts` không mô tả được số token của swin. Swin cũng là encoder đắt nhất: ~220 ms trong bước ~520 ms và ~1,6 GB — SwinV2 ở 448 có 112×112 = 12.544 token ở stage đầu → checkpoint không chuyển qua lại được |
| `num_query_token` / `cross_attention_freq` | 32 / 2 | |
| `max_txt_len` | 256 | |
| `mhcac.uncertain_policy` | `ignore_uncertain` | Cặp mơ hồ không phải target đáng tin |
| `mhcac.label_smoothing` | 0.05 | |
| `multi_view` | **`true`** | |
| `view_fusion.p_view_drop` | 0.15 | |
| `data.study_sampling` | `true` | ★ Một dòng = một study |
| `data.anchor_priority` | `[PA, AP, lateral]` | |
| `data.max_aux_views` | 1 | |
| `explanation.mask_cache_dir` | private mask cache | Dataset đọc JSON + memmap theo split |

⚠ **`data:` nằm TRONG `model:`** — `Config` chỉ merge `run`/`model`/`datasets`.

## Trọng số loss

`itc`/`itm`/`lm` = 0 · `cls` = 1.0 · `teacher_cls`/`distill` = 0 ·
`mhcac_contrastive` = 0.3 · `orthogonality` = 0.7 · `sparsity` = 0.3 ·
`mpc` = 0.1 · `view_consistency` = 0.05 · **`explanation` = 0.0 /
`explanation_strong` = 0.0 (TẮT)** · `itc_queue_size` = 1024.

Hai `lambda_explanation*` là gate duy nhất. Config **không có**
`explanation.enabled`, vì `Blip2Qformer.from_config` không đọc cờ đó và hai gate
có thể mâu thuẫn. Chỉ cần một trong hai `> 0` là module bật
(`blip2_qformer.py:313`); **cả hai bằng 0 tắt hoàn toàn** explanation
module/Grad-CAM/CAM capture ở model — đây là trạng thái hiện tại.

## Khối `model.explanation`

| Key | Giá trị | Nghĩa |
|---|---|---|
| `top_k` | 0.5 | Giữ top 50% CAM mềm |
| `warmup_start_epoch` | 2 | Epoch [0]–[1] có λ=0 |
| `warmup_epochs` | 2 | Epoch [2]–[3] ramp; [4]+ λ=0.25 |
| `streams` | biovil, pubmedclip, swin | `swin` còn trong danh sách dù encoder đã tắt — đây là **bộ lọc** trên các stream thực sự được capture, tên không được sinh ra thì bị bỏ qua. Bật lại encoder không cần sửa thêm |
| `mask_cache_dir` | private path | `ReportDataset` cần đủ cache train/val/test; đặt `""` là cách chạy nhánh đối chứng không explanation loss |

Grad-CAM giờ chỉ còn **hai** độ phân giải: BioViL 14×14 và PubMedCLIP 7×7.

## Khối `run:` — các key hay bị hiểu sai
| Key | Giá trị | Nghĩa thật |
| `batch_size_train` / `accum_grad_iters` | **16 / 4** | Effective batch 64. Đo 2026-08-14, hai encoder, explanation loss bật, 200 vòng mỗi cấu hình: batch 6 → 0,04647 s/mẫu (3.952 MiB torch), **16 → 0,04093 (4.415)**, 24 → 0,04016 (4.889), 32 → 0,03989 (5.414). **Batch size không phải đòn bẩy tốc độ**: 5,3× batch chỉ được 14,2% và bão hoà ở 16; `data: 0.0000` ở mọi cấu hình nên cũng không phải dataloader; GPU util ~37% bất kể batch, **nguyên nhân chưa xác định**. Bộ nhớ 91,5% là tĩnh (56,2 MiB/mẫu trên nền 3.615 MiB) nên batch 32 vẫn chỉ dùng 39% card. Chọn 16 vì lý do mô hình: `lambda_mpc` mang ~25% tổng loss nhưng MPC lấy negative trong microbatch và chỉ ~55% study có view phụ — batch 6 cho ~3,3 study dùng được, 16 cho ~8,8 |
| `image_size` | **448** | Thêm 2026-08-14. Phải bằng `vis_processor.*.image_size`. Trước đó model config thừa hưởng `224` từ YAML BLIP-2 gốc và **không ai đọc** (`init_vision_encoder` bỏ qua nó với biovil). Nay nó quyết định lưới token của BioViL: 448/32 → 14×14 → 196 token. Sai giá trị này thì `AbnormalityClassificationModel.forward` raise ngay batch đầu, không train sai âm thầm |
| `num_workers` | **8** (giảm từ 12 ngày 2026-08-18) | Máy có 12 nhân thật / 20 luồng. Con số so sánh 4-vs-12 worker cũ (0.305 so với 0.253 s/it) đo khi `remap_to_uint8` còn nghẽn băng thông bộ nhớ — nó đo mức tranh chấp DRAM chứ không đo năng lực worker. Sau bản vá, loader cấp **92,9 study/s** trong khi GPU chỉ tiêu thụ ~68, và `next(it)` chiếm **0,1%** wall. Tăng thêm worker hiện không mang lại gì. Xem [`ReportDataset.py`](../../model/lavis/data/ReportDataset.py.doc.md) |
| `lambda_explanation` / `lambda_explanation_strong` | **0.0 / 0.0 (TẮT từ 2026-08-17)** | Tắt sau A/B 5 epoch có kiểm soát, cùng seed/manifest/recipe, chỉ khác hai lambda. Test split, ngưỡng calibrate trên val, `ignore_uncertain`: positive_macro_f1 0.8757 (ON) vs **0.8767** (OFF), macro_auroc 0.7850 vs **0.7879**, macro_specificity 0.3840 vs **0.4127**, wall 5:06:44 vs **4:25:27** (**+15,5%** cho nhánh ON). Mọi CI 95% chồng nhau, mọi chênh lệch nghiêng về OFF. Term cũng **không đạt mục tiêu riêng của nó**: đối chiếu với baseline diện tích mask (lung 0.3301, bbox 0.2366 — mức một CAM ngẫu nhiên ghi được), saliency precision ở mức ngẫu nhiên trong **cả hai** nhánh, và PubMedCLIP còn tệ đi ~0,026. Nguyên nhân là encoder đóng băng: term chỉ đánh lại trọng số kênh của feature map cố định. Xem [D-017](../../_meta/DECISIONS.md#d-017--dừng-explanation-aware-loss-trong-production). Chi phí đo lúc còn bật (bản một-term gộp): **+10,5% wall, +174 MiB**
|---|---|---|
| `selection_metric` | **`loss`** | Tổng val loss; `selection_mode` cố ý vắng để RunnerBase tự suy ra `min`. Thiên lệch với nhãn hiếm — xem DECISIONS.md |
| `warmup_steps` | 800 | **Optimizer update**, không phải microbatch |
| `min_lr` | **2.0e-5** (tăng từ 1.0e-5 ngày 2026-08-18) | Sàn của cosine decay, tức LR mà **đuôi** của run thật sự train ở đó. Đổi theo yêu cầu của user vì các epoch cuối tiến chậm. Đây là knob duy nhất dịch được phần cuối mà gần như không đụng phần đầu: tính trên lịch đang ship (10 epoch, ~3.480 optimizer update/epoch ở effective batch 64, warmup 800), LR tại mốc mỗi epoch đổi **[1] 1,00× · [4] 1,05× · [7] 1,27× · [8] 1,47× · [9] 1,79× · cuối 2,00×**. Tuyệt đối: epoch [8] khởi 2,80e-5 thay vì 1,90e-5, epoch [9] 2,21e-5 thay vì 1,23e-5. Scheduler **stateless** (`cosine_lr_schedule_steps` tính lại từ `(epoch, step)` mỗi update, không lưu gì vào checkpoint) nên resume từ `checkpoint_last` ăn ngay giá trị mới, moment của optimizer vẫn nguyên. ⚠ **Chưa chạy GPU ở giá trị này** |
| `save_freq` | 5 | Checkpoint theo epoch mỗi 5 epoch |
| `batch_size_train` × `accum_grad_iters` | 8 × 8 | Effective batch = **64** |
| `max_epoch` / `early_stop_patience` | **10 / 5** | ⚠ 5+5=10 > index cuối [9] → early stop **không thể kích hoạt**; đặt patience ≤ 4 nếu muốn nó sống |
| `eval_start_epoch` | **5** | Epoch [0]–[4] train nhưng **không** eval. Cũng chặn `checkpoint_best` trước [5], nên không epoch chưa chấm nào được tuyển |
| `amp` / `amp_dtype` | true / `bfloat16` | BF16 giữ dải mũ, tránh GradScaler sụp |
| `test_splits` | `[test]` | Held out; eval **một lần** sau train |
| `save_predictions` | `true` | ★ Cho phép calibrate threshold offline |
| `uncertain_policy` | `ignore_uncertain` | |
| `seed` | 42 | `+ get_rank()` mỗi process |

## Khối `datasets:`
`image_size: 448`, `resize_size: 512`; augmentation train: affine ±5°,
translate 0.02, scale ±0.05, `affine_p`/`jitter_p` 0.5, brightness/contrast 0.1.
Khi mask cache bật, affine được sample một lần cho ảnh/mask; ảnh bilinear, mask
nearest. Không cấu hình cache thì sample không có ba key explanation mới.

## Consumer
`pretraining/train.py` · `precompute_features.py` · `run_medgemma_qlora.py:59`
(mặc định `--stage1-config`) · `scripts/vm_preflight.py:152`

## Developer notes
Comment về contrastive negative đáng nhớ: *"Contrastive negatives come from the
live microbatch plus the queue, not from gradient accumulation."* — tăng
`accum_grad_iters` **không** làm ITC có thêm negative.

← [`_index.md`](_index.md) · [HOME](../../../HOME.md)
