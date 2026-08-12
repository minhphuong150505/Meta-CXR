> Source: thuật ngữ xuất hiện thực tế trong code/config của repository này
> Status: ✅ ACTIVE
> Last verified against source: 2026-08-12

# Glossary

Chỉ những thuật ngữ **thực sự xuất hiện** trong Meta-CXR. Không giải thích khái
niệm ML tổng quát trừ khi repo dùng nó theo nghĩa riêng.

---

## A

### Ablation
Thí nghiệm cố ý bỏ bớt một thành phần để đo đóng góp của nó. Trong repo:
`meta_cxr_qformer*` là ablation so với `medgemma_direct`;
`text_only_language_prior_ablation` bỏ hẳn ảnh.

### AMP (Automatic Mixed Precision)
Config production dùng `amp: true`, `amp_dtype: bfloat16`. BF16 giữ dải số mũ
gần FP32 nên tránh được hiện tượng `GradScaler` sụp đổ, đồng thời vẫn tiết kiệm
bộ nhớ/băng thông.

### Anchor view
View **chính** của một study, được chọn theo `anchor_priority: [PA, AP, lateral]`.
Nó là Query trong cross-attention của view fusion. Đối lập: [auxiliary view](#auxiliary-view).

### AUPRC
Area Under Precision-Recall Curve. `selection_metric` của config production
(`macro_auprc`). Chọn theo AUPRC thay vì F1 vì nó **không cần threshold** — ổn
định hơn khi lớp mất cân bằng nặng.

### Auxiliary view
View **bổ trợ**, tối đa `max_aux_views: 1`. Nó là Key/Value trong view fusion.
Study không có auxiliary view bị **gate về 0**, không bị loại khỏi batch.

---

## B

### BLIP-2
Kiến trúc vision-language của Salesforce, gồm vision encoder đóng băng + Q-Former
+ LLM. Meta-CXR fork phần Stage-1 của nó tại `model/lavis/`.

### BioViL-T
Vision encoder y khoa (Microsoft), chiều ra **1408**. Là `VISUAL_DIM` chuẩn mà
mọi encoder khác được chiếu về. Đóng băng.

⚠ Tồn tại **hai bản sao**: `biovil_t/` (được import) và `vision_encoders/biovil_t/`
(không ai dùng). Xem [LEGACY_AND_OPTIONAL.md §L4](LEGACY_AND_OPTIONAL.md#l4--vision_encodersbiovil_t--bản-sao).

---

## C

### CheXpert labels
Bộ 14 nhãn bất thường chuẩn cho X-quang ngực. Trong CSV mã hóa `0` = negative,
`1` = positive, `-1` = uncertain. Xem [P/N/U](#pnu).

### `classification_mask`
Mask boolean từ cột `classification_valid`. Chặn `L_cls` và `L_teacher_cls` cho
dòng không có nhãn CheXpert. Dòng đó **vẫn train phần sinh báo cáo**.

### Checkpoint selection
Stage 1: `macro_auprc` trên **validation**. Stage 2: **validation cross-entropy**.
Test split bị giữ ngoài hoàn toàn.

---

## D

### DDP (DistributedDataParallel)
**Chỉ Stage 1 có.** Stage 2 là single-process, single-GPU — không DDP, không FSDP,
và không dùng `device_map` rộng thay cho DDP.

Hai job với `CUDA_VISIBLE_DEVICES=0` và `=1` là **hai experiment độc lập**, không
chia sẻ gradient, không phải một distributed run.

### Distillation (chưng cất)
`soft_target_kl_loss` chuyển kiến thức từ [teacher](#teacherstudent) sang student.
Teacher luôn `detach()` — gradient không chảy ngược vào teacher.

---

## E

### Expert token
14 vector học được trong MHCAC, mỗi cái "chuyên trách" một bệnh lý. Chúng
cross-attend vào token thị giác qua 6 lớp rồi mỗi cái đi qua một classifier riêng.

Có ba loss phụ giữ chúng lành mạnh: `orth_loss` (giữ trực giao lẫn nhau),
`sparsity_loss` (ép attention tập trung), `contrastive_loss`.

---

## F

### Feature cache
Đầu ra encoder đóng băng được tính trước bởi `pretraining/precompute_features.py`.
Nó **chỉ** thay thế forward của encoder — các projection có thể train vẫn chạy,
nên training giống hệt.

⚠ Là dẫn xuất từ ảnh MIMIC → chịu cùng lệnh cấm redistribute.

### FINDINGS / IMPRESSION
Hai section của báo cáo X-quang. FINDINGS mô tả quan sát; IMPRESSION là kết luận
tóm tắt.

Một tỉ lệ đáng kể report **không có tag FINDINGS**; `mimic_report_parser.py` khôi
phục phần thân tường thuật thay vì rơi về nguyên văn cả báo cáo.

**Hai section không bao giờ được thay cho nhau** và có giới hạn độ dài riêng.
Mode Q-Former **chỉ** hỗ trợ `findings_only`.

### Fusion gate
Hệ số nhân `[B, 1, 1]` trong `ViewFusionBlock`, bằng `0` cho study không có
auxiliary view. Nhờ nó batch giữ nguyên hình dạng, không cần nhánh điều khiển
phụ thuộc dữ liệu.

---

## G

### `generation_mask`
Mask boolean từ cột `target_valid`. Chặn `L_itc`, `L_itm`, `L_lm` cho dòng không
có FINDINGS dùng được.

---

## H

### Hard negative
Negative sample **khó** (giống positive nhất) trong ITM, lấy bằng
`torch.multinomial` từ phân phối similarity.

⚠ `_hard_negative_sampling_weights` tính trong **FP32** và loại positive **trước**
softmax. Làm ngược lại thì ở BF16, khi model tự tin, hàng trở thành toàn 0 và
`multinomial` bắn device-side assert.

---

## I

### ITC / ITM / LM
Ba mục tiêu huấn luyện Q-Former:

| | Tên | Việc |
|---|---|---|
| **ITC** | Image-Text Contrastive | Kéo cặp (ảnh, report) đúng lại gần nhau |
| **ITM** | Image-Text Matching | Phân loại nhị phân "cặp này có khớp không", dùng hard negative |
| **LM** | Language Modeling | Sinh lại FINDINGS từ query token |

### ITC queue
Ring buffer 1024 mẫu, detach, fp16 (~16 MB). Cần thiết vì microbatch chỉ 8 —
contrastive với 8 negative gần như vô nghĩa.

⚠ `queue_filled = 0` khi `not self.training` — **validation không được phụ thuộc
vào việc batch train nào tình cờ nằm trong buffer**.

---

## L

### LAVIS
Thư viện vision-language của Salesforce. `model/lavis/` là **fork đã sửa**, không
phải package cài qua pip. Bị loại khỏi ruff: reformat sẽ làm mọi diff upstream
sau này không đọc được.

### LoRA / QLoRA
LoRA: fine-tune bằng ma trận hạng thấp thay vì toàn bộ trọng số.
QLoRA: LoRA trên model đã lượng tử hóa 4-bit.

Stage 2: NF4 double quant; dtype do `preferred_dtype()` chọn (bf16 nếu CUDA hỗ
trợ, nếu không fp16; CPU fp32). CLI chính mặc định `r=16`, `alpha=32`.
`_language_lora_targets()` chọn full module name chỉ thuộc language tower và
raise nếu không tìm được; code cố ý **không** fallback sang `all-linear` vì sẽ
adapt cả vision tower.

---

## M

### MHCAC
**M**ulti-**H**ead **C**ross-**A**ttention **C**lassification. Khối phân loại
14 bệnh lý × 3 lớp của Meta-CXR.

Chỉ `mhcac/mhcac_12.py` được wire; 11 variant khác là legacy.

⚠ Tham số khởi tạo có typo: `num_commmon_tokens` (ba chữ m).

### MPC (Multi-Positive Contrastive)
`MultiPositiveContrastiveLoss` — kéo các view khác nhau **của cùng một study** lại
gần nhau. `lambda_mpc: 0.1`. Chỉ chạy khi multi_view bật và batch có aux thật.

### Multi-view
Chế độ dùng nhiều view cho một study. `multi_view: true` ở config production.
Xem [anchor view](#anchor-view), [auxiliary view](#auxiliary-view),
[view fusion](#view-fusion).

---

## P

### P/N/U
**P**ositive / **N**egative / **U**ncertain — ba lớp cho mỗi bệnh lý.

Uncertain **không** phải lỗi thiết kế: báo cáo thật đầy câu kiểu "không loại trừ
khả năng viêm phổi". Ép về nhị phân là mất thông tin lâm sàng.

Xử lý ở đánh giá do `uncertain_policy` quyết định (production: `ignore_uncertain`).

### Pipeline mode
Tên kiến trúc Stage 2. Thay cho cờ cũ `--image-mode {native,qformer,both}` — cờ
đó mô tả chi tiết cài đặt chứ không mô tả kiến trúc, khiến hybrid dễ bị gọi nhầm
là "native MedGemma".

### PromptBuilder
Điểm vào prompt **duy nhất** cho cả train và inference (`stage2/prompts/builder.py`).
Không đụng model/tokenizer/torch → parity kiểm tra được byte-for-byte.

### Prompt v2
Hệ prompt có version, config hash, template hash. **Opt-in** qua
`--prompt-config configs/stage2_prompt_v2.yaml`; bỏ flag thì dùng prompt legacy.

---

## Q

### Q-Former
Querying Transformer của BLIP-2. **32 query token** học được, cross-attention mỗi
2 block, nén chuỗi token thị giác dài thành 32 vector cho LLM.

---

## R

### RadDINO
Vision encoder DINOv2 y khoa (`microsoft/rad-dino`). Có wire đầy đủ nhưng
`raddino: false` ở mọi config → 🟡 CONDITIONAL, **không phải** dead code.

### RadGraph / RadCliQ / CheXbert / RadFact
Chỉ số đánh giá lâm sàng. **Cố ý không cài được** trong repo — research code sau
license riêng, không phải pin tái lập được.

`training/evaluation/clinical.py` báo `unavailable` hoặc `NotImplementedError`.
**Chỉ số thiếu không bao giờ được thay bằng điểm 0.**

---

## S

### `SharedVisualTokens`
Dataclass mang `tokens [B, N, 1408]` + `spans {tên_encoder: slice}`. Là biểu diễn
thị giác **duy nhất** — MHCAC và Q-Former đọc chung nó, nên hai nhánh không thể
trôi dạt sang hai biểu diễn khác nhau.

### Soft token
Vector Q-Former đã chiếu, **THAY THẾ** embedding tại vị trí `<qformer_soft_token>`
trong Stage 2 — **không cộng vào**.

⚠ Chỗ dễ sai nhất repo: index sai theo hàng thì loss vẫn giảm, nhưng mỗi study
được mô tả bằng ảnh của study khác. Hoàn toàn im lặng. Vì thế có validate shape
theo hàng và fail-closed.

### Stage 1 / Stage 2
Stage 1 = biểu diễn + phân loại (`pretraining/train.py`).
Stage 2 = sinh báo cáo (`training/run_medgemma_qlora.py`).

Chúng **cố ý tách rời**; mode mặc định của Stage 2 không dùng Stage 1 gì cả.

### Study
Một lần chụp của một bệnh nhân, có thể gồm nhiều ảnh. **Đơn vị lấy mẫu của
Meta-CXR** (`study_sampling: true`) — một dòng cho một study, không phải một ảnh.

Lý do: coi mỗi ảnh là một mẫu sẽ lặp lại cùng một báo cáo nhiều lần và đánh trọng
số quá cao cho study nhiều view.

---

## T

### Teacher/student
Hai nhánh của MHCAC:

| | Input | Khi nào | Có ở inference? |
|---|---|---|---|
| **student** | chỉ ảnh | luôn luôn | ✅ đây là thứ chạy thật |
| **teacher** | ảnh + report text | **chỉ train** | ❌ không bao giờ |

Text **không có đường nào** rò vào inference. Bảo vệ bởi
`tests/test_stage1_objectives.py`.

### Temporal hallucination
Model viết "không đổi so với phim trước" trong khi input **không có** phim trước.
Prompt v2 có guard cấm so sánh thời gian khi không có prior.

⚠ `temporal_target_policy` mặc định vẫn là `keep` — guard trong prompt **không**
đồng nghĩa dữ liệu train đã có prior linkage.

---

## U

### `uncertain_policy`
Cách xử lý lớp Uncertain. Production: `ignore_uncertain` — cặp mơ hồ không phải
target positive/negative đáng tin. Xem [P/N/U](#pnu).

---

## V

### View fusion
Cross-attention hợp nhất anchor với auxiliary view, chạy trên đầu ra **thô,
trước projection**, một module cho mỗi encoder.

`W_O` và Linear cuối của FFN **zero-init** → tại step 0 là identity chính xác →
checkpoint single-view load vào không hỏng gì.

### View consistency
`view_consistency_loss` — ép logits của bản đã fuse gần với logits chỉ-anchor,
để fusion không làm lệch dự đoán một cách tùy tiện. `lambda_view_consistency: 0.05`.

### `ViewPosition`
Cột trong CSV: `PA`, `AP`, `LATERAL`, `LL`, `UNKNOWN`. `_view_id` map nó thành số
nguyên cho positional embedding của view.

### `VISUAL_DIM`
Hằng `1408` (`blip2_qformer.py:33`). Chiều chung mà mọi encoder được chiếu về
trước khi nối.

---

## W

### `warmup_steps`
Đếm bằng **optimizer update**, không phải microbatch. Production: `300` (~3% của
một run 3 epoch).

⚠ Config legacy `mimic_cxr_2gpu.yaml` để `32000` — ramp **không bao giờ hoàn tất**.
Đừng copy giá trị đó.

---

## Viết tắt nhanh

| | |
|---|---|
| AMP | Automatic Mixed Precision |
| AUPRC / AUROC | Area Under PR / ROC Curve |
| CoW | Copy-on-Write (pandas 3.0) |
| DDP | DistributedDataParallel |
| DUA | Data Use Agreement (PhysioNet) |
| ITC / ITM / LM | Image-Text Contrastive / Matching / Language Modeling |
| MHCAC | Multi-Head Cross-Attention Classification |
| MPC | Multi-Positive Contrastive |
| NF4 | NormalFloat 4-bit |
| NLG | Natural Language Generation |
| P/N/U | Positive / Negative / Uncertain |
| VLM | Vision-Language Model |

---

← [Về HOME](../../HOME.md) · [ARCHITECTURE.md](ARCHITECTURE.md) · [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)
