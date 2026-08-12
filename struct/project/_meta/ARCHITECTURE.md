> Source: `model/lavis/models/blip2_models/blip2_qformer.py`, `mhcac/`, `vision_encoders/`, `training/`
> Status: ✅ ACTIVE
> Last verified against source: 2026-08-12

# Kiến trúc Meta-CXR

Trang này mô tả các **khối** và cách chúng nối với nhau. Nếu bạn chưa đọc
[PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md), đọc nó trước.

Mọi shape ở đây được lấy từ code hoặc config. Chỗ nào không xác định được, trang
này ghi rõ là `dynamic` hoặc `⚠ cần runtime verification` — không đoán.

---

## 1. Bức tranh tổng thể

```text
                     configs/env_config.yaml          pretraining/configs/*.yaml
                              │ (đường dẫn máy)              │ (siêu tham số)
                              ▼                              ▼
                        local_config.py  ──────────►  Config (LAVIS)
                                                             │
  split CSV ──► MIMIC_CXR_Dataset ──► collater ──► batch ────┤
                (study-level)          (pad aux)             │
                                                             ▼
┌────────────────────────────── STAGE 1 : Blip2Qformer ──────────────────────────┐
│                                                                                │
│   image [B,3,448,448]        aux_image [B,N,3,448,448]                         │
│         │                            │                                         │
│         ▼                            ▼                                         │
│   ┌──────────────── encoder đóng băng (torch.no_grad cho nhánh aux) ────────┐  │
│   │  BioViL-T → 1408   PubMedCLIP → 768   SwinV2 → swin_dim   [RadDINO tắt] │  │
│   └────┬──────────────────┬────────────────────┬──────────────────────────── ┘ │
│        │                  │                    │                               │
│        ▼                  ▼                    ▼        ViewFusionModule        │
│   ViewFusion         ViewFusion           ViewFusion    (một cái / encoder,     │
│   [B,P,1408]         [B,P,768]            [B,P,dim]      chạy trên output THÔ)  │
│        │                  │                    │                               │
│        └──────────────────┴────────────────────┘                               │
│                           │                                                    │
│                           ▼                                                    │
│              SharedVisualTokenProjector  (điểm chiếu DUY NHẤT)                  │
│                    → SharedVisualTokens                                        │
│                      .tokens [B, N_total, 1408]                                │
│                      .spans  {biovil: slice, pubmedclip: slice, swin: slice}   │
│                           │                                                    │
│              ┌────────────┴─────────────┐                                      │
│              ▼                          ▼                                      │
│      ┌───────────────┐          ┌────────────────┐                             │
│      │     MHCAC     │          │    Q-Former    │                             │
│      │  (mhcac_12)   │          │   32 query     │                             │
│      │               │          │   cross-attn/2 │                             │
│      │ student: ảnh  │          └───────┬────────┘                             │
│      │ teacher: ảnh  │                  │                                      │
│      │  + text (train│          ┌───────┴────────┐                             │
│      │    only)      │          ▼       ▼        ▼                             │
│      └───────┬───────┘         ITC     ITM      LM                             │
│              ▼                                                                 │
│      logits [B,14,3]                                                           │
│                                                                                │
│   Tổng: 11 loss có trọng số → BlipOutput.loss                                  │
└────────────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼   checkpoint_best.pth
                              │
        ┌─────────────────────┴─────────────────────┐
        │  (medgemma_direct BỎ QUA nhánh này)        │
        ▼                                            ▼
┌───────────────── STAGE 2 : MedGemma 4B QLoRA ──────────────────────────────────┐
│                                                                                │
│  medgemma_direct        │  meta_cxr_qformer          │ +mhcac_prompt            │
│  ─────────────────      │  ────────────────          │ ─────────────            │
│  image tower riêng      │  Q-Former 32 token         │ soft token               │
│  của MedGemma           │  → img_proj → 768          │ + text P/N/U             │
│  + projector            │  → THAY THẾ tại vị trí     │   có cấu trúc            │
│                         │    <qformer_soft_token>    │   trong prompt           │
│         └───────────────┴────────────┬───────────────┘                          │
│                                      ▼                                          │
│                          PromptBuilder (stage2/prompts)                         │
│                                      ▼                                          │
│                          Gemma decoder + LoRA (CLI: r=16, α=32)                  │
│                                      ▼                                          │
│                          FINDINGS (± IMPRESSION)                                │
└────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Stage 1 — chi tiết từng khối

Toàn bộ Stage 1 sống trong một class: `Blip2Qformer`
([`model/lavis/models/blip2_models/blip2_qformer.py`](../model/lavis/models/blip2_models/blip2_qformer.py.doc.md), 1.497 dòng).

### 2.1 Vision encoders — đóng băng

| Encoder | Chiều ra | Bật ở config production? | Class |
|---|---|---|---|
| BioViL-T | 1408 | ✅ | `biovil_t.model.ImageModel` |
| PubMedCLIP | 768 | ✅ | `vision_encoders.pubmedclip.pubmed_clip.Pubmedclip` |
| SwinV2 | `swin.embed_dim` (⚠ đọc từ model đã load, cần runtime verification) | ✅ | `vision_encoders.swin.swin_encoder.SwinEncoder` |
| RadDINO | `raddino.embed_dim` | ❌ `raddino: false` | `vision_encoders.rad_dino.rad_dino_encoder.RadDinoEncoder` |

Cả bốn đều `.eval()` và `requires_grad = False`. Chúng **không học gì**. Thứ được
train là: các projection, view fusion, MHCAC, Q-Former và các head.

`blip2_qformer.py:173` bắt buộc **ít nhất một** encoder phải bật, nếu không raise.

### 2.2 View fusion — hợp nhất đa view

`mhcac/view_fusion.py` · `ViewFusionModule` / `ViewFusionBlock`

Một module **cho mỗi encoder**, chạy trên đầu ra **thô, trước projection**. Anchor
là Query; token của auxiliary view là Key/Value.

```text
anchor  [B, P, D]  ──► Q ─┐
                          ├── MHA ──► + residual ──► FFN ──► out [B, P, D]
aux     [B, N*P, D] ─► K,V┘
```

Ba thiết kế đáng chú ý, mỗi cái giải quyết một vấn đề cụ thể:

1. **`W_O` và Linear cuối của FFN được zero-init.** Tại step 0 khối này là
   **identity chính xác** trên anchor. Nhờ vậy checkpoint single-view load vào mà
   không hỏng gì — model bắt đầu đúng bằng model cũ rồi mới học phần fusion.
2. **Study không có auxiliary view bị gate về 0**, không bị tách khỏi batch.
   Batch giữ nguyên hình dạng; không có nhánh điều khiển phụ thuộc dữ liệu.
3. **Padding dùng `MASK_NEG = -1e4`, không phải `-inf`.** Một hàng toàn padding
   vẫn cho softmax hợp lệ thay vì NaN.

Shape contract: `[B, P, D]` vào → `[B, P, D]` ra. Nhờ vậy MHCAC và Q-Former
không cần biết multi-view có bật hay không.

Bật/tắt bằng `model.multi_view` (production: `true`).

### 2.3 SharedVisualTokenProjector — điểm chiếu duy nhất

`vision_encoders/shared_visual_tokens.py`

Mỗi encoder ra một chiều khác nhau. Projector này chiếu tất cả về `VISUAL_DIM = 1408`
rồi **nối theo trục token**:

```text
{biovil: [B,P₁,1408], pubmedclip: [B,P₂,768], swin: [B,P₃,swin_dim]}
                        │
                        ▼  SharedVisualTokenProjector
        SharedVisualTokens(
            tokens = [B, P₁+P₂+P₃, 1408],
            spans  = {biovil: slice(0,P₁), pubmedclip: slice(P₁,P₁+P₂), …}
        )
```

Tại sao có class riêng cho việc này: trước đây MHCAC và Q-Former mỗi bên tự chiếu
đặc trưng, nên **hai nhánh có thể trôi dạt sang hai biểu diễn thị giác khác nhau**.
Giờ chỉ còn một điểm chiếu, cả hai nhánh đọc chung một tensor. `spans` cho phép
MHCAC vẫn cấp positional encoding riêng cho từng encoder mà không cần chiếu lại.

Thứ tự stream cố định: `CANONICAL_STREAM_ORDER = ("biovil", "pubmedclip", "swin", "raddino")`
— để `spans` ổn định giữa các lần chạy.

### 2.4 MHCAC — phân loại bất thường

`mhcac/mhcac_12.py` · `AbnormalityClassificationModel`

Chỉ `mhcac_12` được wire. 11 variant còn lại là legacy
([D-003](DECISIONS.md#d-003--mhcac-variants-và-encoder-trùng-lặp-là-legacy)).

Tham số khởi tạo (từ `blip2_qformer.py:342`):

```python
AbnormalityClassificationModel(
    embed_dim=768, num_abnormalities=14, num_classes=3,
    num_layers=6, num_commmon_tokens=14,   # (sic — typo trong source)
    visual_dim=1408, text_dropout_rate=..., use_cnn=use_biovil,
    uncertain_policy=...,
)
```

Cơ chế: **expert token**. 14 token học được, mỗi token "chuyên trách" một bệnh lý.
Chúng cross-attend vào chuỗi token thị giác qua 6 lớp, rồi mỗi token đi qua một
classifier riêng ra 3 lớp.

```text
expert_tokens [B,14,768] ──┐
                            ├─ 6 × ExpertTokenCrossAttention ──► [B,14,768]
image_patches [B,ΣP,768] ──┘                                        │
                                                                     ▼
                                            14 × Linear ──► logits [B,14,3]
```

Chi tiết đáng chú ý ở lớp áp chót (`mhcac_12.py:~424`): expert token được **cộng
lại** phiên bản chuẩn hóa của expert token khởi tạo. Đây là residual chống việc
token trôi mất bản sắc ban đầu sau 6 lớp attention.

**Teacher/student — điểm quan trọng nhất của khối này:**

| Nhánh | Input | Khi nào | Có ở inference? |
|---|---|---|---|
| student | chỉ `shared_visual_tokens` | luôn luôn | ✅ **đây là thứ chạy thật** |
| teacher | + `text_embeddings` từ report | chỉ train | ❌ không bao giờ |

`blip2_qformer.py:907` gọi student với `text_embeddings=None`. `:925` gọi teacher
với text. Kiến thức teacher được chưng cất sang student qua `soft_target_kl_loss`
(teacher đã `detach`). Text **không có đường nào** rò vào inference.

Teacher chỉ chạy khi `teacher_mask = classification_mask & generation_mask` có ít
nhất một phần tử — tức là mẫu đó vừa có nhãn CheXpert vừa có FINDINGS hợp lệ.

Ngoài logits, MHCAC còn trả ba loss phụ: `contrastive_loss`, `orth_loss`
(giữ các expert token trực giao nhau), `sparsity_loss` (ép attention tập trung).

### 2.5 Q-Former — căn chỉnh ảnh ↔ text

`model/lavis/models/blip2_models/Qformer.py`, khởi tạo tại `blip2_qformer.py:192`

32 query token học được, cross-attention mỗi 2 block (`cross_attention_freq: 2`).
Ba mục tiêu:

| Loss | Ý nghĩa | Chi tiết |
|---|---|---|
| **ITC** | Image-Text Contrastive | Có **negative queue 1024 mẫu**, detach, fp16 (~16 MB) |
| **ITM** | Image-Text Matching | Hard negative mining từ batch hiện tại |
| **LM** | Language Modeling | Sinh lại FINDINGS từ query token |

**Tại sao có ITC queue:** recipe production dùng microbatch = 8. Contrastive loss
với 8 negative gần như vô nghĩa. Queue cung cấp 1024 negative đã detach từ các
batch trước. Quan trọng: `_image_text_contrastive` đặt `queue_filled = 0` khi
`not self.training` — **validation không được phụ thuộc vào việc batch train nào
tình cờ nằm trong ring buffer**.

Hard negative sampling (`_hard_negative_sampling_weights`, `:52`) tính phân phối
trong FP32 và **loại positive trước softmax**, không phải sau. Lý do ghi rõ trong
docstring: khi model tự tin, BF16 gán positive xác suất 1 và mọi negative 0; bỏ
positive sau đó để lại hàng toàn 0, làm `torch.multinomial` bắn device-side assert.

### 2.6 Tổng hợp loss

`blip2_qformer.py:974` — 11 số hạng có trọng số:

```python
total = λ_itc·L_itc + λ_itm·L_itm + λ_lm·L_lm
      + λ_cls·L_cls + λ_teacher_cls·L_teacher + λ_distill·L_distill
      + λ_mhcac_contrastive·L_contrastive + λ_orth·L_orth + λ_sparsity·L_sparsity
      + λ_mpc·L_mpc + λ_view_consistency·L_vc
```

Trọng số production (`mimic_cxr_full_l4.yaml`):

| λ | Giá trị | | λ | Giá trị |
|---|---|---|---|---|
| `itc` / `itm` / `lm` / `cls` | 1.0 | | `mhcac_contrastive` | 0.1 |
| `teacher_cls` / `distill` | 0.5 | | `orthogonality` | 0.05 |
| `mpc` | 0.1 | | `sparsity` | 0.01 |
| `view_consistency` | 0.05 | | `itc_queue_size` | 1024 |

Hai loss cuối chỉ chạy khi `multi_view` bật **và** batch có ít nhất một auxiliary
view thật (`aux_mask.any()`).

### 2.7 Mask — cách dữ liệu thiếu được xử lý

Đây là cơ chế xuyên suốt, không phải chi tiết vụn:

| Mask | Nguồn | Chặn loss nào |
|---|---|---|
| `classification_mask` | cột `classification_valid` trong CSV | `L_cls`, `L_teacher_cls` |
| `generation_mask` | cột `target_valid` trong CSV | `L_itc`, `L_itm`, `L_lm` |
| `teacher_mask` | `classification_mask & generation_mask` | nhánh teacher + distill |

Nhờ vậy một dòng không có nhãn CheXpert **vẫn train được phần sinh báo cáo**, chỉ
bị loại khỏi loss phân loại — thay vì bị vứt khỏi dataset.

---

## 3. Stage 2 — chi tiết

Entrypoint: [`training/run_medgemma_qlora.py`](../training/run_medgemma_qlora.py.doc.md)
Động cơ: [`training/train_eval_figure9_llm_variants_200.py`](../training/train_eval_figure9_llm_variants_200.py.doc.md)
(tên file gây hiểu nhầm — xem [D-006](DECISIONS.md#d-006--độ-sâu-documentation-cho-động-cơ-stage-2))

| | |
|---|---|
| Model | `google/medgemma-1.5-4b-it` |
| Lượng tử hóa | 4-bit NF4, double quant, compute dtype bfloat16 |
| LoRA | CLI mặc định `r=16`, `alpha=32`; full module name thuộc language tower, **không** `all-linear` |
| Song song | **Không có** — single process, single GPU. Không DDP, không FSDP. |
| Chọn checkpoint | Validation cross-entropy |

### 3.1 Soft token — chỗ dễ sai nhất repo

`training/medgemma/soft_tokens.py` · `SoftTokenEmbeddingWrapper`

Với mode Q-Former, output 768-d của Q-Former đi qua `img_proj` rồi **THAY THẾ**
embedding tại vị trí `<qformer_soft_token>` — **không cộng vào**.

Tại sao điều này đáng một cảnh báo riêng: nếu index theo hàng bị sai, **loss vẫn
giảm bình thường**, nhưng mỗi study đang được mô tả bằng ảnh của study khác. Lỗi
này im lặng hoàn toàn. Vì thế code validate shape theo từng hàng và fail-closed
thay vì đoán.

Ngoài ra soft token được đưa vào `bad_words_ids` khi generate, để model không tự
sinh ra ký hiệu đó.

### 3.2 Ràng buộc section

Mode Q-Former **chỉ hỗ trợ `findings_only`**, vì `ReportDataset.text_output`
không phát ra IMPRESSION. Nếu bạn yêu cầu `findings_and_impression` với mode
Q-Former, chương trình **báo lỗi và dừng** thay vì âm thầm đổi target.

Đường native hỗ trợ cả `findings_only`, `impression_only`, `findings_and_impression`
(mặc định).

### 3.3 Ranh giới độc lập

```text
training/stage1/lavis_loader.py   ← DUY NHẤT nơi được import LAVIS/Stage-1
        ▲
        │ chỉ được gọi từ bên trong nhánh đã quyết định là cần Stage 1
        │
training/run_medgemma_qlora.py :: build_stage1_records()
```

`training/dataio/manifest.py` đọc split CSV **chỉ bằng pandas** vì cùng lý do.

Enforce bằng `tests/test_native_independence.py`. Vi phạm ranh giới này = `medgemma_direct`
không khởi động nổi trên máy không có LAVIS, và ablation bị nhiễm.

---

## 4. Prompt v2

`stage2/prompts/` · `PromptBuilder` — điểm vào prompt **duy nhất** cho cả train và
inference, không đụng model/tokenizer/torch. Nhờ vậy parity giữa train và inference
kiểm tra được **byte-for-byte**.

Năm visual mode (`stage2/prompts/schemas.py`):

| Mode | Thấy Stage-1 labels? |
|---|---|
| `native_anchor_only` | ❌ |
| `native_anchor_guided` | ✅ |
| `native_multiview` | ❌ (⚠ manifest native hiện chỉ luồn anchor, `auxiliary_views` rỗng) |
| `qformer_visual_only` | ❌ — đây là thứ giữ cho ablation không bị nhiễm |
| `qformer_guided` | ✅ |

Chỉ guided mode nhìn thấy prediction có cấu trúc, và chúng được diễn đạt là
**gợi ý phụ trợ có thể sai**, không phải ground truth.

Prompt v2 là **opt-in** qua `--prompt-config configs/stage2_prompt_v2.yaml`; bỏ
flag thì dùng prompt legacy. Prompt prefix bị mask khỏi training label.

Mỗi prompt phát ra kèm version, config hash và template hash, được ghi vào
artifact metadata — để sau này biết chính xác một kết quả được sinh bằng prompt nào.

---

## 5. Hai khối stdlib-only

Cả hai không import torch, chạy được ở bất cứ đâu test chạy được.

### `safety/`
Điều phối: draft report → parse claims → verify → báo cáo cuối hoặc **từ chối trả lời**.
`safety/pipeline.py` không tự chứa logic verify, để có thể cắm một model
phrase-grounding thật vào qua cùng protocol.

`parse_coverage` được phơi ra **có chủ đích**: một pipeline chỉ parse được 2/12 câu
thì chưa kiểm tra được báo cáo, dù các con số của nó trông sạch đến đâu.

Record đầu ra **không mang** `subject_id`/`study_id`/`dicom_id`/path/reference text
→ an toàn để lưu.

⚠ `pipeline.py`, `verifiers.py`, `reconciler.py` hiện chưa có caller production —
xem [D-001](DECISIONS.md#d-001--hạ-tầng-đã-viết-nhưng-chưa-nối-vào-pipeline).
Riêng `safety/claims.py` **có** dùng thật ở `training/evaluation/error_analysis.py`.

### `runtime/`
`budget.py` tính tiền theo **wall-clock** — một GPU treo tốn đúng bằng một GPU
bận. Nó mang `prior_elapsed_seconds` để lần resume không reset trần chi phí. Nó
**chỉ dừng run**, không bao giờ tự hạ cấp model hay bật thêm section.

`device.py` resolve device/dtype từ config hoặc từ máy — không chỗ nào hardcode
`cuda:0`.

---

## 6. Cấu hình phân tầng

```text
configs/env_config.yaml         ─► local_config.py  ─► đường dẫn của MÁY
   (git-ignored, mỗi máy khác)                          VIS_ROOT, split CSV, W&B…

pretraining/configs/*.yaml      ─► LAVIS Config      ─► SIÊU THAM SỐ của RUN
                                                        model / run / datasets
```

**Bẫy:** `Config` của LAVIS chỉ merge ba block `run` / `model` / `datasets`. Vì
vậy block `data:` (study_sampling, anchor_priority, max_aux_views) **phải nằm bên
trong `model:`**. Đặt nó ở top level thì nó bị bỏ qua âm thầm.

Xem [`pretraining/configs/_index.md`](../pretraining/configs/_index.md) để biết
config nào là production, nào là legacy.

---

## 7. Những chỗ documentation cũ nói sai

Ghi lại để không ai đi vòng lại:

| Nguồn | Nói gì | Code thực tế |
|---|---|---|
| `CLAUDE.md`, `README.md` §Stage 1 | `selection_metric: f1_positive_macro` | `mimic_cxr_full_l4.yaml:109` → **`macro_auprc`** |
| Tên file `mimic_cxr_full_l4.yaml` | recipe cho NVIDIA L4 | Comment `:~135` ghi "Verified on RTX 5060 Ti 16 GB" |
| `docs/stage2_prompt_audit.md:20` | `utils/prompter.py` dùng bởi `inference.py` | Không có import nào — xem [D-002](DECISIONS.md#d-002--đường-vicuna-7b-legacy-vẫn-là-demo-active) |
| Tên file `train_eval_figure9_llm_variants_200.py` | script vẽ figure | Là động cơ Stage 2 |

---

## Liên kết

- [PIPELINES.md](PIPELINES.md) — cách chạy từng kiến trúc ở trên
- [DATA_FLOW.md](DATA_FLOW.md) — tensor biến đổi từng bước
- [CALL_GRAPH.md](CALL_GRAPH.md) — hàm gọi hàm
- [GLOSSARY.md](GLOSSARY.md) — MHCAC, Q-Former, anchor view, soft token, P/N/U…

← [Về HOME](../../HOME.md)
