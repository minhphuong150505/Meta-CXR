# META-Former + MHCAC, three phases as in the META-CXR paper

## Plan (the user's prompt, verbatim)

# NHIỆM VỤ: Huấn luyện META-Former + MHCAC theo đúng 3 pha của bài báo META-CXR

Repo: ~/Meta-CXR. Đọc CLAUDE.md, docs/handoff/README.md và file bàn giao blank-as-negative (PLAN-2026-09-24-blank-as-negative.md). Nhiệm vụ này chạy TRÊN cấu hình đó: blank → negative, 14 nhãn, mention gate tắt, `ignore_uncertain`.
Tạo file bàn giao: docs/handoff/PLAN-2026-09-24-meta-former-3phase.md (chép prompt này vào phần Plan). Khi xong, nối `## Execution report` vào cùng file.

## Nguồn sự thật
Nguồn chính là BÀI BÁO (Edirisinghe et al., IEEE Access 2025, doi 10.1109/ACCESS.2025.3606961, mục Implementation → Representation Learning). KHÔNG lấy code DasithEdirisinghe/META-CXR làm chuẩn: code đó đã comment toàn bộ ITC/ITM/ITG ở mọi commit và chỉ phản ánh bước train MHCAC.

Nội dung bài báo, tóm tắt:
- Encoder đóng băng; đặc trưng patch của từng encoder được chiếu tuyến tính về cùng không gian, ghép theo chiều sequence, rồi đưa vào CẢ META-Former lẫn MHCAC.
- Mask ngẫu nhiên 10% đặc trưng của mỗi encoder, làm regularization.
- **Pha 1a**: đóng băng MHCAC; chỉ tối ưu ITC + ITG + ITM cho query token của META-Former.
- **Pha 1b**: mở băng dần MHCAC, đồng thời đóng băng dần META-Former. Warm-up LR 5e-5 → 2e-4, L2, dropout 0.2, cosine 2e-4 → 1e-5 trong 5 epoch.
- **Pha 1c**: khi phân loại đã hội tụ, mở băng META-Former và projection head của các encoder, train chung phân loại + căn chỉnh bằng loss có trọng số.
- Tổng pha representation: 20,000 bước.

## Quyết định người dùng ĐÃ CHỐT (không được đổi)
1. Khởi tạo Q-Former: GIỮ `blip2_pretrained.pth` của LAVIS như `mimic_cxr_full.yaml` hiện tại. KHÔNG dùng checkpoint của RaDialog.
2. Encoder: BẬT LẠI Swin, dùng checkpoint SwinV2 đã có trong repo (`ChayanM/SwinV2-GPT2_Mimic`) → 3 stream: BioViL-T + PubMedCLIP + SwinV2.
   - Việc đầu tiên: đọc model card trên Hugging Face và báo lại model này được train trên dữ liệu nào.
   - Nếu KHÔNG phải MIMIC-CXR(-JPG): DỪNG và báo cáo, không tự chọn Swin khác.
   - Báo số token Swin đưa vào chuỗi ghép. Bài báo dùng 50 token cho Swin (49 patch + 1 vector pooled).
3. Giữ các thành phần repo tự thêm: teacher/student distill, multi-view, MPC + StreamAdapter, view-consistency. Chỉ THÊM lịch 3 pha. Chúng hoạt động ở pha 1b và 1c như cấu hình hiện tại, và TẮT ở pha 1a (bài báo: pha 1a "optimizes only ITC, ITG, ITM").

## Việc cần làm

### A. Lịch pha trong config + runner
Thêm khối `run.phases` vào `mimic_cxr_full.yaml`, mỗi pha khai báo:
- tên pha;
- độ dài (theo optimizer update, giống `warmup_steps`);
- lambda của từng loss;
- tập tham số được train (theo tiền tố tên module);
- LR/lịch LR riêng.

Runner chuyển pha theo bước. Mỗi lần chuyển pha:
- dựng lại param groups của optimizer, chỉ gồm tham số trainable;
- log rõ số tham số trainable theo từng module;
- lưu checkpoint cuối mỗi pha: `checkpoint_phase1a`, `checkpoint_phase1b`, `checkpoint_phase1c`.

Không có khối `phases` thì chạy như cũ (một pha), để các run cũ vẫn tái lập được.

### B. Pha 1a: căn chỉnh META-Former
- **Trainable**: `query_tokens` và toàn bộ Q-Former (image + text tower), `vision_proj`, `text_proj`, `itm_head`, `temp`. Nếu projection chung (shared visual tokens) nằm TRƯỚC Q-Former và đang được train sẵn, thì báo lại vị trí và cách xử lý hiện tại, không tự quyết.
- **Đóng băng**: MHCAC, mọi encoder (kể cả layer4/projector BioViL và block 10–11 CLIP đang mở băng trong cấu hình hiện tại), StreamAdapter.
- **Loss**: `lambda_itc = lambda_itm = lambda_lm = 1.0`. Mọi lambda khác = 0. ITC có label smoothing 0.1, lấy max theo query, ITM dùng hard negative sampling (theo BLIP-2/RaDialog). Kiểm tra code ITC/ITM/LM hiện có trong repo có đúng như vậy không; báo mọi điểm khác.
- **Đầu vào Q-Former**: đúng chuỗi token ghép mà MHCAC nhận, gồm cả Swin, sau mask 10%. Xác nhận bằng test rằng hai module nhận cùng một tensor.
- **Batch — điểm then chốt**: repo đã ghi ITC ở mức ngẫu nhiên với batch 8–16. Vì encoder đóng băng hoàn toàn trong pha 1a:
  1. Dùng `run.feature_cache_dir` để cache đặc trưng của 3 encoder (anchor + view phụ nếu multi-view dùng tới).
  2. Train Q-Former trên cache với batch mục tiêu ≥ 32. Mask 10% áp SAU khi đọc cache.
  3. TRƯỚC khi ghi cache: ước tính và báo dung lượng (số study × token × dim × dtype), cùng dung lượng đĩa trống trên /home. Không đủ chỗ → DỪNG và báo cáo.
  4. Đo VRAM thực tế ở batch 32. Không vừa → thử 24/16 và báo; KHÔNG tự bật GradCache hay SigLIP.
- **Độ dài**: tối đa 4 epoch (RaDialog dùng checkpoint epoch 4 cho Q-Former). Cuối mỗi epoch chạy `scripts/check_itc_gate.py` trên val.
- **CỔNG DỪNG**: nếu sau 2 epoch retrieval R@1/R@5 vẫn không vượt mức ngẫu nhiên theo đúng tiêu chí của gate check hiện có → DỪNG TOÀN BỘ, không sang pha 1b. Báo cáo loss ITC/ITM/LM theo bước, giá trị `temp`, batch thực tế.

### C. Pha 1b: chuyển sang MHCAC
- Loss = toàn bộ loss phân loại hiện có (CE với class weight mới + 0.3·contrastive + 0.7·orth + 0.3·sparsity) + distill, MPC, view-consistency, với lambda như `mimic_cxr_full.yaml` hiện tại. ITC/ITM/LM = 0.
- **"Mở dần / đóng dần"** (bài báo không nêu công thức) — cài đặt như sau:
  - trong `transition_steps` đầu pha, hệ số LR của MHCAC tăng tuyến tính 0 → 1, còn của META-Former giảm 1 → 0;
  - hết `transition_steps` thì đặt `requires_grad=False` cho META-Former;
  - mặc định `transition_steps` = 10% độ dài pha.
- **LR**: warm-up 5e-5 → 2e-4, sau đó cosine về 1e-5 trong 5 epoch; dropout MHCAC = 0.2; weight_decay giữ như config. Báo nếu dropout hiện tại khác 0.2.
- **Encoder**: vẫn đóng băng (bài báo: projection head chỉ mở ở pha 1c).
- **Độ dài**: 5 epoch.

### D. Pha 1c: train chung
- **Trainable**: META-Former + MHCAC + projection head của từng encoder (projector BioViL, lớp chiếu PubMedCLIP, lớp chiếu Swin) + StreamAdapter.
  - Việc có mở lại layer4 BioViL / block 10–11 CLIP như cấu hình cũ hay không: đặt thành flag `phase1c.unfreeze_encoder_blocks`, mặc định `false` (đúng bài báo). Báo lại để người dùng quyết.
- **Loss**: loss pha 1b + `w_align · (ITC + ITM + LM)`, mặc định `w_align = 1.0`.
- **Độ dài**: mặc định 1 epoch.
- **Không vừa VRAM** (Swin + teacher + ITM): báo con số đo được, không tự giảm thành phần.

### E. Test (CPU, `CUDA_VISIBLE_DEVICES=""`)
- Mỗi pha: đúng tập tham số `requires_grad`; tham số đóng băng có grad = None sau một bước backward.
- Pha 1a: MHCAC không đổi trọng số sau một bước optimizer; loss chỉ gồm ITC/ITM/LM.
- Chuyển pha: param groups được dựng lại, không còn tham số đã đóng băng trong optimizer.
- Hệ số LR trong `transition_steps` đúng tuyến tính.
- Q-Former và MHCAC nhận cùng tensor token ghép (có Swin).
- Feature cache: đặc trưng đọc lại khớp đặc trưng tính trực tiếp (sai số theo dtype).
- Không có `run.phases` thì hành vi y hệt trước.

### F. Kiểm chứng GPU (chỉ SMOKE)
- Trước mỗi lần chạy: `pgrep`, `nvidia-smi` rảnh, output_dir mới trên /home, bấm launch MỘT lần.
- Smoke mỗi pha với `truncate_train` nhỏ, vài trăm bước: loss từng term, `s/it`, `max mem`, không NaN/inf, số tham số trainable.
- KHÔNG chạy full 3 pha. Chờ người dùng duyệt execution report.

### G. Tài liệu (cùng commit)
- README, CLAUDE.md, docs/so_sanh_voi_repo_goc.md: thêm mục "META-Former 3 pha theo bài báo". Ghi rõ nguồn là bài báo, không phải code gốc.
- struct/project/_meta/DECISIONS.md: ghi quyết định bật lại ITC/ITM/LM (ở pha 1a và 1c) cùng Swin. Lý do mới: tách pha + encoder đóng băng → feature cache → batch lớn. Ghi rõ đây là quyết định của người dùng, và tham chiếu lịch sử bật/tắt cũ trong CLAUDE.md.

### H. Đo tác động của ITC/ITM/LM lên phân loại (chỉ đo, không đổi thiết kế)
- Cuối pha 1b và mỗi epoch pha 1c: đánh giá MHCAC trên VAL (framing `study_presence`). Báo macro AUROC, macro AUPRC, positive macro F1 — cả 13 nhãn lẫn 14 nhãn — và delta so với `checkpoint_phase1b`.
- Trong pha 1c, cứ mỗi 200 optimizer update:
  - tính riêng gradient của (a) nhóm loss phân loại và (b) `w_align·(ITC+ITM+LM)` lên các tham số dùng chung: `shared_visual_projector`, StreamAdapter, projection head của encoder, text tower của Q-Former;
  - log cosine similarity và tỉ lệ chuẩn ‖g_b‖/‖g_a‖ theo từng nhóm tham số.
  - Dùng `torch.autograd.grad` với `retain_graph`; không làm thay đổi bước optimizer thật.
- Trong pha 1c: log giá trị gate ITC trên val mỗi epoch như pha 1a.
- Báo cáo bảng tổng hợp.
- KHÔNG tự thêm stop-gradient, CAME-Grad hay giảm `w_align`, kể cả khi thấy phân loại giảm. Chỉ báo số liệu để người dùng quyết.

## Execution report cần có
- Nguồn dữ liệu train của SwinV2 và số token Swin.
- Bảng tham số trainable theo pha.
- Ước tính dung lượng cache và dung lượng đĩa trống.
- VRAM và s/it từng pha.
- Kết quả pytest (baseline so với mới).
- Kết quả smoke.
- Mọi chỗ code hiện tại mâu thuẫn với prompt này: DỪNG và ghi lại, không tự ứng biến.

Tóm tắt log, không dán nguyên file. Không đưa dữ liệu bệnh nhân vào commit, handoff hay tóm tắt.

---

## Execution report — 2026-09-24, planning checkout (no code changed)

### ⛔ STOPPED at decision 2: the SwinV2 checkpoint cannot be shown to be trained on MIMIC-CXR

Model card of `ChayanM/SwinV2-GPT2_Mimic` (Hugging Face README + `config.json`,
read 2026-09-24):

- **"This model is a fine-tuned version of [](https://huggingface.co/) on an
  unknown dataset."** The "Training and evaluation data" section says "More
  information needed". The only reference to MIMIC is the repository **name**.
  No dataset tag, no paper, no description.
- The card is the auto-generated `Trainer` stub. Its training log is **2 epochs
  x 125 steps at batch 4, i.e. ~500 training examples**, with ROUGE identical to
  four decimals in both epochs and `Gen Len 9.0` — a captioning fine-tune that
  barely moved, not a representation trained on MIMIC-CXR at scale.
- Architecture: `VisionEncoderDecoderModel`, encoder
  `microsoft/swinv2-base-patch4-window12to24-192to384-22kto1k-ft` (**ImageNet-22k
  → 1k**), decoder `gpt2`; ImageNet normalisation, 384 x 384 input. So the
  visual features are ImageNet features plus, at most, ~500 captioning steps on
  data of unknown origin.

By the rule in the prompt ("Nếu KHÔNG phải MIMIC-CXR(-JPG): DỪNG"), and since the
card does not establish MIMIC-CXR, **nothing else was started**: no config, no
runner, no cache, no GPU. The choice of a different Swin is the user's.

### Swin token count (asked for regardless)

- The encoder emits `last_hidden_state` of the final stage: 384 / 32 = **12 x 12
  = 144 tokens x 1024**, no pooled vector (`vision_encoders/swin/swin_encoder.py:116-140`).
- With `encoders.swin: true`, `Blip2Qformer._native_stream_layouts` returns
  `None` (`blip2_qformer.py:695`), so MHCAC falls back to its legacy path:
  every stream is resized to `target_patch_count = 49` (`mhcac/mhcac_12.py:246`),
  BioViL goes through `cnn_downsampler` 14x14 → 7x7, and PubMedCLIP's CLS is
  dropped. **MHCAC would see 3 x 49 = 147 tokens, Swin contributing 49 and no
  pooled token.** The paper's 50 (49 + 1 pooled) is not what the code does.

### Other conflicts noticed on the way (not investigated further, recorded so they are not lost)

1. **Turning Swin on silently reverts the other two streams** to the pooled
   7x7 / CLS-less layout that CLAUDE.md records as measured-worse (the 2026-08-14
   native-layout change). The prompt keeps "all repo additions", but this one is
   switched off by the same flag.
2. **Q-Former and MHCAC do not see the same tensor today.** Both read the output
   of `shared_visual_projector` (`blip2_qformer.py:814`), which sits BEFORE the
   Q-Former and is trainable, but MHCAC then resizes each stream internally
   (legacy path), so its token sequence differs from the Q-Former's. Section B
   asks to report exactly this and not decide.
3. **No 10% per-encoder feature mask** exists in the model path as far as a
   grep shows; it would be new code.
4. `run.feature_cache_dir` exists, but whether it caches the auxiliary view and
   the Swin stream, and at what dtype, has not been checked yet.

### State left behind

- No source change. This file is the only new file.
- Commit `08592d3` (gate off) is still **not** on `origin`: after a fetch,
  `feat/stage2-finding-tokens` is ahead of `origin/feat/stage2-finding-tokens`
  by one commit.

---

## Decision update (user, 2026-09-24) — replaces decision 2

Swin = **MedCLIP Swin-Tiny** (Wang et al., EMNLP 2022), backbone
`microsoft/swin-tiny-patch4-window7-224`, weights via
`MedCLIPModel(vision_cls=MedCLIPVisionModelViT).from_pretrained()`; tokens =
`last_hidden_state` [B,49,768] + `pooler_output` [B,768] → [B,50,768], pooled at
position 0, no 512-d `projection_head`; MedCLIP preprocessing from the raw
image (224, 1→3 channels, mean 0.5862785803043838, std 0.27950088968644304);
frozen in every phase; cached with phase 1a; `ChayanM/SwinV2-GPT2_Mimic` kept
only as an ablation path. Items a–g as given by the user.

## Execution report — 2026-09-24 20:15–20:45, planning checkout + host (no source change)

### (a) MedCLIP weights — PASS

Installed `medclip==0.0.3` and `wget` with `pip --no-deps --target ~/medclip_pkg`
on the host (used via `PYTHONPATH` only; the production venv and site-packages
are untouched). Script `~/medclip_gate.py`, log `~/medclip_gate.log`.

- The zip (506,392,441 B, from the URL in `medclip_vit_weight.txt`) downloads;
  `pytorch_model.bin` 546,888,925 B in `~/medclip_work/pretrained/medclip-vit/`.
- `MedCLIPModel.from_pretrained()` **fails its strict load** under transformers
  4.53.2 on exactly one key: `text_model.model.embeddings.position_ids` (a
  buffer newer transformers no longer persists). Text side only. Loading the
  same file with `strict=False`: **missing 0, unexpected 1** (that key).
- `vision_model.model.*`: 231 keys in the checkpoint, 231 in HF `SwinModel`,
  **0 missing, 0 unexpected**.
- Sanity: **219 of 219** float tensors differ from ImageNet Swin-Tiny; e.g.
  `patch_embeddings.projection.weight` max |Δ| 0.0073,
  `layers.3.blocks.1.attention.self.query.weight` 0.0522, final `layernorm.weight`
  0.0440 — MedCLIP weights are really loaded.
- Shapes: `last_hidden_state` (B, 49, 768), `pooler_output` (B, 768);
  `MedCLIPVisionModelViT.forward` returns (B, 512) — confirming (b).
- Note: the checkpoint's tensors were saved on CUDA, so `from_pretrained()` on a
  CPU-only process dies in `torch.load` before any key check.

### (b) Old wrapper — confirmed by reading

`vision_encoders/medclip/medclip.py` does
`pool_embeds, patch_embeddings = self.model.vision_model(pixel_values=...)`,
but that call returns one (B, 512) tensor. Batch 2 unpacks into two 512-vectors
and fails later at the concat; batch 3 raises "too many values to unpack". Test
not written yet (nothing written, see "stopped" below).

### (f) Leakage — what the MedCLIP paper says

arXiv 2210.10163, §4.2/4.3: *"MIMIC-CXR … We use the training split of this
dataset for pre-training"* and *"MIMIC-CXR and CheXpert are used for
pre-training where we held 5000 samples out for evaluation."* But Table 3 lists
MIMIC-CXR **377,111 images / 201,063 reports** — the size of the WHOLE dataset,
not of the official train split. The repo's dataset code reads
`./local_data/mimic-cxr-train-meta.csv`, which is not published. So the text
claims the train split and the table contradicts it; whether the official test
split was excluded **cannot be verified**. To go into README/DECISIONS as a
limitation (same encoder as the original META-CXR, so the comparison
condition is shared).

Also from the paper: *"All images are padded to square then scaled to
224 × 224"*, matching `MedCLIPFeatureExtractor` (`do_pad_square=True`, resize,
center crop 224). But MedCLIP's own training dataset transform is
`ToTensor → Resize((224,224)) → Normalize` — **no padding, aspect not kept**.
Which one to reproduce is a choice (see D6).

### ⛔ STOPPED at B.3 — the phase-1a cache does not fit on /home

Per-image fp16 size of the three raw streams: BioViL 196 × 1408 = 551,936 B,
PubMedCLIP 50 × 768 = 76,800 B, MedCLIP-Swin 50 × 768 = 76,800 B →
**705,536 B**. Counts from `full_allviews_v2`: train 368,960 images / 222,758
studies, val 2,991 / 1,808, test 5,159 / 3,269. Free on `/home`:
**152.6 GB (142 GiB)**.

| cache layout | train + val | fits? |
|---|---:|---|
| every image row (what `precompute_features.py` writes today, needed for aux views) | **262.4 GB** | no |
| anchor only, one per study | **158.4 GB** | no (−5.8 GB) |
| Swin only, every image | 28.6 GB | yes |
| PubMedCLIP + Swin only, every image | 57.1 GB | yes |
| PubMedCLIP + Swin only, anchor only | 34.5 GB | yes |

The Swin stream alone adds 76.8 KB per image (28.6 GB image-rows, 17.2 GB
anchor-only). Nothing was written.

### Mismatches between the code and the prompt (recorded, not resolved)

- **D1 cache** — above.
- **D2 projection before the Q-Former is trainable today.**
  `shared_visual_projector` (`blip2_qformer.py:471`) sits before the Q-Former:
  `nn.Identity` for BioViL (whose 1408-d projection lives inside the encoder, as
  `visual_encoder.projector`, after which `ln_vision` is applied), and a single
  `nn.Linear(768, 1408)` each for PubMedCLIP and Swin. It is part of every
  optimizer group today. If phase 1a freezes it, the Q-Former reads PubMedCLIP
  and Swin through **randomly initialised** linear maps for the whole phase.
  `ln_vision` is also neither an encoder nor in the phase-1a trainable list.
- **D3 projection form.** Paper: linear to 1408. Original code
  (`vision_encoders/medclip/medclip.py`): MLP 768→1024→1408 with ReLU. This
  repo: one `nn.Linear(768, 1408)`. Not changed.
- **D4 ITC/ITM/LM vs BLIP-2/RaDialog.** Max over query tokens ✔
  (`.amax(dim=-1)`, `blip2_qformer.py:958-963`); ITM hard negatives from the
  ITC similarities ✔ (`_hard_negative_sampling_weights`), drawn from the
  current batch; LM ✔. **No label smoothing** in ITC (`F.cross_entropy`
  without it; prompt asks 0.1). **A 256-entry negative queue** filled from the
  live encoder, which BLIP-2 does not have (`itc_queue_size`).
- **D5 MHCAC dropout is 0.1** (`mhcac_12.py:144`), prompt asks 0.2 for phase 1b;
  `text_dropout` is already 0.2.
- **D6 MedCLIP preprocessing**: pad-to-square (processor, paper) vs plain
  resize (MedCLIP training code).
- **D7 no 10% mask exists.** `Blip2Qformer._create_mask` is defined and never
  called; as written it also masks the same positions for the whole batch and
  over the concatenated sequence, not per encoder.
- **D8 Q-Former and MHCAC share one tensor only under native stream layouts.**
  Both read `shared_visual.tokens`; with Swin on,
  `_native_stream_layouts` returns `None` and MHCAC resamples every stream to
  49 internally (BioViL 7x7, PubMedCLIP CLS dropped). A `StreamLayout(50,
  num_global_tokens=1)` for the MedCLIP stream would keep all three native:
  196 + 50 + 50 = **296 tokens** into both branches.
- **D9 schedule length.** Paper: 20,000 steps for the whole representation
  phase. Prompt: 1a ≤ 4 epochs at batch ≥ 32 (~6,900 updates/epoch → up to
  ~27,600), 1b 5 epochs, 1c 1 epoch. At the current effective batch 64,
  one epoch is ~3,480 updates, so 1b+1c alone ≈ 20,900. The two lengths cannot
  both hold.
- **D10 the phase-1a ITC gate needs more than the current script.**
  `scripts/check_itc_gate.py` reports ranks and `delta_nats`; the prompt's gate
  is "R@1/R@5 above chance". R@k is not computed today.

### State left behind

No source change. On the host: `~/medclip_pkg` (medclip 0.0.3 + wget, not on
any default path), `~/medclip_work/pretrained/medclip-vit/` (zip + weights,
1.05 GB), `~/medclip_gate.{py,log}`.

---

## Decision update (user, 2026-09-24, after the report above)

1. Move old runs off `/home` to the other disk (`/mnt/drive1tb`, remounted rw
   by the user, Windows removed) instead of shrinking the cache.
2. The shared projection before the Q-Former **trains** in 1a.
3. Phase lengths **by epoch**.
4. ITC label smoothing **0.1**; ITC queue **off**.
5. MedCLIP preprocessing as the processor/paper (pad to square).
6. Dropout 0.2 in 1b: yes.
7. Gate with R@1/R@5 as proposed.
8. MHCAC as the paper: element-wise Bernoulli text mask (Eq. 3, no rescale),
   text in the first layers only, progressive (un)freezing in 1b, and
   **remove the teacher/student branch**.

## Execution report — 2026-09-24 20:45–21:15, planning checkout + host `minhphuong`

### Disk

`~/archive_runs.sh` (log `~/archive_runs.log`) moved 20 run directories
(~110 GB) to `/mnt/drive1tb/archive-home/`, each verified by file count and
total bytes before the original was removed and a symlink left in its place;
0 mismatches. `/home` free: 152.6 GB → **250 GB**; `/mnt/drive1tb` 212 GB free.
fstab now mounts it `ntfs3 rw,uid=1000,gid=1000,nofail`.

### MedCLIP Swin (decision 2 a–g)

- **a** Weights load: 231/231 vision keys, 0 missing/unexpected; 219/219 float
  tensors differ from ImageNet (see the earlier report). Runtime loads
  `pytorch_model.bin` directly (`vision_encoders/swin/medclip_swin.py`,
  strict); the `medclip` package is used only for the check, from
  `~/medclip_pkg` via `PYTHONPATH` — site-packages untouched.
- **b** New backend `medclip` in `SwinEncoder`: HF `SwinModel`
  `last_hidden_state` [B,49,768] + `pooler_output` [B,768] → **[B,50,768]**,
  pooled first; `projection_head` unused. The inherited
  `vision_encoders/medclip/medclip.py` bug is pinned by a test (batch 2 and
  3 both fail) and recorded in DECISIONS D-020.
- **c** Preprocessing reimplemented (pad square black → 224 bicubic → /255 →
  MedCLIP mean/std → 3 channels) from the raw image, never the BioViL tensor.
  ⚠ The `medclip` package's own processor **cannot run under transformers
  4.53** (positional arguments shifted: `rescale_factor`=mean,
  `image_mean`=False, then `AttributeError`). The reference was produced by
  the real processor under **transformers 4.24.0** in a separate venv
  (`~/medclip_work/ref_venv`) on synthetic images; ours matches to **< 1e-5**
  on 4 shapes (portrait, landscape, square, small).
- **d** Projection: paper "linear to 1408"; original code MLP 768→1024→1408;
  this repo one `nn.Linear(768, 1408)` in `SharedVisualTokenProjector` — not
  changed. The 10% mask applies to this stream like the others.
- **e** Frozen in every phase (the apply_trainable tests pin it); cached with
  phase 1a: +76.8 KB per image, **+17.2 GB** for the anchor-only train+val cache.
- **f** Leakage: MedCLIP paper says "training split" but its Table 3 counts
  377,111 images (all of MIMIC-CXR); unverifiable → limitation in README and
  D-020.
- **g** Swin extraction (bf16, no_grad, 5060 Ti): **0.81 ms/image at batch
  16, 0.87 at 32, 0.96 at 64** (5.1 ms at batch 1); peak extra VRAM 214 / 427 /
  854 MiB; weights 161 MiB (27.5M params). Tokens into the Q-Former and MHCAC:
  **196 + 50 + 50 = 296** (MHCAC raises if any stream disagrees with its
  layout; the smoke ran without that error).

### Code, by file

| file | change |
|---|---|
| `pretraining/phases.py` (new) | `run.phases` schema, `apply_phase_to_config` (merges run/model/datasets, sets `max_epoch`, disables encoder_finetune), `apply_trainable`, transition roles/multipliers, `checkpoint_keep`, `resolve_init_checkpoint` |
| `pretraining/itc_gate.py` (new) | shared gate: `delta_nats`, ranks, R@1/R@5 + exact binomial threshold; `meets_threshold` needs both |
| `pretraining/train.py` | apply phase before build; `prepare_phase_model` (init from previous phase, unexpected keys → error; requires_grad) |
| `model/lavis/runners/runner_base.py` | role-aware param groups, `_on_phase_update` / `_finish_transition` (freeze + drop groups and Adam state), checkpoints keep all phase-trained params, `checkpoint_<phase>.pth` to output_dir and phase_root, per-epoch ITC gate with STOP, `phase_metrics.jsonl`, gradient-interference hook, resume after hand-over |
| `model/lavis/tasks/base_task.py` | `pre_backward` hook (measurement only) |
| `model/lavis/common/optims.py` | `_set_lr` × `phase_lr_mult` |
| `model/lavis/models/blip2_models/blip2_qformer.py` | `encode_samples`; MedCLIP Swin input + native layout; 10% mask; single-path MHCAC text; `needs_mhcac` (1a skips MHCAC); ITC smoothing; dead `_create_mask` removed |
| `mhcac/mhcac_12.py` | `layer_order` (self_first = paper), `text_mask_mode` (element = Eq. 3), `text_row_mask`; defaults reproduce the old layer exactly |
| `mhcac/loss.py` | `smoothed_cross_entropy` (smoothing over finite candidates only) |
| `vision_encoders/swin/{swin_encoder,medclip_swin}.py` | backend `medclip` |
| `vision_encoders/feature_mask.py` (new) | per-encoder, per-sample token mask |
| `model/lavis/data/ReportDataset.py` | `swin_image` / `aux_swin_image`; collater generalised |
| `pretraining/precompute_features.py` | `--anchor-only`, `--truncate`, disk check before writing, swin input |
| `model/lavis/tasks/image_text_pretrain.py` | `report_study_presence` → `sp_*` (12/13/14-label) metrics per epoch |
| `scripts/check_itc_gate.py` | uses the shared gate; prints R@k |
| `scripts/run_stage1_phases.sh`, `scripts/phase_report.py`, `scripts/make_medclip_preprocess_reference.py` (new) | driver, report, fixture generator |
| `pretraining/configs/mimic_cxr_full.yaml` | Swin MedCLIP on, `feature_mask_ratio 0.1`, MHCAC single_path/self_first/element, teacher/distill 0, queue 0, ITC smoothing 0.1, `run.phases` |
| tests | new `test_phases.py`, `test_paper_mode.py`, `test_medclip_swin.py` (+ fixture); `test_loss_weight_gating.py` updated for D-020 |

### Trainable parameters by phase (measured in the smoke logs)

| phase | trainable | detail |
|---|---:|---|
| 1a | **188.87M** | Qformer 186.28M, shared_visual_projector 2.17M, query_tokens 0.02M, vision_proj 0.20M, text_proj 0.20M, itm_head, ln_vision, temp |
| 1b start | **310.93M** | 1a set + mhcac 80.57M, view_fusion 38.00M, mpc_heads 1.91M, stream_adapters 1.58M |
| 1b after hand-over | **122.06M** | froze 188.87M (META-Former + projection) after 3 updates (10% of the smoke's 32) |
| 1c | **313.64M** | 1b-start set + visual_encoder (BioViL projector) 2.71M |

### Cache and disk

Anchor-only, fp16, per study: BioViL 196×1408 + PubMedCLIP 50×768 + Swin
50×768 = 705,536 B. Train 222,758 + val 1,808 studies → **158.4 GB**; free
on `/home` **250 GB**, so it fits (the builder re-checks and stops before
writing if not). Throughput of the build, all three encoders: ~72 images/s
→ ~52 min for train. Smoke cache (2,000 train + 1,808 val): 1.4 + 1.3 GB.

### Tests

- CPU box: failure set unchanged from the gate-off baseline (the same missing
  torchvision/transformers ones) plus nothing new; the two
  `test_loss_weight_gating` assertions that pinned the teacher were updated.
- Host, snapshot `~/metaformer3_src` (a git repo so the git tests run):
  **1164 passed, 2 skipped, 0 failed** (log `~/metaformer3_pytest.log`),
  including the host-only runner, layout and real-MedCLIP-weights tests.

### GPU smoke (2,000 train studies, val 400, 1 epoch per phase)

Every launch was guarded (no other trainer, GPU idle, new output dir) and
launched once.

| phase | batch | result | max mem | s/it | notes |
|---|---|---|---:|---:|---|
| cache | 16 | ok | — | ~72 img/s | anchor-only |
| 1a | 32 | **OOM, iteration 0** | — | — | in ITM (96 seqs × (32 + 256) tokens) |
| 1a | 24 | **OOM, iteration 0** | — | — | |
| 1a | 16 | **OOM, iteration 0** | — | — | |
| 1a | 8 (measurement only) | ok, 250 iters | **12,456 MiB** | 0.366 | |
| 1b | 16 × 4 | ok, 125 iters | 8,888 MiB | 0.274 | hand-over at update 3 |
| 1c | 16 × 4 | **OOM, iteration 0** | — | — | in the Q-Former LM pass |
| 1c | 8 × 8 (measurement only) | ok, 250 iters | **15,043 MiB (97%)** | 0.454 | also with the interference hook: 15,037 MiB, 0.469 s/it |

Losses, first → last iteration: **1a** `loss_itc` 3.103 → 1.544,
`loss_itm` 1.668 → **0.6365** (= the 1:2 prior entropy, the historical
collapse value), `loss_lm` 7.433 → 3.658; all classification terms 0.0000.
**1b** `loss_cls` 1.333 → 1.048, `loss_contrastive` 0.305 → 0.177,
`loss_orthagonal` 0.103 → 0.0024, `loss_mpc` 4.70 → 2.12, ITC/ITM/LM 0.
**1c** total 7.315 → 6.670, `loss_itc` 1.819 → 1.506, `loss_itm` 0.746 →
0.629, `loss_lm` 3.671 → 3.441, `loss_cls` 1.042 → 1.042. No NaN/inf. The 1c
run with and without the measurement hook produced **identical** losses to
4 decimals: the hook does not touch the optimizer step.

The 1c OOM at batch 8 × 8 seen first was the measurement hook's two flattened
gradient copies; it now accumulates per tensor and fits (above).

ITC gate (128 val pairs; smoke numbers, not a result):

| phase | delta_nats | ranks i2t/t2i (chance 63.5) | R@5 i2t/t2i (chance 0.039) | pass |
|---|---:|---|---|---|
| 1a, 250 updates at batch 8 | +0.431 | 35.7 / 33.1 | 0.109 / 0.109 | yes |
| 1c | +0.605 | — | 0.117 / 0.078 | no (t2i 10 hits, below the 12-hit threshold) |

H — gradient interference in 1c (7 measurements, updates 0–30):
cosine between classification and alignment gradients ≈ 0 on every shared
group (means −0.008 … +0.018), and the alignment gradient is **80–1,500×
larger** in norm (median ratios: stream_adapters 80, text tower 186, shared
projector 322, BioViL projection head 887). Smoke only; nothing was changed
because of it. Classification deltas 1c vs end of 1b on 400 val studies are
±0.005 AUROC and not interpretable at this size.

### Mismatches / decisions still open

1. **The batch the plan requires does not fit.** Phase 1a needs batch ≥ 32
   and OOMs at 32/24/16; only 8 fits — the batch at which ITC measured chance
   four times. Per the plan, GradCache/SigLIP were NOT enabled. Levers the
   user can choose from: `max_txt_len` 256 → shorter (ITM and LM scale with
   it), gradient checkpointing in the Q-Former, GradCache, SigLIP, or batch 8.
2. **Phase 1c at the configured 16 × 4 OOMs**; 8 × 8 fits at 97% of the card.
3. **ITM sits at 0.6365** at the end of the 1a smoke — the collapse value.
4. The paper's 20,000 steps do not match the epoch counts (recorded before).
5. The 1a gate PASSED on the smoke (128 pairs) — the first above-chance ITC
   measurement in this repo — but n=128 and 250 updates settle nothing.
6. `phase1c.unfreeze_encoder_blocks` is `false` (paper); the user's call.

### Not done

- No full run of any phase (as instructed). The full anchor-only cache is not
  built yet (~158 GB, ~1 h) — it waits for the batch decision.
- Nothing pushed.

---

## Decision update (user, 2026-09-24) — D-021

1. For the batch: Q-Former **gradient checkpointing**, **GradCache** and
   **SigLIP**, all three.
2. Phase 1c at batch 8 × 8: ok.
3. `phase1c.unfreeze_encoder_blocks`: **true** — try it.
4. Push.

## Execution report — 2026-09-24 (continued session), planning checkout + host

Pushed `08592d3` and `9350440` to `origin/feat/stage2-finding-tokens` first.

### What was built

- **Checkpointing** (`model.qformer_grad_checkpointing: true`): the LAVIS
  `BertEncoder` branch, switched to `use_reentrant=False`. Two defects found
  and fixed on the way: (a) checkpointing turns off the query KV cache, and
  `BertEncoder` then returns an **empty tuple**, not `None` — the LM term now
  re-feeds the queries and the image when the cache is empty, instead of
  crashing (or, had the check been `is None` on another code path, training the
  LM without the image); (b) autocast's weight-cast cache made the backward
  recompute record extra casts and `torch.utils.checkpoint` aborted with
  "Recomputed values ... different metadata" — the train loop now disables the
  cache when checkpointing is on.
- **SigLIP** (`model.loss.itc_loss: sigmoid`): pairwise sigmoid over valid
  pairs, learnable scale (init log 10) and bias (init −10). `temp` and ITC label
  smoothing apply only to `softmax`. ITM hard negatives use the SigLIP logits.
- **GradCache** (phase 1a, `gradcache_chunk_size: 16`, batch 128): no-grad
  features for the whole batch → ITC over 128 → gradients cached on the
  features → per chunk, RNG replayed, features recomputed with grad, cached
  gradient + ITM/LM (normalised by the whole batch) backpropagated. Negatives
  for ITM drawn from the whole batch; out-of-chunk negative images re-encoded.
  Refuses MHCAC objectives and fp16 GradScaler.
- **Phase 1c** batch 8 × 8, `unfreeze_encoder_blocks: true` with the shallow
  set (BioViL layer4 + projector, CLIP blocks 10–11 + post_layernorm).

### Tests

- `tests/test_gradcache_siglip.py` (10; 6 need transformers → host only), on a
  tiny real LAVIS Q-Former: chunk 2 vs chunk 6 gradients equal to < 1e-4 of each
  tensor's scale (both SigLIP and softmax); GradCache == ordinary backward on
  ITC + LM; LM without the cache == LM with it; checkpointing does not change
  gradients; SigLIP matches the published formula, masks invalid pairs, trains
  its scale and bias.
- Host (git snapshot): **1174 passed, 2 skipped, 0 failed**
  (`~/d021_pytest.log`). CPU box: unchanged baseline failures only.

### GPU smoke (2,000 train studies, 1 epoch per phase, one guarded launch;
log `~/smoke_3phase_d021b.log`)

| phase | setting | max mem | s/it | trainable |
|---|---|---:|---:|---:|
| 1a | batch 128 = 8 × chunk 16, GradCache + checkpointing + SigLIP | **7,221 MiB** | 8.03 (0.063 s/study) | 188.87M |
| 1b | 16 × 4 | 6,466 MiB | 0.23 | 310.93M → hand-over after 3 updates |
| 1c | 8 × 8, checkpointing, shallow encoder unfreeze | **9,337 MiB** | 0.63 | **342.78M** (+ visual_encoder 17.67M, pubmedclip 14.18M) |

Before D-021 the same phases OOMed (1a at 32/24/16; 1c at 16 × 4) or used
15,043 MiB (1c at 8 × 8 without checkpointing or unfreeze). Full-data estimate
for 1a: 1,740 updates/epoch × 8.0 s ≈ **3.9 h/epoch**, ≤ 4 epochs.

Losses, first → last: 1a (15 updates) SigLIP `loss_itc` 8.72 → 7.26,
`loss_itm` 1.47 → 0.79, `loss_lm` 7.19 → 7.05; 1b `loss_cls` 1.333 → 1.050;
1c `loss_itc` 6.89 → 2.70, `loss_itm` 1.20 → 0.63, `loss_lm` 6.87 → 5.47,
`loss_cls` 1.051 → 1.046. No NaN/inf, no traceback.

ITC gate (128 val pairs): 1a ranks 62.6 / 63.2 (chance 63.5), R@5 0.063 /
0.055 — at chance, expected after 15 updates still inside the 500-update
warm-up; 1c ranks 51.1 / 47.1, R@5 0.047 / 0.055, not above chance. Smoke only.

### Open

- The full run: build the 158 GB anchor cache, then
  `ROOT=$HOME/run_<date>_3phase bash scripts/run_stage1_phases.sh`. Not started.
- Phase 1a now has headroom (7.2 of 15.5 GB): chunk 32 would be faster; not
  changed without a decision.

---

## Go-ahead (user, 2026-09-24) — conditions

1. Before the cache: report free space on /home and the checkpoint estimate;
   stop if < ~50 GB would remain.
2. Explain whether the GradCache chunk changes ITC/ITM negatives; raise it to 32
   either way (record in DECISIONS if it does); 50-step smoke, peak < 14 GB.
3. Confirm the phase-1a gate after epoch 2 stops the WHOLE script, and that a
   phase resumes from a mid-phase checkpoint.
4. No other GPU job during the run.
5. Report the gate as soon as 1a finishes epoch 2, and the H table after 1c.

## Execution report — 2026-09-24 22:00–22:30, before the full run

1. **Disk.** Free on /home 218.4 GB. Estimate from the smoke files: per phase
   `checkpoint_last` (with optimizer) 2.60 / 2.23 / 4.00 GB, `checkpoint_best`
   1.09 / 1.26 / 1.37, `checkpoint_<phase>` twice (output dir + phase_root)
   2.18 / 2.51 / 2.74, plus 1b's `checkpoint_4` (save_freq 5) 2.23 → **~22 GB**,
   ~26 GB with one in-flight `.tmp`. Cache 158.4 GB → only ~34 GB would remain,
   below 50. The deficit was this session's own smoke/verify directories
   (~46 GB): moved to `/mnt/drive1tb/archive-home/` with file-count and byte
   verification (`~/archive_smoke.sh`), 0 mismatches. Free now **268.0 GB** →
   ~83 GB remains after cache + checkpoints. The cache builder is also called
   with `--min-free-gb 76` so it refuses to write if that changes.
2. **Chunk.** Pure computation split: ITC scores the full 128 × 128 matrix,
   ITM negatives are drawn from the whole batch's similarities (out-of-chunk
   negative images are re-encoded), LM is normalised by the whole batch's
   tokens; chunk 2 vs 6 gradients agree to < 1e-4 of scale. Only dropout masks
   are drawn in a different pattern (statistically equivalent). Negatives do
   not change → no DECISIONS entry; raised to 32 with a YAML comment.
3. **Gate stop and resume, verified on GPU** (smoke cache, chunk 32,
   `~/verify.log`): (A) gate forced to fail (`min_delta=100`, 3 epochs
   configured): after epoch 2 the runner wrote `PHASE_GATE_FAILED`, epoch 3 did
   not run, no `checkpoint_phase1a`, driver **exit 3**, phase 1b never
   launched. Fixed on the way: the stop check was `>=`, so a gate passing at
   epoch 2 and dipping later would also have stopped the run; it is now `==`.
   (B) process group killed after the mid-epoch checkpoint at iter 10;
   `RESUME=1` resumed from `checkpoint_last` ("mid-epoch ... restarting that
   epoch from its first batch"), finished, wrote `checkpoint_phase1a`. Driver
   gained `RESUME=1` for this.
4. Pipeline `~/pipeline_3phase.sh` (log `~/pipeline_3phase.log`) launched once
   at 22:26: cache → 50-update smoke at chunk 32 (stops if peak ≥ 14,000 MiB) →
   full run in `~/run_20260925_3phase` (log `~/run_20260925_3phase.log`).

## Full run — `~/run_20260925_3phase` (started 2026-09-24 23:20)

- Cache: 158.47 GB (222,758 train + 1,808 val anchors) in 47 min; /home free
  after it 109.5 GB.
- 50-update smoke at chunk 32 on the full cache: peak **9,575 MiB** (< 14,000),
  8.12 s/it. Run launched.

### Phase 1a — ITC gate (256 val pairs, valid fraction 0.615) — PASSED at epoch 2

| epoch | delta_nats | mean rank i2t / t2i (chance 127.5) | R@1 i2t / t2i | R@5 i2t / t2i (chance 0.0195; ≥ 12 hits) | pass |
|---:|---:|---|---|---|---|
| 1 | +2.569 | 11.14 / 13.59 | 0.258 / 0.266 | 0.602 / 0.547 | yes |
| **2** | **+2.859** | **8.77 / 10.44** | **0.328 / 0.285** | **0.652 / 0.613** | **yes** |
| 3 | +2.976 | 8.11 / 10.20 | 0.383 / 0.281 | 0.672 / 0.641 | yes |

First above-chance ITC in this repository (four previous measurements, all at
chance, at batch 8 with softmax InfoNCE and a live-encoder queue). SigLIP
temperature equivalent 0.088 → 0.077.

Epochs 3:52:31 / 3:52:33 / 3:52:36 at 8.02 s/it; GPU ~12.1 GB, 70 °C; no
NaN/inf. Train losses per epoch: ITC (SigLIP) 3.66 / 2.87 / 2.54, ITM 0.60 /
0.55 / 0.52, LM 2.83 / 2.40 / 2.33.

⚠ Val `loss_itm` 0.635 / 0.651 / 0.692 against a chance value of 0.6365: ITM
does not generalise even as retrieval improves; val SigLIP loss 3.08 / 3.04 /
3.12 is flat. Recorded, nothing changed.

### Stopped on 2026-09-25 ~15:50, in phase 1b — for D-022

Phase 1a finished all 4 epochs (`checkpoint_phase1a.pth` written); phase 1b was
at epoch 0, iteration 5,550 / 13,922, when the user chose to apply D-022
(Uncertain as a trained third class; PubMedCLIP on its own CLIP preprocessing)
and rerun all three phases. The pipeline, the phase driver and the trainer were
killed; GPU back to 117 MiB. No 1b/1c result exists. The 1a checkpoint and gate
JSONs stay in `~/run_20260925_3phase` as the record of the gate above; they
are not reused, because PubMedCLIP's input changed under the Q-Former.

The rerun reuses the BioViL and Swin caches (their inputs did not change) and
rebuilds only PubMedCLIP's with `precompute_features.py --encoders pubmedclip`.

## Full run with D-022 — `~/run_20260925b_3phase` — COMPLETED 2026-09-26

Pipeline `~/pipeline_d022.sh` (snapshot `~/metaformer3b_src` = commit `81cc9a9`):
PubMedCLIP cache rebuilt 15:56 → 17:15 (BioViL/Swin caches reused by hard
link, ids identical across the three encoders); three-phase smoke peak
9,622 MiB, no NaN/inf; full run 2026-09-25 17:26 → 2026-09-26 21:40, `rc=0`,
no Traceback. 1a 15h36m, 1b 7h53m, 1c 4h45m.

### ITC gate (256 val pairs)

| phase | epoch | delta_nats | R@1 i2t / t2i | R@5 i2t / t2i | pass |
|---|---:|---:|---|---|---|
| 1a | 0 | +2.690 | 0.305 / 0.270 | 0.613 / 0.613 | yes |
| 1a | 1 | +2.935 | 0.352 / 0.285 | 0.633 / 0.617 | yes |
| 1a | **2 (gate)** | **+2.931** | 0.379 / 0.316 | **0.664 / 0.641** | **yes** |
| 1a | 3 | +2.934 | 0.344 / 0.332 | 0.668 / 0.617 | yes |
| 1c | 0 | +2.759 | 0.285 / 0.277 | 0.578 / 0.606 | yes |

### Section H — val, `study_presence` + `q_pos`, threshold 0.5

| | macro AUROC 12 / 13 / 14 | macro AUPRC 12 | pos. macro F1 12 |
|---|---|---:|---:|
| 1b epochs 0..4 | 0.7705, 0.7849, 0.7875, 0.7867, 0.7869 (12) | 0.3103 → 0.3198 | 0.2910 → 0.3200 |
| end of 1b | 0.7869 / 0.7979 / 0.8030 | 0.3198 | 0.3200 |
| 1c | 0.7851 / 0.7959 / 0.8011 | 0.3188 | 0.3215 |

Phase 1c moves classification by -0.002 AUROC, i.e. nothing. Gradient
interference, 18 measurements over updates 0..3400: mean cosine
+0.003 (BioViL head), +0.019 (Q-Former text tower), +0.000 (shared projector),
-0.004 (stream adapters); alignment/classification norm ratio 10-95x (median
8-40x). Orthogonal, and alignment dominates -- the D-020 smoke signature.

### Test, from the 1c checkpoint (n=3,269; CPU, 2026-09-26)

`scripts/calibrate_thresholds.py` on 1c val, then `scripts/evaluate_stage1.py`
on 1c test, both with `--label-framing study_presence --uncertain-policy
three_class`, default score `conditional_positive`, `--selection plateau
--plateau-fraction 0.95 --min-positive 5`. Outputs in `~/eval_d022/` (private).

| | macro AUROC | micro AUROC | macro AUPRC | pos. macro F1 | precision | recall | specificity |
|---|---:|---:|---:|---:|---:|---:|---:|
| 12 labels | 0.7642 | 0.8460 | 0.3220 | 0.3498 | 0.2979 | 0.4616 | 0.8325 |
| 13 / 14 labels (AUROC) | 0.7746 / 0.7773 | | | | | | |
| `run_20260820_ft` (reported model) | 0.7643 | 0.8166 | 0.3203 | 0.3542 | 0.2931 | 0.5373 | 0.8020 |

⚠ Not a controlled comparison: the old row is `marginal_presence` with the
mention gate trained and blanks masked; this row is `q_pos` with the gate off,
blanks negative, three phases and D-022. No paired bootstrap has been run.
Read as: macro AUROC/AUPRC essentially equal, micro AUROC higher (the
cross-label-calibration signature seen with the joint objective), operating
point toward specificity. `Fracture` gets 1 TP of 89 at its calibrated
threshold -- threshold noise on a rare label, check AUROC before reading it.

Host note: after the run `nvidia-smi` reported `Driver/library version
mismatch` (driver updated underneath); the user rebooted 2026-09-26 22:06,
GPU and `ntfs3` verified after.
