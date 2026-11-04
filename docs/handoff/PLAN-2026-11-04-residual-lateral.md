# Multi-view: OFF đóng băng + residual ở lớp đầu ra (chuyên gia lateral)

## Trạng thái

- Ngày tạo: 2026-11-04. Commit lúc bắt đầu: `9c33aa5` (nhánh `feat/stage2-finding-tokens`).
- 2026-11-04: user đã trả lời mục 13 (xem "Quyết định của user"). Bước 0.1 (code CPU) XONG,
  chưa commit. Chưa chạy GPU. Đang chờ user duyệt commit/push và lượt GPU Bước 0.2.
- Ghi chú đối chiếu code (2026-11-04, đọc trên checkout planner):
  - `holdout_split` nằm ở `scripts/swin_finetune_fit.py:58`, không phải `encoder_probe_fit.py`;
    `_masked_ce` có ở cả hai (`encoder_probe_fit.py:95`).
  - `build_attention_probe` (`encoder_probe_fit.py:101`), `warmup_cosine` / `augment` /
    `to_swin_input` (`swin_finetune_fit.py`), `swin_image_to_uint8` (`swin_finetune_cache.py:41`),
    `fit_severity_cutpoints` / `apply_cutpoints` (`threshold_calibration.py`),
    `build_study_index` (`model/lavis/data/mimic_cxr_utils.py:31`), `build_sample_keys`
    (`training/evaluation/schemas.py:201`) đều tồn tại.
  - `encoder_probe_features.py` đã có `--raddino` / `--medsiglip`.
  - `vision_encoders/lora.py` chưa có.

## Plan (prompt gốc, chép nguyên văn)

# NHIỆM VỤ: Multi-view theo hướng "OFF đóng băng + residual ở lớp đầu ra" (kèm Bước 1: thích nghi encoder cho ảnh lateral)

Repo: `~/Meta-CXR` (github.com/minhphuong150505/Meta-CXR), nhánh `feat/stage2-finding-tokens`. HEAD lúc viết prompt là `9c33aa5`; ghi lại sha thực tế khi bắt đầu.

Đọc trước khi làm bất cứ việc gì:
- `CLAUDE.md`, nhất là mục USER GOAL 2026-10-25 và D-023 (ba lớp). Đọc thêm `AGENTS.md` và `docs/handoff/README.md`.
- `.claude/skills/multi-view-novelty/SKILL.md`. Rule 1 và Rule 2 trong đó áp dụng nguyên vẹn cho prompt này.
- `docs/KET_QUA_MULTI_VIEW.md` và các plan sau:
  - `PLAN-2026-10-24-multi-view-ablation.md`: bẫy phase override, đường dẫn của OFF.
  - `PLAN-2026-10-26-pilot-seed-noise.md`: công thức NB.
  - `PLAN-2026-10-27-multi-view-evidence-val.md`: nhóm PL, mẹo anchor lateral, re-key.
  - `PLAN-2026-10-31-encoder-probe.md` và `PLAN-2026-10-31-swin-finetune-probe.md`.

Tạo file bàn giao `docs/handoff/PLAN-<YYYY-MM-DD>-residual-lateral.md` và chép nguyên prompt này vào phần Plan. Sau mỗi bước, nối `## Execution report — <bước>, <ngày>` vào cùng file, theo mẫu trong `docs/handoff/README.md`.

## 0. Quy tắc bắt buộc (không có ngoại lệ)

1. **Hỏi trước.** Trước mỗi lượt chạy GPU, và trước mỗi thay đổi về code, YAML, loss, recipe hay quy tắc đọc kết quả: DỪNG và hỏi user bằng tiếng Việt, dạng văn bản thường (không dùng picker AskUserQuestion). Nêu rõ thay đổi gì, vì sao, tốn bao nhiêu GPU, và sẽ đọc kết quả thế nào. Chờ user đồng ý rõ ràng. User duyệt bước này không có nghĩa là duyệt bước sau.
2. **Chọn trên val.** Mọi lựa chọn đều làm trên val. Test chỉ đọc MỘT lần, cho ứng viên cuối, sau khi user duyệt (Bước 3).
3. **Reading rule chốt trước.** Ghi reading rule (mục 7) vào file bàn giao và để user chốt TRƯỚC khi có kết quả đầu tiên. Không sửa sau khi đã thấy số.
4. **Ba lớp (D-023) ở mọi nơi:** model, loss, chỉ số báo cáo. "AP present" chỉ là số mô tả cho probe, để nối với encoder probe 2026-10-31.
5. **Không có dữ liệu bệnh nhân** trong repo, chat hay commit message: không report text, không `subject_id`/`study_id`/`dicom_id`, không đường dẫn ảnh. Đặc trưng, cache, npz và checkpoint chỉ nằm trên máy train; chỉ số tổng hợp mới được đưa vào repo.
6. **Mọi hành vi mới là opt-in.** Thiếu khoá config thì chạy đúng hành vi cũ, bit-identical. Không sửa mô hình đang báo cáo `run_20261005_paper`, cũng không sửa OFF `run_20261024_nomv`.
7. **Guard trước MỖI lượt GPU:**
   - `findmnt -no FSTYPE /mnt/drive1tb` trả về `ntfs3`.
   - `nvidia-smi` đang rảnh.
   - Không có process `pretraining.train` hay `run_medgemma_qlora` (so khớp theo tham số dòng lệnh).
   - Thư mục output chưa tồn tại.
   - Còn đủ chỗ trống trên đĩa trước khi ghi cache.
   - Chỉ bấm chạy đúng MỘT lần.

## 1. Bối cảnh: vì sao chọn hướng này

- **Các biến thể trước đều không thắng OFF vượt NB:** early fusion (P15/P30/P50), cờ EVOKE, 3 aux view, Swin-only, late fusion (LF/LD/AH), tokens (TK). Mẫu lặp lại ở mọi lần: lợi nhỏ ở study hai view, mất nhỏ ở study một view, tổng gần 0. Xem `docs/KET_QUA_MULTI_VIEW.md`.
- **Nút thắt là biểu diễn.** Encoder probe cho thấy một probe attention trên MedCLIP-Swin đóng băng đạt ngang toàn bộ mô hình.
- **Các encoder đọc ảnh lateral kém, hoặc chưa rõ.**
  - BioViL và BioViL-T chỉ được pretrain trên ảnh frontal (arXiv 2204.09817 và 2301.04558).
  - Chưa xác minh được MedCLIP và PubMedCLIP có lọc view hay không.
  - RAD-DINO: checkpoint được train trên 882.775 ảnh. Số ảnh MIMIC (368.960) và CheXpert (223.648) gần bằng tổng ảnh mọi view của hai bộ, nên nhiều khả năng có lateral. Đây là suy luận từ số đếm; model card không ghi rõ.
- **CheXFusion (arXiv 2308.03968).** Backbone được train trên mọi view, kể cả lateral. Khi đó chỉ cần trung bình dự đoán frontal:lateral theo tỉ lệ 7:3 là mAP đã tăng từ 0,340 lên 0,362. LF trong repo cũng gần như lấy trung bình 0,5/0,5 mà không tăng gì, nên chỗ yếu là bộ dự đoán trên ảnh lateral.
- **MAIRA-2 (arXiv 2406.04449, RAD-DINO đóng băng).** Khi bỏ lateral (cùng phần Technique) lúc train, RadFact và Macro F1-14 gần như không đổi. Hiệu ứng thật của lateral rất nhỏ: từ +0,007 (van Tulder 2021) đến +0,016 AUROC (Hashir 2020).
- **Thiết kế mới vì vậy gồm ba ý:**
  - (a) giữ OFF nguyên vẹn và đóng băng;
  - (b) dùng một "chuyên gia lateral" riêng;
  - (c) chỉ cộng phần hiệu chỉnh vào logit đầu ra khi study có lateral.
- **Hệ quả theo cấu trúc:**
  - Study không có lateral cho ra đúng dự đoán của OFF, nên phần mất ở `views_1` biến mất.
  - ON − OFF chỉ có thể đến từ ảnh lateral.
  - OFF dùng chung cho mọi arm, nên nhiễu seed chỉ còn đến từ chuyên gia lateral.

## 2. Định nghĩa

- **OFF**
  - Checkpoint: `~/run_20261024_nomv/phase1c/mimic_cxr_full_blip2/checkpoint_best.pth`.
  - Dự đoán sẵn có: `.../result/{val,test}_predictions_epoch_best.npz`. Kiểm tên file bằng `ls`.
  - Cutpoint: `~/nomv_probe_20261024/cutpoints_nomv.json`.
  - Không train lại, không sửa.
- **Nhóm PL**
  - Gồm các study có anchor PA/AP và aux có view id 2 (LATERAL/LL).
  - Dựng lại bằng `build_study_index` (`model/lavis/data/mimic_cxr_utils.py`), với `anchor_priority=[PA,AP,lateral]` và `max_aux_views=1`.
  - Kỳ vọng: val PL 797, F1 930, other 81; test PL 1.196. Lệch thì DỪNG.
  - Mọi study ngoài PL, kể cả nhóm "other", luôn nhận đúng dự đoán của OFF.
- **Chuyên gia lateral `L_enc`**
  - Gồm encoder `enc` (đóng băng ở Bước 0, có LoRA ở Bước 1) và head `build_attention_probe` (`scripts/encoder_probe_fit.py`), dự đoán ba lớp.
  - Train bằng nhãn study, trên ẢNH LATERAL của các study PL trong train.
- **Chuyên gia đối chứng `C_enc`:** giống hệt `L_enc` (cùng encoder, cùng recipe, cùng tập study, cùng số ảnh), chỉ khác là đọc ẢNH FRONTAL của chính các study đó.
- **Residual R1 ở lớp đầu ra (mặc định)**
  - Với mỗi finding f và mỗi study PL: `z_ON = log p_OFF + Σ_E a[f,E] · log p_E + b[f]`, trong đó E chạy qua tập chuyên gia của arm, `a ≥ 0`, `b[f] ∈ R³`, `b[f,neg] = 0`.
  - Fit riêng từng finding bằng CE ba lớp có mask, cộng L2: `λ(Σa² + ‖b‖²)`. Mặc định `λ = 1e-2` trên CE trung bình [TUỲ CHỈNH]. Tối ưu bằng L-BFGS-B, `a ∈ [0, 3]`.
  - Điểm xuất phát `a = 0, b = 0` chính là OFF, và L2 kéo tham số về OFF.
  - **Fit trên VAL bằng cross-fitting hai nửa, đúng quy ước của cutpoint:** sắp theo `sample_key`, `fold = default_rng(16).permutation(n) % 2`, fit trên nửa này và chấm trên nửa kia. Với test thì fit trên toàn bộ val PL rồi áp lên test PL một lần.
  - Lý do fit trên val: logit của OFF trên train là in-sample nên quá tự tin, còn trên val thì không.
  - Lưu tham số giống file cutpoint: `configs/stage1_residual/<tên>.json`. File chỉ chứa a, b, λ và metadata, không có dữ liệu bệnh nhân.
- **Các arm so sánh ở mỗi bước**
  - OFF, R1(OFF+C), R1(OFF+L), R1(OFF+C+L).
  - Sanity: R1(OFF+L_perm), với logit của L bị xáo trộn giữa các study PL trong cùng fold.
- **Các hiệu cần đọc**
  - D1 = R1(OFF+L) − OFF: claim chính.
  - D2 = R1(OFF+C+L) − R1(OFF+C): phần đóng góp RIÊNG của lateral, ngoài hiệu ứng ghép thêm một mô hình.
  - D3 = R1(OFF+C) − OFF: hiệu ứng ensemble, chỉ để mô tả.

## 3. Bước 0 — Hạ tầng và mốc với encoder đóng băng (viết code trên CPU; khoảng 1–2 giờ GPU)

Mục tiêu: với encoder ĐÓNG BĂNG, xem chuyên gia lateral đã thêm được gì cho OFF chưa, và encoder nào đọc lateral tốt nhất.

### 0.1 Code (CPU, có test)

- **Trích đặc trưng lateral.** Viết `scripts/mv_probe_features.py`, hoặc thêm cờ `--views frontal,lateral` cho `encoder_probe_features.py` mà không đổi hành vi mặc định.
  - Trích đặc trưng ảnh LATERAL cho các study PL.
  - Cách lấy ảnh lateral: dựng dataset với `anchor_priority=[lateral,PA,AP]` và `max_aux_views=0` (mẹo của evidence package), rồi chỉ giữ study có anchor view id 2.
  - Phần frontal tái dùng `~/encprobe_20261031/`, nếu key và nhãn khớp (kiểm như gate G0 cũ).
- **Re-key.** `sample_keys` của eval hook băm `image_id` của ẢNH ANCHOR, nên lượt chạy lateral có key theo ảnh lateral.
  - Viết một hàm có test, ví dụ `training/evaluation/view_keys.py`. Hàm map mỗi study thành (key của frontal anchor, key của lateral), dùng `build_study_index` và `build_sample_keys`.
  - Mọi bảng đều nối theo key của frontal anchor.
  - OFF npz KHÔNG có `num_views`, nên nhóm phải dựng lại từ CSV. Kiểm `1 + len(aux) == num_views` với file của A, như evidence package đã làm.
- **Encoder:** `swin` (MedCLIP-Swin), `raddino` (`--raddino`), `medsiglip` (`--medsiglip`). Nếu dùng BioViL thì chỉ đọc theo cách đúng của probe (512 chiều), không dùng `legacy_reshape`. [TUỲ CHỈNH]
- **Tập dữ liệu:**
  - Train: cùng 40.000 study của encoder probe (`default_rng(16)`), chỉ giữ các study PL trong đó (khoảng 22.000). Holdout 10% theo `holdout_split`.
  - Val: toàn bộ val.
  - Test: không đụng tới.
- **Script residual.** Viết `scripts/residual_stack.py` để fit và chấm R1 trên npz (mục 2): cross-fit, các arm, bootstrap ghép cặp, ghi JSON tổng hợp.
- **Test CPU mới, chạy trên dữ liệu giả:**
  - re-key đúng trên bảng giả có các study PA+L, PA, AP+L và chỉ có L;
  - R1 với `a=0, b=0` cho ra đúng OFF;
  - mọi study ngoài PL bit-identical với OFF;
  - fold cross-fit tất định;
  - λ lớn thì tham số hội tụ về OFF;
  - arm hoán vị không học được gì (`a ≈ 0`) trên dữ liệu không có tín hiệu.

### 0.2 GPU (sau khi user duyệt)

Chạy smoke với `--limit 256` trước. Sau đó trích đặc trưng, fit `L_enc` và `C_enc` với 3 seed (16, 17, 18) cho mỗi encoder, rồi chấm trên val PL.

### 0.3 Báo cáo (chỉ số tổng hợp)

- Chất lượng encoder trên ảnh lateral (mô tả): AP_pos (one-vs-rest) và AP present của `L_enc` so với `C_enc` trên val PL, theo từng finding.
- D1, D2, D3 cho từng encoder trên chỉ số chính (mục 7): trung bình 3 seed, CI, NB.
- Trọng số `a[f,L]` theo từng finding, để thấy finding nào thật sự dùng lateral.

### Cổng G0 (đề xuất; user chốt)

- **GO-thẳng:** có encoder mà D1 và D2 đều qua quy tắc mục 7. Khi đó đã có ứng viên; Bước 1 vẫn chạy để tăng biên độ.
- **CẦN-BƯỚC-1:** D1 và D2 không qua với mọi encoder, nhưng `L_enc` kém `C_enc` rõ rệt, ví dụ AP_pos trung bình trên T thấp hơn quá 0,02. Nghĩa là encoder chưa đọc được lateral, nên làm Bước 1.
- **NO-GO-sớm:** `L_enc ≈ C_enc` nhưng D2 ≈ 0 với mọi encoder. Nghĩa là lateral không mang thông tin riêng đo được qua probe. Báo user trước khi làm Bước 1.
- **Chọn encoder cho Bước 1:** encoder có D2 lớn nhất trên val. Nếu hoà thì chọn encoder rẻ hơn (Swin).

## 4. Bước 1 — Thích nghi encoder cho lateral bằng LoRA (chỉ khi G0 cần, hoặc user muốn tăng biên độ)

### 1.1 LoRA tối giản trong repo

- Viết `vision_encoders/lora.py`. Venv Stage 1 không có `peft`, và không được thêm dependency mới.
- Bọc `nn.Linear` với rank r, alpha và dropout. A khởi tạo Kaiming, B = 0, nên ở bước 0 đầu ra bit-identical với encoder đóng băng.
- Các hàm cần có: `apply_lora(module, target_names, r, alpha)`, `lora_parameters()`, `merge()`.
- Test:
  - B = 0 thì đầu ra trùng từng bit;
  - chỉ tham số LoRA và head có `requires_grad`;
  - sau `merge()` đầu ra không đổi (sai số ≤ 1e-5).

### 1.2 Target

In tên module trên CPU để xác nhận trước khi chạy.

- **MedCLIP-Swin** (HF `SwinModel`, Swin-T): `attention.self.query` và `attention.self.value` ở stage 3–4 (`encoder.layers.2`, `encoder.layers.3`). Tuỳ chọn thêm `key` và `attention.output.dense`. Mặc định r=8, alpha=16, dropout 0,05. [TUỲ CHỈNH]
- **RAD-DINO** (HF `Dinov2Model`): `attention.attention.query` và `value` ở 4 block cuối.
- **MedSigLIP** (`SiglipVisionModel`): `self_attn.q_proj` và `v_proj` ở 4 layer cuối.
- **Biến thể "adapter" để so sánh:** chỉ mở khoá stage cuối cùng các LayerNorm (partial fine-tune). [TUỲ CHỈNH]

### 1.3 Dữ liệu

- `L` dùng ảnh lateral của TẤT CẢ study PL trong train (khoảng 120.000). `C` dùng ảnh frontal của đúng các study đó.
- Cache uint8 theo tiền xử lý gốc của từng encoder:
  - Swin ở 224: tái dùng `swin_image_to_uint8` của `swin_finetune_cache.py`, khoảng 6 GB.
  - RAD-DINO ở 518: rất nặng, khoảng 32 GB cho 120.000 ảnh. Hỏi user nên dùng subset hay decode on-the-fly.
- Kiểm dung lượng đĩa trống trước khi ghi cache.

### 1.4 Train

Tái dùng `swin_finetune_fit.py`: `holdout_split`, `augment`/`to_swin_input`, `warmup_cosine`, `_masked_ce`.

- **Ba arm, cùng dữ liệu, cùng head, cùng budget:**
  - `L_frozen`: encoder đóng băng, giống arm F của swinft, trên cùng khoảng 120.000 ảnh. Arm này cần để phép so LoRA không bị lẫn với hiệu ứng tăng dữ liệu.
  - `L_LoRA`.
  - `C_LoRA`.
- Mỗi arm 3 seed.
- **Siêu tham số:**
  - Head là `build_attention_probe`.
  - LR của head 3e-4, của LoRA 1e-4 đến 2e-4; weight decay 0,05; batch 64; bf16.
  - Tối đa 8 epoch, early stopping theo holdout của train (patience 2), vì lần fine-tune Swin trước đã học thuộc từ epoch 3.
  - Không lật ngang ảnh lateral, vì lật sẽ đảo trước và sau.
- **`report.json`** ghi: lịch sử holdout, epoch được chọn, max mem, mức thay đổi trọng số tương đối. Gate giống G2 của swinft: tham số LoRA có đổi, backbone gốc bit-identical.

### 1.5 Chấm trên val PL

Chấm như Bước 0, nhưng thay bản đóng băng bằng `L_LoRA` và `C_LoRA`: D1, D2, D3, NB.

### Cổng G1 (đề xuất)

- (a) `L_LoRA − L_frozen`: AP_pos trung bình trên T có CI dưới > 0 và vượt NB/√3.
- (b) D2 với `L_LoRA` qua quy tắc mục 7.
- Qua (a) nhưng không qua (b): encoder đọc lateral tốt hơn, nhưng thông tin đó không bổ sung được cho OFF. Báo user.
- Không qua (a): dừng nhánh LoRA.

## 5. Bước 2 — Residual trên OFF thật

### 2a. R1 (mặc định; CPU, vài phút)

- Chạy R1 với chuyên gia tốt nhất từ G0 hoặc G1.
- **Cutpoint cho chỉ số headline:**
  - `views_1` và "other" dùng cutpoint của OFF.
  - Nhóm PL fit cutpoint RIÊNG trên val PL cho CẢ OFF lẫn ON, dùng cross-fit cho lượt đọc val.
  - Tái dùng `fit_severity_cutpoints` và `apply_cutpoints` trong `training/evaluation/threshold_calibration.py`; không viết thuật toán mới.

### 2b. Residual mức đặc trưng R2 (tuỳ chọn; chỉ khi 2a qua VÀ user duyệt)

- **Công thức:**
  - Δ = H(token lateral từ `E_L`, ngữ cảnh frontal từ OFF).
  - H nhỏ: 14 query, 1–2 lớp cross-attention, d = 256. Linear cuối khởi tạo bằng 0, nên ở bước 0 mô hình đúng là OFF.
  - `z_ON = z_OFF.detach() + Δ`, chỉ áp cho study PL.
- **Thứ tự làm:**
  - Train trước trên đặc trưng đã CACHE, vì cả OFF lẫn `E_L` đều đóng băng.
  - Chỉ tích hợp thành `model.view_fusion.mode: residual` nếu R2 thắng 2a. Khi tích hợp: thêm phase `phase1r` với `init_from` = checkpoint OFF, `trainable` = head, `model.eval_mode_modules` = mọi module của OFF.
- **Bắt buộc đo trước:** OFF trên in-sample so với out-of-sample (CE và ECE trên khoảng 10.000 study train so với val).
  - Nếu `CE_train < 0,9 × CE_val`, logit OFF trên train quá tự tin.
  - Khi đó đề xuất cho user một trong ba cách: (i) giữ R1; (ii) cross-fit OFF 2 fold, tức 2 lượt 1b+1c trên nửa dữ liệu, ước khoảng 10 giờ; (iii) hiệu chỉnh nhiệt độ.
- **Test CPU cho mode mới:**
  - lúc khởi tạo, mô hình ≡ OFF;
  - study một view vẫn bit-identical sau vài bước train;
  - không có gradient chảy vào tham số của OFF;
  - `resolve_view_fusion_mode` chỉ nhận `residual` khi `multi_view: true`, và từ chối MPC, view consistency, explanation.

## 6. Bước 3 — Đọc test một lần (sau khi user duyệt ứng viên cuối)

- Chạy suy luận L (và C, nếu claim D2) trên ảnh test PL. Fit R1 trên toàn bộ val PL rồi áp lên test PL. Mọi study khác nhận đúng dự đoán của OFF.
- Báo đủ mọi chỉ số và mọi nhóm (PL, `views_1`, toàn test), kể cả những số đi ngược kỳ vọng. Bootstrap ghép cặp 2.000 lần, seed 16.
- **Guard:**
  - dự đoán ngoài PL trùng OFF từng bit;
  - số study của các nhóm khớp với số đã ghi.

## 7. Reading rule (đề xuất; user chốt trước khi chạy)

- **Tập finding mục tiêu T**, khai báo từ literature chứ không từ số test/val của repo: Pleural Effusion, Atelectasis, Support Devices, Consolidation, Cardiomegaly. [TUỲ CHỈNH: hoặc dùng đúng 5 finding của F1-5]
- **Chỉ số chính:** trung bình trên T của AP one-vs-rest cho lớp Positive (mô hình ba lớp), tính trên val PL (n = 797) theo cross-fit.
  - Nhãn: Positive so với Negative ∪ Uncertain, chỉ tính các ô có nhãn. Điểm: `p_pos`.
  - CI: bootstrap ghép cặp theo study, 2.000 lần, seed 16.
- **QUA, claim "lateral giúp":** cần đủ ba điều kiện:
  - D1 có cận dưới CI 95% > 0, VÀ trung bình 3 seed > NB/√3;
  - rào chắn: với trung bình AP_pos của 9 finding ngoài T, Δ ≥ −NB/√3;
  - `views_1` = OFF (tự động đúng theo cấu trúc).
- **QUA, claim "đặc thù lateral":** thêm điều kiện D2 có cận dưới CI > 0.
- **Chỉ qua D1 mà D2 không qua:** kết luận "PARTIAL: lợi ích chủ yếu đến từ việc ghép thêm một mô hình". Phải viết đúng như vậy.
- **NB:**
  - σ là SD giữa các seed của chỉ số ở arm đó.
  - `NB = 2 × sqrt(σ_A² + σ_B²)` cho so sánh một seed; trung bình k seed thì dùng NB/√k.
  - OFF cố định nên `σ_OFF = 0`.
  - Ghi rõ NB được ước từ 3 seed.
- **Chỉ số phụ (báo cáo đủ, không dùng để quyết định):**
  - AUROC one-vs-rest (Pos, Unc);
  - wF1 và F1-5 theo cutpoint, macro recall, trên PL và trên toàn val;
  - AP present, để nối với encoder probe;
  - từng finding cho cả 14 finding, hiệu chỉnh Holm cho T;
  - `a[f]` theo từng finding;
  - arm hoán vị (kỳ vọng ≈ 0).
- **Test** chỉ đọc ở Bước 3, theo đúng quy tắc này.

## 8. GIỮ NGUYÊN — không được tự đổi

- OFF cùng các npz và cutpoint của nó; mô hình đang báo cáo `run_20261005_paper` và số liệu của nó.
- Ba lớp (D-023), `blank_label_policy: negative`, quy ước cutpoint, quy ước bootstrap (seed 16).
- Stage 2 và mọi cue rule. Prompt này không đụng tới.
- Không dùng MPC hay KL view-consistency trong nhánh mới.
- Không thêm dependency mới vào venv Stage 1.

## 9. Bẫy đã biết

- Phase block ghi đè `--options` (xem mục "Trap" trong `PLAN-2026-10-24`).
- `run.phases.phase1c.run.test_splits` mặc định là `[test]`. Khi chấm val phải override, nếu không sẽ lỡ đọc test.
- `sample_keys` đi theo ảnh anchor, nên phải re-key lượt chạy lateral.
- OFF npz không có `num_views`.
- Ổ `/mnt/drive1tb` là `ntfs3` và từng bị đánh dấu dirty: chỉ đọc, ghi cache lên `/home`.
- MedCLIP có thể đã thấy ảnh val lúc pretrain. Ảnh hưởng này như nhau cho mọi arm; ghi vào phần giới hạn.
- Không dùng AUROC đơn seed làm tiêu chí, vì NB của AUROC lớn.

## 10. Kiểm chứng bắt buộc trước khi báo xong mỗi bước

- **Test CPU:**
  - đo baseline CPU test suite TRƯỚC khi sửa code (`CUDA_VISIBLE_DEVICES=""`);
  - sau khi sửa, báo số pass/fail so với baseline đó;
  - mọi lỗi mới phải được giải thích.
- **GPU:** chạy smoke nhỏ trước lượt chính.
- **Dữ liệu:** mọi guard đều qua; số study của các nhóm khớp; key khớp 100%.
- **Tài liệu**, cập nhật trong cùng commit:
  - `CLAUDE.md` và `README.md`;
  - `.claude/skills/multi-view-novelty/SKILL.md`, mục "Where things stand";
  - `docs/KET_QUA_MULTI_VIEW.md`, thêm một mục mới;
  - trang `struct/` cho mỗi script và module mới.
- **Commit và push** chỉ sau khi user duyệt.

## 11. Chi phí ước tính (dùng khi xin user duyệt)

| Bước | GPU | Ghi chú |
|---|---|---|
| 0 | khoảng 1–2 giờ | đặc trưng lateral cho 3 encoder, probe L/C với 3 seed |
| 1 | khoảng 5–7 giờ | cache khoảng 30 phút; `L_frozen`/`L_LoRA`/`C_LoRA`, mỗi arm 3 seed, trên Swin 224. RAD-DINO 518 đắt hơn nhiều |
| 2a | 0 | CPU vài phút |
| 2b | khoảng 2–4 giờ mỗi seed | tuỳ chọn |
| 3 | khoảng 15–30 phút | suy luận trên test PL |

## 12. Định dạng báo cáo về chat

Báo bằng bảng tổng hợp, không có dữ liệu bệnh nhân:
- theo encoder × bước: L so với C (AP_pos trên T);
- D1, D2, D3: trung bình, CI, NB và kết luận theo quy tắc;
- từng finding cho cả 14 finding;
- `a[f]` theo từng finding;
- arm hoán vị;
- chi phí thực tế.

## 13. Câu hỏi phải hỏi user trước khi bắt đầu

1. Chốt tập T (mặc định ở mục 7, hay 5 finding của F1-5)?
2. Chốt chỉ số chính: AP_pos one-vs-rest, hay AP present như encoder probe?
3. Bước 0 chạy cả RAD-DINO và MedSigLIP, hay chỉ Swin?
4. Budget cho Bước 1: số seed, rank LoRA, và có thử RAD-DINO 518 không?
5. Có làm 2b (R2) không, hay dừng ở R1?

## Quyết định của user (2026-11-04, chốt TRƯỚC mọi kết quả)

User trả lời: "1 B 2 Đồng ý 3 tôi muốn đủ bộ 3 4 các bước sau làm theo bạn đề xuất".

1. **T = 5 finding của F1-5:** Atelectasis, Cardiomegaly, Consolidation, Edema, Pleural Effusion.
   Rào chắn = trung bình AP_pos của 9 finding còn lại (gồm Support Devices, No Finding).
2. **Chỉ số chính: AP one-vs-rest của lớp Positive** (mục 7 giữ nguyên). AP present chỉ để mô tả.
3. **Bước 0 chạy đủ ba encoder:** Swin (MedCLIP), RAD-DINO, MedSigLIP-448.
4. Theo đề xuất của planner (được user chấp nhận cho câu 4 trở đi):
   - Bước 1 (LoRA): chốt sau G0; nếu làm thì mặc định Swin 224, 3 seed, r=8 / alpha=16, chưa thử RAD-DINO 518.
   - Bước 2b (R2): không làm; dừng ở R1, hỏi lại chỉ khi 2a qua.
   - Reading rule mục 7 và cổng G0 mục 3 giữ nguyên như prompt, với T ở trên. Ghi rõ: NB ước từ
     3 seed của chuyên gia, σ_OFF = 0, nên NB chỉ phản ánh nhiễu seed của chuyên gia.
     λ = 1e-2 và a ∈ [0, 3] cố định trước, không tune trên val.
   - Planner được làm Bước 0.1 trên CPU (không GPU).
5. Quy ước bổ sung do planner cố định trước khi có số (để không phải chọn sau):
   - Chuyên gia chọn epoch theo macro AP "có bệnh" trên holdout 10% của train, đúng như encoder probe
     (không theo chỉ số chính, để không tune theo T).
   - CI của hiệu trung bình k seed: mỗi lần bootstrap tính hiệu của từng seed trên cùng resample rồi lấy
     trung bình theo seed; percentile 2,5 / 97,5.
   - Seed s của L ghép với seed s của C trong arm OFF+C+L.
   - Arm hoán vị: hàng của L xáo trong cùng fold, `default_rng(16)`.

## Execution report — Bước 0.1 (code CPU), 2026-11-04, planner checkout

- commit lúc bắt đầu: `9c33aa5`; thay đổi chưa commit.
- Baseline CPU suite TRƯỚC khi sửa (`CUDA_VISIBLE_DEVICES=""`, `/home/phuong/venv`, bỏ qua
  `test_blip2_negative_sampling.py` và `test_encoder_ablation.py`): **1301 passed, 21 failed, 52 skipped**.
- Sau khi sửa: **1315 passed, 21 failed, 52 skipped**. +14 = đúng 14 test mới; 21 lỗi trùng hệt danh sách
  baseline (thiếu torchvision / env config trên máy CPU, không liên quan).
- `ruff check` sạch trên mọi file đã sửa hoặc thêm.
- File mới / sửa:
  - `training/evaluation/view_keys.py` (mới): map study → key frontal-anchor / key lateral / nhóm PL-F1-other /
    num_views, dựng từ `dataset.annotation` theo đúng quy tắc key của eval hook (digest vị trí dòng anchor).
  - `scripts/encoder_probe_features.py`: thêm `--views lateral` (opt-in; mặc định không đổi). Chỉ giữ study PL,
    `keys.npy` = key frontal-anchor, `image_keys.npy` = key ảnh lateral, `view_map.npz` cho cả split;
    ba kiểm tra cứng (thứ tự study, anchor lateral = lateral của PL, key ảnh từng batch).
  - `scripts/lateral_expert_fit.py` (mới): L / C một encoder, từng seed, cổng G0 nhãn.
  - `scripts/residual_stack.py` (mới): R1 cross-fit, các arm, bootstrap, NB, verdict, bảng toàn val.
  - `tests/test_residual_lateral.py` (mới, 14 test): re-key trên bảng giả PA+L / PA / AP+LL / chỉ L /
    PA+AP+L / L đứng trước PA; R1 a=b=0 trùng OFF từng bit; fold tất định và đi theo key; λ lớn → OFF;
    chuyên gia nhiễu và arm hoán vị có a ≈ 0; chuyên gia có tín hiệu có a > 0,5; `main` với số nhóm sai thì dừng.
  - `struct/`: trang mới cho 3 file mới, cập nhật `encoder_probe_features.py.doc.md` và hai `_index.md`.
- Bẫy gặp khi viết: `fit_severity_cutpoints` coi **-1** là thiếu nhãn, còn đặc trưng probe ghi **-100**.
  `residual_stack.py` chuẩn hoá mọi nhãn âm về -1. Lưới cutpoint 121 điểm tốn O(lưới²): ước vài phút CPU
  cho mỗi encoder trên val PL.
- Chưa làm: chưa chạy GPU, chưa commit/push, chưa cập nhật CLAUDE.md / README / skill / KET_QUA
  (sẽ cập nhật cùng commit, khi có kết quả thì thêm số).

## Bước 0.2 — lệnh dự kiến (CHƯA DUYỆT)

Host cần code này: commit + push từ planner rồi `git pull` trên host (cần user duyệt commit/push).

```bash
cd ~/Meta-CXR && git pull --ff-only && git rev-parse --short HEAD     # ghi sha
PY=~/.venvs/meta-cxr-stage1-311/bin/python
RADDINO=<thư mục microsoft/rad-dino cục bộ mà encoder probe 2026-10-31 đã dùng>
MSIG=google/medsiglip-448
OUT=~/reslat_20261104; test ! -e $OUT || exit 1; mkdir -p $OUT
# guard trước mỗi lượt GPU: findmnt -no FSTYPE /mnt/drive1tb == ntfs3; nvidia-smi rảnh;
#   không có pretraining.train / run_medgemma_qlora (ps -eo pid,args, khớp theo vị trí);
#   df -h /home còn chỗ (đặc trưng lateral train ~22k study ước vài GB).
# 0) smoke: val, 256 study ngẫu nhiên, lọc PL (~110)
$PY scripts/encoder_probe_features.py --split val --limit 256 --views lateral     --raddino $RADDINO --medsiglip $MSIG --out-dir $OUT/smoke_val_lat > $OUT/smoke.log 2>&1
#    kỳ vọng: in "groups {'PL': 797, 'F1': 930, 'other': 81}"; nhãn khớp encprobe val trên các key đó.
# 1) chính
$PY scripts/encoder_probe_features.py --split val --views lateral     --raddino $RADDINO --medsiglip $MSIG --out-dir $OUT/val_lat > $OUT/val_lat.log 2>&1      # kỳ vọng 797 PL
$PY scripts/encoder_probe_features.py --split train --limit 40000 --seed 16 --views lateral     --raddino $RADDINO --medsiglip $MSIG --out-dir $OUT/train_lat > $OUT/train_lat.log 2>&1  # ~22.000 PL
for E in swin raddino medsiglip; do for V in lateral frontal; do
  $PY scripts/lateral_expert_fit.py --encoder $E --view $V     --lateral-train $OUT/train_lat --lateral-val $OUT/val_lat     --frontal-train ~/encprobe_20261031/train --frontal-val ~/encprobe_20261031/val     --out-dir $OUT/experts >> $OUT/experts.log 2>&1
done; done
# 2) CPU
B=~/run_20261024_nomv/phase1c/mimic_cxr_full_blip2/result
for E in swin raddino medsiglip; do
  $PY scripts/residual_stack.py --off $B/val_predictions_epoch_best.npz --view-map $OUT/val_lat/view_map.npz     --off-cutpoints ~/nomv_probe_20261024/cutpoints_nomv.json     --expert L=$OUT/experts/${E}_lateral_s16.npz,$OUT/experts/${E}_lateral_s17.npz,$OUT/experts/${E}_lateral_s18.npz     --expert C=$OUT/experts/${E}_frontal_s16.npz,$OUT/experts/${E}_frontal_s17.npz,$OUT/experts/${E}_frontal_s18.npz     --name $E --out $OUT/residual_$E.json > $OUT/residual_$E.log 2>&1
done
```

Dừng nếu: số nhóm khác 797 / 930 / 81; bất kỳ key nào không khớp; cổng G0 nhãn hỏng; tên file npz của OFF
khác (kiểm bằng `ls`); `ls ~/encprobe_20261031/{train,val}` thiếu `raddino_*` / `medsiglip_*`.

## Execution report — Bước 0.2, 2026-11-04 (DỪNG trước GPU)

- User duyệt commit/push và lượt GPU 0.2. Đã push `a5a3c0e`; host `~/Meta-CXR` `git pull --ff-only` -> `a5a3c0e`.
  Test trên host (CPU): `tests/test_residual_lateral.py` + `tests/test_encoder_probe.py` 24 passed.
- Guard lúc 10:09: `/mnt/drive1tb` ntfs3, GPU 79 MiB / 0 %, không có job python nào, `/home` còn 75 GB.
- **Điều kiện dừng đã ghi sẵn bị kích hoạt:** `~/encprobe_20261031/` (đặc trưng frontal của 10-31) và
  `~/Meta-CXR-encprobe` không còn trên host, chỉ còn log `~/encprobe_*.{sh,out,log}`. Chưa chạy lệnh GPU nào.
  RAD-DINO cục bộ: `~/models/rad-dino` (đúng thư mục `encprobe_chain.sh` đã dùng).
- Đề xuất (chờ user): trích lại đặc trưng frontal bằng chế độ mặc định không đổi của
  `encoder_probe_features.py` (val toàn bộ, train `--limit 40000 --seed 16` = cùng 40.000 study của 10-31),
  ra `$OUT/val_front`, `$OUT/train_front`; kiểm lại bằng số study có nhãn 1786/1808 và 39615/40000 như log 10-31.
  Thêm khoảng 35 phút GPU.
- **User duyệt 2026-11-04: trích lại frontal rồi chạy tiếp.** Chuỗi `~/reslat_chain.sh` (bản sao trong log) khởi chạy
  MỘT lần lúc 10:12:35 từ `~/Meta-CXR` @ `a5a3c0e`, xong 11:19:37 (`RESLAT_DONE`), không bước nào abort.
  Log: `~/reslat_20261104.log`; đầu ra `~/reslat_20261104/` (chỉ trên host).

## Execution report — Bước 0.2 (GPU) + chấm R1 trên val PL, 2026-11-04

Kiểm tra (đều qua): smoke val 256 -> nhóm 797 / 930 / 81, 113 PL; frontal val 1786/1808 và train 39615/40000 có nhãn
(= log 10-31); val lateral 797 PL (779 có nhãn); train 40.000 study -> 18.553 PL (18.252 có nhãn; cả train có
103.062 PL); G0 nhãn lateral = nhãn frontal ở cả 6 lượt fit; mọi key của OFF tìm thấy; `views_1` = OFF từng bit.
Thời gian: frontal train 33 phút, lateral train 15 phút, 18 lượt fit chuyên gia ~3 phút, R1 ~4 phút / encoder.

Chuyên gia, val PL n=797, macro AP present (mô tả; trung bình 3 seed): Swin L 0.3754 / C 0.4123;
RAD-DINO L 0.3295 / C 0.3771; MedSigLIP L 0.3191 / C 0.3779.

Chỉ số chính AP_pos trung bình trên T (F1-5), val PL, cross-fit, trung bình 3 seed. OFF = 0.4573.

| encoder | OFF+C | OFF+L | OFF+C+L | OFF+Lperm | L_alone | C_alone |
|---|---:|---:|---:|---:|---:|---:|
| Swin | 0.4607 | 0.4741 | 0.4736 | 0.4465 | 0.4406 | 0.4630 |
| RAD-DINO | 0.4718 | 0.4706 | 0.4718 | 0.4504 | 0.4126 | 0.4525 |
| MedSigLIP | 0.4638 | 0.4661 | 0.4666 | 0.4474 | 0.3715 | 0.4399 |

| encoder | D1 = (OFF+L)−OFF | D2 = (OFF+C+L)−(OFF+C) | D3 = (OFF+C)−OFF | L−C alone | rào chắn nonT (D1) | verdict |
|---|---|---|---|---|---|---|
| Swin | **+0.0168 [+0.0022, +0.0340]** | +0.0129 [−0.0002, +0.0279] | +0.0034 [−0.0039, +0.0122] | −0.0224 | +0.0059 | **PARTIAL** |
| RAD-DINO | +0.0133 [−0.0016, +0.0300] | +0.0000 [−0.0102, +0.0108] | +0.0145 [+0.0034, +0.0264] | −0.0399 | +0.0110 | KHÔNG QUA |
| MedSigLIP | +0.0088 [−0.0083, +0.0252] | +0.0028 [−0.0100, +0.0142] | +0.0065 [−0.0040, +0.0176] | −0.0684 | +0.0055 | KHÔNG QUA |

NB/√3 (D1): Swin 0.0003, RAD-DINO 0.0014, MedSigLIP 0.0028 -- chỉ là nhiễu seed của head chuyên gia (encoder đóng
băng, σ_OFF = 0), nên rất nhỏ; CI bootstrap mới là điều kiện ràng buộc. Dperm âm ở cả ba (−0.0108 / −0.0069 / −0.0099),
như kỳ vọng.

Đọc theo quy tắc mục 7 (chốt trước):
- **Swin: "PARTIAL: lợi ích chủ yếu đến từ việc ghép thêm một mô hình".** D1 qua (CI dưới > 0, > NB/√3, rào chắn qua);
  D2 không qua -- cận dưới −0.0002 nằm sát 0, nên "đặc thù lateral" là CHƯA XÁC LẬP, không phải bác bỏ.
- RAD-DINO, MedSigLIP: không qua.
- Từng finding (Swin, D1, Holm trên T): không finding nào qua; Cardiomegaly +0.051, Edema +0.034, Consolidation +0.010
  (Holm p 0.09), Atelectasis −0.001, **Pleural Effusion −0.009** -- lateral qua probe đóng băng không giúp PE.
- Phụ, mô tả: trên PL AUROC Pos của Swin D1 +0.0074; cutpoint wF1 / F1-5 trên PL phẳng (−0.0004 / +0.0014).
  ⚠ Trên TOÀN val, auroc_pos_mean của MỌI arm residual thấp hơn OFF ~0.011 (0.803 -> 0.792), kể cả OFF+C: R1 chỉ đổi
  xác suất của study PL, nên thang điểm PL và non-PL lệch nhau khi xếp hạng gộp (giả thuyết, chưa đo riêng). wF1 / F1-5
  toàn val phẳng (Swin OFF+L 0.8578 / 0.8073 vs OFF 0.8579 / 0.8064).

Cổng G0: không encoder nào qua cả D1 và D2 -> không phải GO-thẳng; L kém C hơn 0.02 ở cả ba encoder
(−0.022 / −0.040 / −0.068) -> **CẦN-BƯỚC-1** (đọc "D1 và D2 không qua" là "không qua cả hai", vì Swin qua D1); không
phải NO-GO (L không ≈ C). Encoder cho Bước 1 theo quy tắc: D2 lớn nhất = **Swin**. Bước 1 CHƯA chạy, chờ user duyệt.
Test chưa đọc.

## Bước 1 — code CPU (2026-11-04, user duyệt "1 đồng ý làm Bước 1")

File mới: `vision_encoders/lora.py`, `scripts/lateral_lora_cache.py`, `scripts/lateral_lora_fit.py`,
`tests/test_lateral_lora.py` (14 test), struct docs. Không đổi file nào đang có.

Quy ước cố định TRƯỚC mọi số của Bước 1 (theo mục 4, chỗ để ngỏ do planner chọn):
- Chỉ Swin (MedCLIP, 224), như user chốt; không làm biến thể "adapter" (chỉ LayerNorm) ở mục 1.2.
- LoRA: r=8, alpha=16, dropout 0,05, query + value của mọi block ở stage 3–4 (16 lớp). LR LoRA **2e-4**
  (cận trên của khoảng 1e-4–2e-4 trong mục 1.4), LR head 3e-4, wd 0,05 cho cả hai, batch 64, bf16.
- Tối đa 8 epoch, warmup-cosine (5%) trên 8 epoch; dừng sớm khi 2 epoch liền không vượt holdout tốt nhất;
  chọn epoch theo macro AP "có bệnh" trên holdout 10% (như Bước 0, không theo chỉ số chính).
- Dữ liệu: mọi study PL của train (trên host đếm được 103.062, kế hoạch ước 120.000), cả hai view từ CÙNG
  một lượt dataset (anchor frontal + aux lateral). Augmentation chỉ khi train, không lật (của `swin_finetune_fit`).
- Seed s ∈ {16, 17, 18}; arm `L_frozen` cũng chạy augmentation, nên khác `L` của Bước 0 ở dữ liệu
  (103k so với 18,5k study) và augmentation — đó là lý do arm này tồn tại.
- G1(a) = `L_lora − L_frozen`, AP_pos(T) val PL, bootstrap 2.000 seed 16, qua khi cận dưới > 0 VÀ > NB/√3.
  G1(b) = `residual_stack.py` với `L = L_lora`, `C = C_lora`, quy tắc mục 7 không đổi (D2 cận dưới > 0).
  Mô tả thêm (không quyết định): `residual_stack.py` với `L = L_frozen`, `C = C_lora`.
- Đọc G1 đúng như mục "Cổng G1"; test vẫn không đọc.

### Lệnh dự kiến (CHƯA DUYỆT GPU)

```bash
cd ~/Meta-CXR && git pull --ff-only    # sha ghi vào log
PY=~/.venvs/meta-cxr-stage1-311/bin/python; R=~/reslat_20261104; OUT=~/reslat_lora_20261104
test ! -e $OUT || exit 1               # + guard: ntfs3, GPU rảnh, không job train, /home >= 40 GB
$PY scripts/lateral_lora_cache.py --split val   --out-dir $OUT/cache/val     # kỳ vọng 797 PL
$PY scripts/lateral_lora_cache.py --split train --out-dir $OUT/cache/train   # kỳ vọng 103062 PL
# smoke: 1 epoch, đo phút/epoch và max mem, G0 + G2 phải qua
$PY scripts/lateral_lora_fit.py fit --arm L_lora --seed 16 --epochs 1 --cache-dir $OUT/cache --ref-dir $R --out-dir $OUT/smoke
for A in L_frozen L_lora C_lora; do for S in 16 17 18; do
  $PY scripts/lateral_lora_fit.py fit --arm $A --seed $S --cache-dir $OUT/cache --ref-dir $R --out-dir $OUT/experts
done; done
B=~/run_20261024_nomv/phase1c/mimic_cxr_full_blip2/result/val_predictions_epoch_best.npz
e() { echo $OUT/experts/$1_s16.npz,$OUT/experts/$1_s17.npz,$OUT/experts/$1_s18.npz; }
$PY scripts/lateral_lora_fit.py compare --base L_frozen=$(e L_frozen) --new L_lora=$(e L_lora) --names-from $B --out $OUT/G1a.json
$PY scripts/residual_stack.py --off $B --view-map $R/val_lat/view_map.npz --off-cutpoints ~/nomv_probe_20261024/cutpoints_nomv.json \
  --expert L=$(e L_lora) --expert C=$(e C_lora) --name lora --out $OUT/residual_lora.json
$PY scripts/residual_stack.py ... --expert L=$(e L_frozen) --expert C=$(e C_lora) --name frozenL --out $OUT/residual_frozenL.json
```

Dừng nếu: số PL khác 797 (val); bất kỳ kiểm tra của cache; G0 hoặc G2 hỏng; smoke vượt 14 GB hoặc ước tổng
quá 12 giờ GPU (báo user trước khi chạy tiếp).

## Execution report — Bước 1 (GPU, 2026-11-04, val PL n=797; test KHÔNG đọc)

Chuỗi `~/reslat_lora_20261104/` trên host, chạy một lần từ commit `8b3ecd7`, kết thúc 15:11 (`RESLAT_LORA_DONE`,
không ABORT). Cache: val 797 PL, train 103.062 PL; G0 qua; smoke qua. 9 fit (3 arm x 3 seed), mỗi fit
7–15 phút, max mem 3.636 MiB; G2 qua ở cả 9 (backbone bit-identical; LoRA đổi 0,06–0,14 tương đối ở L_lora/C_lora,
đúng 0 ở L_frozen).

**G1(a) — KHÔNG QUA.** `L_lora − L_frozen`, AP_pos(T) val PL: **−0,0011 [−0,0193, +0,0154]**, NB/√3 0,0041.
Theo seed: L_frozen 0,4621 / 0,4587 / 0,4590; L_lora 0,4621 / 0,4580 / 0,4564. LoRA không làm encoder đọc
lateral tốt hơn trên T. (Mô tả, không quyết định: macro AP "có bệnh" trên cả 14 bệnh tăng 0,384–0,387 →
0,395–0,400; chỉ số quyết định không tăng.)

**G1(b) (đọc theo quy tắc, nhưng (a) đã không qua):** `residual_stack.py`, L = L_lora, C = C_lora:

| hiệu | AP_pos(T) | NB/√k |
|---|---|---|
| D1 = (OFF+L) − OFF | +0,0186 [+0,0028, +0,0363] | 0,0019 |
| Dperm (lateral hoán vị) | −0,0121 [−0,0315, +0,0084] | 0,0014 |
| D2 = (OFF+C+L) − (OFF+C) | +0,0141 [**+0,0002**, +0,0289] | 0,0037 |
| D3 = (OFF+C) − OFF | +0,0028 [−0,0048, +0,0117] | 0,0002 |
| L_alone − C_alone | +0,0058 [−0,0363, +0,0535] | 0,0051 |

Script in "QUA" cho (b). Cận dưới của D2 là +0,0002, nằm sát 0 (một bound cách 0 dưới 0,0005 có thể đổi dấu
theo nhiễu Monte-Carlo của bootstrap).

Mô tả (L = L_frozen 103k + augmentation, C = C_lora): D1 +0,0154 [+0,0017, +0,0350], D2 +0,0122
[−0,0011, +0,0297] → PARTIAL, giống Bước 0 Swin (D2 +0,0129 [−0,0002, +0,0279]).

**Kết luận theo cổng G1 đã chốt trước:** "Không qua (a): dừng nhánh LoRA." Phần D2 của (b) với LoRA
(+0,0141, cận dưới +0,0002) không đổi kết luận đó: LoRA không cải thiện chuyên gia lateral (G1a ≈ 0), và
D2 với L_frozen không qua, nên khác biệt nhỏ giữa hai lượt đọc D2 không đến từ lateral tốt hơn. Bằng chứng
"đặc thù lateral" vẫn KHÔNG ĐƯỢC XÁC LẬP; lợi ích D1 (~+0,015–0,019) vẫn chủ yếu là ghép thêm một mô hình.
Không chạy Bước 2 hay đọc test khi chưa có quyết định mới của user.

## Quyết định của user sau Bước 1 (2026-11-04) và kế hoạch B (CHƯA chạy)

**Quyết định (user, 2026-11-04):** chọn "Residual + Swin LoRA (103k study PL)" làm **hướng phát triển chính**, và
chọn phương án **B**: đưa residual vào Stage 1 và train trên tập train. Đây là quyết định của user, **đi ngược cổng G1
đã chốt trước** ("Không qua (a): dừng nhánh LoRA"). Các số ở trên không đổi và phải nêu cạnh mọi claim:
- G1(a) LoRA so với frozen: −0,0011 [−0,0193, +0,0154], tức LoRA không cải thiện chuyên gia lateral trên T;
- D1 có thật (+0,0186 [+0,0028, +0,0363]), nhưng Swin đóng băng cho mức gần bằng (+0,0154 / +0,0168);
- D2 có cận dưới +0,0002, sát 0, nên "đặc thù lateral" CHƯA được xác lập; claim hiện tại chỉ là "PARTIAL: lợi ích
  chủ yếu đến từ việc ghép thêm một mô hình";
- ở Bước 0, AUROC Pos gộp trên toàn val giảm khoảng 0,011 (giả thuyết: thang điểm PL và non-PL lệch nhau).
YAML không đổi. Test vẫn chưa đọc. 103.062 = toàn bộ study PL của train, nên L_lora đã dùng hết dữ liệu lateral có
được; phần "toàn bộ dữ liệu" ở B là train cái residual (Δ) trên tập train thay vì fit R1 trên 797 study val.

**Trở ngại đã biết:** `lateral_lora_fit.py` chỉ lưu xác suất trên val, KHÔNG lưu trọng số LoRA và head. Muốn có đầu
vào trên tập train thì phải train lại L (và C), lần này có lưu trọng số. Logit của OFF và của L trên chính dữ liệu đã
train chúng là in-sample (quá tự tin), nên Δ học trên đó dễ tin OFF/L quá mức.

**Các bước (mỗi bước cần user duyệt riêng; chưa bước nào được duyệt chạy):**
- **B0 — đo OFF in-sample (GPU ~1–1,5 h, không sửa code).** Chạy evaluate-only của OFF
  (`run_20261024_nomv` phase1c `checkpoint_best`) trên split train, ghi npz vào thư mục mới trên `/home`. So CE và ECE
  của OFF trên train PL với val PL. Quy tắc (mục 2b, chốt trước): nếu `CE_train < 0,9 × CE_val` thì OFF quá tự tin;
  khi đó hỏi user chọn (i) giữ R1, (ii) cross-fit OFF 2 fold (~10 h), hoặc (iii) hiệu chỉnh nhiệt độ.
- **B1 — chuyên gia cross-fit trên train (sửa code nhỏ + GPU ~3 h).** Thêm vào `lateral_lora_fit.py`: lưu trọng số
  LoRA và head, và tuỳ chọn cross-fit 2 fold trên train PL (mỗi fold train trên nửa kia), để có xác suất/token
  out-of-fold cho cả 103.062 study PL train. Arm L_lora và C_lora, 3 seed.
- **B2 — residual R2 trên đặc trưng đã cache (code mới + GPU vài giờ).** `z = z_OFF.detach() + Δ`, Δ là head nhỏ
  (14 query, 1–2 lớp cross-attention, d=256) đọc token lateral của L và logit OFF, Linear cuối khởi tạo 0, chỉ áp cho
  study PL. Train trên train PL, chọn epoch theo holdout của train, đọc trên val PL theo đúng mục 7 (AP_pos(T),
  D1 = R2(OFF+L) − OFF, D2 = R2(OFF+C+L) − R2(OFF+C), NB từ 3 seed), và so R2 với R1 trên val.
- **B3 — tích hợp `model.view_fusion.mode: residual` / phase `phase1r`** chỉ khi R2 thắng R1 trên val, kèm các test CPU
  ở mục 2b. Test đọc một lần ở Bước 3, sau khi user duyệt ứng viên cuối.
