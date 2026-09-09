# META-CXR

Repository nghiên cứu cho bài toán hiểu ảnh X-quang ngực và sinh báo cáo. Stage 1 học biểu diễn thị giác theo study, hợp nhất nhiều view, tạo Q-Former tokens và dự đoán bất thường. Stage 2 dùng MedGemma để sinh nội dung báo cáo, với đường ảnh native hoặc Q-Former soft tokens. Repository cũng có evaluator cho classification và report generation.

> **Trạng thái hiện tại:** Stage-1 đã chạy đủ trên GPU. Explanation loss đã được
> A/B có kiểm soát 5 epoch (bật/tắt) và evaluator XAI đã chạy trên cả hai
> checkpoint — kết quả: **không có tác dụng đo được, và CAM ở mức ngẫu nhiên ở cả
> hai nhánh**, nên hướng này đã **tắt trong production từ 2026-08-17** (chi tiết ở
> [Explanation-aware learning](#explanation-aware-learning-đã-tắt-trong-production-từ-2026-08-17)).
> Table 5 Stage-1 inference ablation hoàn tất. Stage 2 đã train trên GPU và có
> so sánh NLG Arm A/C trên cùng 2.772 ca test: Arm C hiện kém hơn Arm A.
> Chưa có metric lâm sàng; lịch training khác nhau nên chưa thể quy toàn bộ
> chênh lệch cho kiến trúc. Xem phân tích Stage 2 và sửa contract cues bên dưới.
>
> **Máy train:** một host duy nhất, `phuong@100.116.167.90` (`minhphuong`, máy cá
> nhân của tác giả), **1× RTX 5060 Ti 16 GB**. Không còn đường chạy cloud: các
> recipe GCP/L4/Kaggle/2×3090 đã bị gỡ ngày 2026-08-13 để tối ưu chi phí.
>
> ⚠ **MÁY ĐÃ ĐƯỢC CÀI LẠI HĐH NGÀY 2026-08-17** (Ubuntu 26.04 LTS, kiểm tra qua
> SSH 2026-08-18). Ranh giới thiệt hại là **theo ổ đĩa**: `/home` bị xoá sạch,
> còn **ổ dữ liệu 1 TB không hề bị đụng** — installer đi vào ổ khác.
>
> **Còn nguyên** (trên `/mnt/drive1tb`): dataset `mimic-cxr-jpg-full` + manifest
> `processed/full_allviews_v2`, checkpoint của `run_b16fast_20260814` (có
> `checkpoint_best`) và `run_gate3_20260815`, cache `datasets/explanation_masks`,
> nguồn CheXmask/MS-CXR, kết quả Table 5 và `private-results`. Tổng 9 file
> `.pth`, 11,5 GB.
>
> **Mất** (nằm trên `/home`): hai run A/B `abl_on`/`abl_off` của explanation loss
> — nên **kết quả D-017 không chấm lại hay mở rộng được**, dù con số vẫn còn
> trong git; và `explanation_masks_v2` (cache có `masks_bbox_*`), nên **term
> strong mất toàn bộ supervision** cho tới khi build lại cache.
>
> ⚠ Ổ dữ liệu **đổi device**: `/dev/nvme1n1p2` → **`/dev/nvme0n1p2`**. Đã mount
> lại ngày 2026-08-18 bằng `ntfs3 -o ro` (đúng cấu hình khuyến nghị).
> Chi tiết: mục "The training host" trong `CLAUDE.md`.

## Trạng thái hiện tại

| Thành phần | Trạng thái |
|---|---|
| Branch integration | Các nhánh tính năng đã được tích hợp tuyến tính vào `main`; xem [integration audit](docs/final_branch_integration_audit.md) |
| Stage 1 implementation | Study-level/multi-view, Q-Former, MHCAC có trong code và đã chạy full trên GPU. Explanation loss **tắt** (lambda 0/0) sau A/B 2026-08-17; khối vision-language ITC/ITM/LM **tắt** (lambda 0/0/0) từ 2026-08-19 sau gate check — giống repo gốc |
| Stage 2 implementation | MedGemma QLoRA. ✅ **Đã chạy trên GPU**: arm A (`medgemma_direct`) dừng ở **0,86 epoch** — không có validation, `best_val_loss` = `+inf`, adapter promote bằng tay. arm C (`meta_cxr_native_qformer_guided`) **xong trọn 1 epoch 2026-09-07**, 82h28m, `val_loss` **1,0019** — run Stage-2 đầu tiên thực sự có chọn mô hình |
| Explanation masks | Full cache đã build và kiểm chứng (`explanation_masks_v2`, có `masks_bbox_*`). Không còn được training tiêu thụ; giữ cho evaluator |
| XAI evaluation | Đã chạy trên GPU với 2 checkpoint (test split). **Cảnh báo: saliency precision ở mức ngẫu nhiên** — luôn kèm baseline diện tích mask |
| CPU tests | Xem mục [Testing](#testing) cho output chạy thật của Phase 3 |
| GPU evidence | Stage-1 full run xong; A/B explanation loss bật/tắt 5 epoch xong (2026-08-16/17) kèm calibration + eval test + XAI cả hai nhánh |
| Checkpoint cũ | **Đã xoá toàn bộ 2026-08-14** (15 file, 39 GB) — các run đó đi sai hướng và không nạp được vào recipe hiện tại (Swin tắt → 98 token thay vì 147). Số liệu Table 5 còn trong `results/` nhưng không tái lập được |
| Full MIMIC-CXR training | ✅ **Đã chạy xong 2026-08-20** — `run_20260819_xmpoff`, 10/10 epoch, `rc=0`, 12h35m, 0.3505 s/it, 0 kernel fault. Best epoch 6 |
| Reproduced metrics | ✅ Stage 1 đã có kết quả test split (3.269 study), ngưỡng calibrate trên val. Stage 2 có NLG Arm A/C trên cùng 2.772 ca test; Arm C kém hơn, nhưng lịch training khác nhau. Chưa có kết luận về độ đúng lâm sàng hay lợi ích so với zero-shot trên cùng cohort. |
| Máy train | Lỗi kernel fault liên tục từ 17/08 đã **hết** sau khi **tắt XMP** trong BIOS (4 thanh RAM 2 hãng chạy 3200 MT/s ngoài mức Intel validate). Chi phí: **+1.3%** tốc độ |

## Những thay đổi so với META-CXR gốc

Repository kế thừa công trình META-CXR nhưng code hiện tại đã bổ sung:

- sampling theo study và Stage 1 multi-view với anchor/auxiliary view;
- đường Stage 2 MedGemma native độc lập với Stage 1, bên cạnh Q-Former ablations;
- Prompt v2 có cấu hình, version và hash;
- evaluator cho classification, report generation, error analysis và counterfactual checks;
- explanation-aware Grad-CAM loss, cache CheXmask/MS-CXR và evaluator XAI;
- workflow preflight và config Stage 1 cho máy train một GPU.

Các thay đổi này chưa kèm bằng chứng rằng pipeline mới tốt hơn kết quả của bài báo gốc.

## Kiến trúc tổng quan

```text
Chest X-ray study
    -> Stage 1 visual encoders
    -> anchor/auxiliary multi-view fusion
    -> Q-Former representations + abnormality classification
       + optional Grad-CAM explanation loss (lung/bbox mask)
    -> Stage 2 MedGemma (native image hoặc Q-Former soft tokens)
    -> FINDINGS/report output
    -> classification, generation và XAI evaluators
```

Config Stage 1 production bật **BioViL-T và PubMedCLIP**; SwinV2 tắt từ
2026-08-14, RadDINO có implementation nhưng cũng đang tắt. MHCAC dự đoán 14 nhãn
theo Positive/Negative/Uncertain, còn Q-Former tạo 32 query tokens.

**Hai encoder giữ nguyên thang đo riêng (từ 2026-08-14).** Đây là lý do chạy hai
encoder — một cái nhìn kỹ, một cái nhìn tổng quát:

| Encoder | Đầu vào | Token vào MHCAC | Vai trò |
|---|---|---|---|
| BioViL-T | 448×448 | 196 (lưới 14×14, ô 32 px) | cục bộ, chi tiết |
| PubMedCLIP | 224×224 | 1 CLS + 49 (lưới 7×7, ô 64 px) | toàn cục + ngữ cảnh vùng |

Trước đó cả hai bị ép về lưới 7×7 và CLS của PubMedCLIP bị xoá, nên thực chất
model chỉ nhận hai bản đồ thô giống nhau và không có token toàn cục nào. Ngoài ra
49 patch của PubMedCLIP có cosine đôi một 0,674 (BioViL: 0,0017) vì một thành
phần DC cố định; nay chúng được đọc qua `post_layernorm` rồi trừ mean từng ảnh,
đưa về −0,014. Chi phí của 148 token thêm vào: **+0,6% s/it**. Chi tiết và số đo
đầy đủ ở `struct/project/_meta/DECISIONS.md` (D-016).

⚠ **Mọi checkpoint tạo trước 2026-08-14 không load được** với kiến trúc này.

## Stage 1

Stage 1 nhận mẫu theo study. Với `multi_view: true`, view ưu tiên PA/AP/lateral được chọn làm anchor và tối đa một view phụ được fuse trước projection. Nhánh student dùng ảnh để tạo abnormality predictions và Q-Former representations; report text chỉ tham gia teacher branch trong lúc train.

- Entrypoint: [`pretraining/train.py`](pretraining/train.py)
- Config production (một GPU, recipe duy nhất): [`pretraining/configs/mimic_cxr_full.yaml`](pretraining/configs/mimic_cxr_full.yaml)
- Checkpoint selection: **`loss`** (tổng val loss) trên validation; test được giữ ngoài quá trình chọn checkpoint.
  `macro_auprc` vẫn được log mỗi epoch được chấm để đối chiếu — val loss bị các nhãn phổ biến chi phối,
  nên một model bỏ hẳn nhãn hiếm có thể ăn điểm hơn model đôi khi tìm ra nó.

#### Khối vision-language (ITC/ITM/LM) ĐÃ TẮT từ 2026-08-19

`lambda_itc`, `lambda_itm`, `lambda_lm` đều là `0.0`, giống hệt repo gốc
(`DasithEdirisinghe/META-CXR`) — ở đó toàn bộ khối này bị comment out và loss
Stage-1 chỉ là `cls + 0.3*contrastive + 0.7*orth + 0.3*sparsity`.

Quyết định này dựa trên đo đạc, không phải chi phí. `scripts/check_itc_gate.py`
trên val, 256 cặp **hợp lệ**, chance rank 127.5:

| nhánh | nhiệt độ | rank i2t | rank t2i | `delta_nats` |
|---|---|---|---|---|
| chưa train | 0.07 (pin) | 130.68 | 130.30 | −0.0833 |
| 525 update, nhiệt độ học được | 0.00796 | 127.43 | 127.65 | −1.1168 |
| 500 update, nhiệt độ pin | 0.07 | 128.38 | 127.45 | **−0.0025** |

Mọi nhánh nằm đúng mức ngẫu nhiên; gate yêu cầu `delta ≥ +0.10`. Nhiệt độ học
được đã sụp từ 0.0249 xuống 0.00796, nhưng pin nó lại không thay đổi gì — nhiệt
độ chỉ là triệu chứng.

⚠ **Hệ quả: checkpoint Stage-1 hiện tại KHÔNG dùng được cho các mode Stage-2
`meta_cxr_qformer*`**, vì cross-attention sinh soft token không hề thấy ảnh y
khoa nào trong Stage 1. Repo gốc cũng ở đúng tình trạng này. Text tower của
Q-Former thì vẫn được train, qua `lambda_teacher_cls`/`lambda_distill`.
`medgemma_direct` không bị ảnh hưởng.

Batch trở lại **16 × accum 4** (effective 64 không đổi) vì batch 8 chỉ tồn tại để
nhét vừa khối VL. Swin vẫn tắt — tắt Swin khôi phục `_native_stream_layouts`,
cấu hình đã đo là tốt hơn.

Chi tiết đầy đủ: [docs/handoff/PLAN-2026-08-19-itc-temp-probe.md](docs/handoff/PLAN-2026-08-19-itc-temp-probe.md).

### Explanation-aware learning (ĐÃ TẮT trong production từ 2026-08-17)

> **Kết luận: hướng này đã được dừng.** `lambda_explanation` và
> `lambda_explanation_strong` đều là `0.0` trong `mimic_cxr_full.yaml`. Code,
> mask cache và `scripts/evaluate_explanation.py` **được giữ nguyên** — evaluator
> chính là thứ tạo ra bằng chứng dưới đây, và là cách duy nhất để thử lại sau khi
> mở băng encoder. Phần mô tả bên dưới giữ lại để tham chiếu.
>
> **Bằng chứng (A/B có kiểm soát, 5 epoch, cùng seed/manifest/recipe, chỉ khác
> hai lambda; test split, ngưỡng calibrate trên val, `ignore_uncertain`):**
>
> | | ON (0.05/0.25) | OFF (0/0) |
> |---|---:|---:|
> | positive_macro_f1 | 0.8757 | **0.8767** |
> | macro_auroc | 0.7850 | **0.7879** |
> | macro_specificity | 0.3840 | **0.4127** |
> | wall clock | 5:06:44 | **4:25:27** (−15.5%) |
>
> Mọi khoảng tin cậy 95% chồng nhau và mọi chênh lệch nghiêng về OFF.
>
> **Quan trọng hơn: nó cũng không đạt được chính mục tiêu của nó.** `L_exp` tối đa
> hoá phần saliency nằm trong mask — đúng đại lượng `evaluate_explanation.py` đo.
> Baseline trung thực của con số đó là **tỉ lệ diện tích mask**, vì một CAM ngẫu
> nhiên ghi đúng bằng đó: test lung **0.3301**, bbox **0.2366**.
>
> | stream | quần thể | ON | OFF | ngẫu nhiên |
> |---|---|---:|---:|---:|
> | biovil | lung | 0.3692 | 0.3383 | 0.3301 |
> | biovil | bbox | 0.2552 | 0.2533 | 0.2366 |
> | pubmedclip | lung | 0.3810 | 0.4085 | 0.3301 |
> | pubmedclip | bbox | 0.1769 | 0.2021 | 0.2366 |
>
> Một stream nhích đúng hướng +0.031; stream kia lệch ~0.026 **ngược hướng** trên
> cả hai quần thể và nằm **dưới mức ngẫu nhiên** ở bbox. CAM ở mức ngẫu nhiên
> trong **cả hai** nhánh — kể cả nhánh chưa từng bị ràng buộc — nên giới hạn nằm
> ở biểu diễn, không phải ở hàm mất mát. **Với encoder đóng băng, term này chỉ có
> thể đánh lại trọng số kênh của một feature map cố định; nó không dạy được
> encoder nhìn chỗ khác.** Chỉ xem xét lại sau khi mở băng encoder.
>
> ⚠ **Không bao giờ trích dẫn saliency precision mà thiếu baseline diện tích mask
> bên cạnh.** 0.25 trông như một kết quả nhưng bằng ngẫu nhiên.
>
> ⚠ Hai term **không được log riêng** (`blip2_qformer.py:1297` trộn thành một
> scalar), và term strong chỉ kích hoạt trên **869/222.758 study train (0.39%)**,
> đạt trọng số đầy đủ đúng 1 epoch. Nên A/B này kiểm chứng *công thức hiện tại*,
> không phải ý tưởng nói chung.

Với mỗi study có ít nhất một nhãn Positive, score Grad-CAM là tổng Logit
Difference Squared trên các bệnh dương tính:

```text
s = Σ_positive (logit_pos - logit_neg)²
H = ReLU(Σ_c mean_ij(∂s/∂A_cij) · A_c)
H_norm = min-max(H)
H_plus = H_norm · 1[H_norm >= quantile(H_norm, 1-top_k)]
L_exp = 1 - Σ(H_plus · M) / (ΣH_plus + eps)
```

Trong **loss**, `H_plus` giữ giá trị mềm phía trong gate có threshold detach để
double backprop còn gradient. Top-saliency **metric** mới dùng mask nhị phân đúng
Eq. (5). Loss chạy riêng trên BioViL 14×14 và PubMedCLIP 7×7 (CLS không có toạ
độ không gian nên bị loại khỏi lưới CAM); encoder vẫn
đóng băng, nên nó nắn cách projection/MHCAC/head đọc feature chứ không đổi
feature encoder.

Hai `lambda` là cờ bật/tắt duy nhất: chỉ cần một trong hai `> 0` là module bật;
**cả hai bằng `0.0` tắt hoàn toàn CAM capture/double backprop**. Không có key
`explanation.enabled` riêng để tránh mâu thuẫn. Config production hiện ở `0.0` /
`0.0`. Cấu hình lúc còn bật, giữ lại vì bật lại nghĩa là khôi phục đúng khối này:

```yaml
model:
  loss:
    lambda_explanation: 0.05          # weak, mask phổi CheXmask
    lambda_explanation_strong: 0.25   # strong, box MS-CXR theo bệnh
  explanation:
    top_k: 0.2
    strong_top_k: 0.5
    warmup_start_epoch: 2
    warmup_epochs: 2
    streams: [biovil, pubmedclip, swin]
    mask_cache_dir: /mnt/drive1tb/datasets/explanation_masks
```

Warmup: epoch [0]–[1] = 0; [2] = 0.125; [3] = 0.1875; [4]+ = 0.25.

## Stage 2

Entrypoint [`training/run_medgemma_qlora.py`](training/run_medgemma_qlora.py) dùng `google/medgemma-1.5-4b-it` với QLoRA/NF4. Script là single-process, single-GPU và chọn checkpoint bằng validation cross-entropy.

Các `--pipeline-mode` mà CLI fine-tuning thực sự chấp nhận:

| Mode | Vai trò |
|---|---|
| `medgemma_direct` | Mặc định; image tower/projector native của MedGemma, không cần Stage 1 |
| `meta_cxr_qformer` | Q-Former visual soft-token ablation, cần Stage 1 |
| `meta_cxr_qformer_with_mhcac_prompt` | Q-Former soft tokens cộng structured P/N/U cues, cần Stage 1 |
| `meta_cxr_native_qformer_guided` | **Kiến trúc thiết kế ban đầu (arm C)**: MedGemma giữ vision tower riêng **cộng thêm** 32 Q-Former soft tokens **cộng** P/N/U cues. Khác hai mode trên ở chỗ soft token **bổ sung** cho ảnh chứ không **thay thế** ảnh. Cần Stage 1 và bắt buộc `--prompt-config` |
| `text_only_language_prior_ablation` | Ablation không có ảnh; không phải vision pipeline |
| `both_for_ablation` | Chạy `medgemma_direct`, sau đó `meta_cxr_qformer` trên cùng một GPU |

`training/pipeline_modes.py` còn đăng ký hai mode dành riêng cho external-checkpoint inference. `pretrained_medgemma_findings_first` chạy qua `medgemma_inference.run_pretrained_findings`, không qua fine-tuning CLI; `pretrained_medgemma_impression_phase2` chỉ được khai báo và đang bị runtime guard vô hiệu hóa.

Các mode Q-Former chỉ hỗ trợ target `findings_only`. Native route còn hỗ trợ `impression_only` và `findings_and_impression`; Prompt v2 được thiết kế để sinh FINDINGS.

Prompt v2 định nghĩa riêng năm visual mode trong [`stage2/prompts/schemas.py`](stage2/prompts/schemas.py):

- `native_anchor_only`
- `native_anchor_guided`
- `native_multiview`
- `qformer_visual_only`
- `qformer_guided`

`qformer_visual_only` không nhận Stage-1 labels. Chỉ guided modes đưa structured predictions vào prompt, và các prediction này được mô tả là auxiliary cues có thể sai, không phải ground truth. Train và inference đi qua cùng `PromptBuilder`; prompt prefix được mask khỏi training labels, còn Q-Former special token được đưa vào `bad_words_ids` khi generation.

## Prompt v2

Prompt v2 ở [`configs/stage2_prompt_v2.yaml`](configs/stage2_prompt_v2.yaml) là opt-in qua `--prompt-config`; nếu bỏ flag này, code giữ legacy prompt. Thiết kế hiện tại:

- dùng compact summary khi Stage 1 không dự đoán positive/uncertain;
- giới hạn negative findings bằng policy và số lượng tối đa;
- diễn đạt uncertain findings bằng ngôn ngữ thận trọng;
- thêm guard cấm temporal comparison khi không có prior;
- lưu prompt version, config hash và template hash trong artifact metadata;
- tách visual-only khỏi structured labels để tránh label leakage qua prompt.

Temporal target policy mặc định vẫn là `keep`; guard trong prompt không đồng nghĩa dữ liệu train đã có prior linkage. Chi tiết:

- [Stage 2 prompt design](docs/stage2_prompt_design.md)
- [Prompt audit](docs/stage2_prompt_audit.md)
- [Prompt ablation](docs/stage2_prompt_ablation.md)
- [Temporal-target audit](docs/stage2_temporal_target_audit.md)

## Evaluation

Evaluator nằm trong [`training/evaluation/`](training/evaluation/) và được gọi qua CLI trong `scripts/`.

### Stage 1

- positive macro precision, recall và F1;
- per-pathology metrics, AUROC và AUPRC;
- threshold calibration chỉ trên validation;
- bootstrap confidence intervals;
- three-class confusion matrices, ROC/PR và các plot tùy chọn;
- all-negative và các baseline comparisons.

#### `--label-framing` — câu hỏi mà metric đang trả lời (từ 2026-08-20)

Ma trận nhãn CheXpert có hai cách đọc, và **F1 chỉ có nghĩa ở một trong hai**.
`training/evaluation/label_framing.py` đặt tên cho cả hai và ghi lựa chọn vào mọi
file kết quả, giống cách `uncertain_policy.py` xử lý lớp Uncertain.

| | `masked_polarity` (mặc định, lịch sử) | `study_presence` |
|---|---|---|
| Ô trống nghĩa là | **bị mask** — chấm polarity *với điều kiện* đã được nhắc | **không có** |
| Prevalence mỗi nhãn (test) | 0.13–1.00, 12/14 nhãn > 0.55 | **0.019–0.344** |
| `all_positive` ăn được macro F1 | **0.8397** | **0.2280** |
| `all_negative` / `majority_class` | 0.0000 / 0.8200 | 0.0000 / 0.0000 |
| Nhãn suy biến trên test | 3 | **0** |

`masked_polarity` đúng cho **hàm loss** nhưng hỏng cho **F1**: một hằng số hơn model
0.032, và calibrate ngưỡng chỉ mua thêm 0.0004 so với ngưỡng 0.5. **Chỉ trích dẫn F1
dưới `study_presence`.**

`--score marginal_presence` nhân thêm mention gate: `P(có) = sigmoid(mention) × q_pos`.
Cần `mention_probabilities` trong `.npz` (chỉ có ở run mà eval hook thu gate). Không có
thì raise, **không** âm thầm rơi về `conditional_positive`.

⚠ Calibrate và evaluate phải dùng **cùng một cặp** `--label-framing` / `--score`;
`evaluate_stage1.py` từ chối chạy nếu lệch.

#### `--selection plateau` — chọn ngưỡng bền hơn (từ 2026-08-20)

Đứng ở **trung vị vùng đạt ≥ 95% đỉnh** thay vì đúng đỉnh đường cong mục tiêu. Trên val
1,808 study, đỉnh phụ thuộc study nào tình cờ rơi vào val. Kết hợp `--min-positive 5`
để hai nhãn hiếm (`Pleural Other` 15 dương, `Fracture` 18) được calibrate thật thay vì
rơi về ngưỡng mặc định 0.5 — ở mặc định cũ **`Fracture` không bao giờ được dự đoán dương,
F1 = 0.0000**, và riêng hai nhãn đó chiếm **59%** khoảng cách tới trần.

```bash
--selection plateau --plateau-fraction 0.95 --min-positive 5
```

Chọn bằng CV 5-fold × 10 lần **bên trong val** (0.3246 vs 0.3202) trước khi chạm test.

### XAI / Grad-CAM

Ba metric theo mục III.C của bài báo explanation-aware:

- top saliency precision: tỉ lệ pixel trong top-50% nhị phân nằm trong mask;
- all saliency precision: tỉ lệ toàn bộ khối lượng CAM liên tục nằm trong mask;
- annotation coverage: tỉ lệ **từng bbox** MS-CXR có ít nhất 1% pixel salient.

Báo cáo luôn tách `mask_source=0` (lung anatomical prior) khỏi `mask_source=1`
(expert pathology bbox). Hai nhóm không có aggregate chung. Annotation coverage
ở nhóm lung là `unavailable`, không phải 0.

XAI không đi qua `evaluate_stage1.py`: script đó cố ý model-free và chỉ đọc
`.npz`, trong khi Grad-CAM cần graph autograd sống; evaluation hook LAVIS còn có
`@torch.no_grad()`. Entrypoint riêng dùng `model.eval()` với grad nhưng không
optimizer hay update:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_explanation.py \
  --checkpoint <checkpoint_best.pth> \
  --cfg-path pretraining/configs/mimic_cxr_full.yaml \
  --split test \
  --mask-cache-dir /mnt/drive1tb/datasets/explanation_masks \
  --ms-cxr-csv /mnt/drive1tb/datasets/ms-cxr/MS_CXR_Local_Alignment_v1.1.0.csv \
  --output-dir /mnt/drive1tb/private-results/xai \
  --save-cams --export-figures 12 --device cuda
```

`metrics.json` luôn được ghi; `cams.npz` chỉ khi có `--save-cams`; N PNG overlay
chỉ khi `--export-figures N`. PNG/NPZ là dữ liệu bệnh nhân: script từ chối path
trong repo nếu Git không xác nhận path đó đã ignore, không dùng identifier trong
tên figure và không in identifier ra stdout.

### Stage 2

- BLEU và ROUGE-L được implement trong repository;
- METEOR, CIDEr và BERTScore dùng package tham chiếu tùy chọn;
- per-sample error analysis, subgroup analysis và cờ possible temporal hallucination;
- bootstrap intervals cho các per-sample metric khả dụng.

#### Sinh báo cáo trước khi chấm — `scripts/generate_stage2_reports.py`

`evaluate_stage2.py` đọc `.jsonl`; file đó do script này sinh ra. Từ 2026-09-07
script nhận `--pipeline-mode` và tự chọn nguồn record: `medgemma_direct` đọc
split CSV, còn mọi mode `meta_cxr_*` gọi `build_stage1_records` để lấy 32 soft
token và cue P/N/U. Trước đó script hardcode đường native nên **arm C không có
cách nào sinh báo cáo**.

```bash
# arm C, trên đúng những study mà arm A đã sinh
python scripts/generate_stage2_reports.py \
    --pipeline-mode meta_cxr_native_qformer_guided \
    --prompt-config configs/experiment_native_qformer_guided.yaml \
    --adapter <ft_guided_full>/adapters/medgemma_qlora_meta_cxr_native_qformer_guided \
    --checkpoint-root <run_20260820_ft> --stage1-cache-dir <ft_guided_full> \
    --restrict-to <gen_armA>/generated_test.jsonl \
    --output-dir <private>/gen_armc --split test
```

⚠⚠ **Hai nguồn record cho hai cohort KHÁC NHAU.** Native lọc PA/AP rồi lấy
`DataFrame.sample`; nhánh Stage-1 đi theo thứ tự dataset, lọc bằng
`generation_mask`, không lọc view. Cùng `--limit` **không** phải cùng study.
`--restrict-to` là thứ khiến phép so sánh có giá trị — `sample_key` là
`blake2b(dicom_id)` nên khớp trên cả hai nhánh và với mọi file `.jsonl` cũ.

⚠ Mode soft-token **từ chối chạy zero-shot** và **từ chối `--adapter` thiếu
`img_proj.pt`**: `load_img_proj_if_present()` im lặng khi thiếu file, để lại
projector khởi tạo ngẫu nhiên, và báo cáo sinh ra sẽ trôi chảy nhưng mô tả
nhiễu mà không có lỗi nào.

⚠ `--stage1-cache-dir` trỏ vào thư mục output của lần **train** thì dùng lại
được pass encode Stage-1 (~73 phút). Chỉ trúng khi `--checkpoint-root` và
`--stage1-run` khớp lần train đó.

Clinical adapters hiện chỉ khai báo CheXbert, RadGraph và CheXpert labeler. Chúng cần dependency/checkpoint riêng và chưa được wire/validate để trả metric; evaluator báo `unavailable` hoặc `not implemented`, không thay bằng điểm 0. Không nên suy diễn lexical metrics thành độ đúng lâm sàng. Xem [evaluator validation](docs/evaluator_validation.md) và [evaluator audit](docs/evaluator_audit.md).

## Cấu trúc repository

```text
Meta-CXR/
├── configs/                 environment, experiment và prompt configs
├── pretraining/             Stage 1 entrypoint và configs
├── training/                Stage 2, data I/O và evaluation implementation
│   └── evaluation/          gồm explanation_metrics.py thuần NumPy
├── stage2/                  Prompt v2 builder và policies
├── model/                   LAVIS fork và model integrations
├── mhcac/                   abnormality classification và view fusion
├── vision_encoders/         visual encoder implementations
├── scripts/                 preflight, calibration, classification/generation/XAI CLIs
├── tests/                   CPU test suite
└── docs/                    hướng dẫn và audit chi tiết
```

Không có thư mục top-level `evaluation/`; evaluator hiện nằm tại `training/evaluation/`.

## Cài đặt

```bash
git clone https://github.com/minhphuong150505/Meta-CXR.git
cd Meta-CXR
```

Project yêu cầu Python 3.10 trở lên. Stage 1 và Stage 2 có requirement files riêng; hướng dẫn VM dùng hai virtual environment để cô lập runtime:

```bash
python3 -m venv .venv-stage1
source .venv-stage1/bin/activate
pip install -U pip
pip install -r requirements-stage1.txt
deactivate

python3 -m venv .venv-stage2
source .venv-stage2/bin/activate
pip install -U pip
pip install -r requirements-stage2.txt
deactivate
```

> **Môi trường trên máy train đã được dựng lại ngày 2026-08-18** tại
> `~/.venvs/meta-cxr-stage1-311`: **Python 3.11.16, torch 2.9.1+cu129,
> torchvision 0.24.1+cu129, transformers 4.53.2**. Đã xác minh trên GPU thật —
> `capability (12, 0)`, `sm_120` có trong `get_arch_list()`, matmul chạy được.
> Checkout ở `~/Meta-CXR`. Test CPU: **565 passed, 4 failed, 4 skipped** (4 fail
> là baseline cũ, do thiếu `configs/env_config.yaml` riêng tư).
>
> ⚠ Python hệ thống là **3.14** và không dùng được — bản 3.11 lấy qua `uv`.
> ⚠ **Đừng `pip install -r requirements-stage1.txt` nguyên bản**: nó ghim
> `torch==2.5.1` (kernel chỉ tới sm_90) trong khi GPU là sm_120. Cài torch từ
> kênh cu129 trước, rồi cài phần còn lại sau khi bỏ hai dòng torch.
> Chi tiết: mục "The venv" trong `CLAUDE.md`.
>
> RTX 5060 Ti là kiến trúc **sm_120**, cần CUDA 12.8 trở lên. Bản torch build
> cho CUDA 12.4 chỉ có kernel tới sm_90 và sẽ chết bằng
> `CUDA error: no kernel image is available for execution on the device` —
> nhưng chỉ sau khi đã nạp xong toàn bộ model, nên rất dễ tưởng là lỗi code.
> Một venv cũ như vậy (`meta-cxr-rtx4060`) đã bị xoá ngày 2026-08-14.

`requirements-stage2.txt` bao gồm Stage 1 requirements rồi bổ sung Accelerate, bitsandbytes, PEFT và các package Stage 2. MedGemma là gated model; dùng `HF_TOKEN` hoặc đăng nhập Hugging Face và không ghi credential vào repository.

## Cấu hình

```bash
cp configs/env_config.yaml.example configs/env_config.yaml
```

Điền đường dẫn local trong `configs/env_config.yaml`; không commit file local, token hoặc credential. [`configs/env_config.yaml.example`](configs/env_config.yaml.example) mô tả:

- root chứa trực tiếp `files/` của MIMIC-CXR-JPG;
- train/val/test CSV trong `processed/full_allviews/`;
- output/checkpoint directories;
- output local và Weights & Biases settings nếu dùng.

`image_path` trong processed CSV là đường dẫn tương đối dạng `files/p1X/.../<dicom>.jpg` và được nối với `mimic_cxr_jpg_root`; không đổi nó thành đường dẫn tuyệt đối.

## Dữ liệu

MIMIC-CXR là dữ liệu hạn chế truy cập theo DUA. Người dùng phải tự có quyền truy cập hợp lệ; ảnh, report text, processed splits, credentials và model artifacts không được phân phối trong repository. Pipeline hiện nhắm tới full p10–p19 splits, không phải notebook p10 cũ. Cấu trúc mount chi tiết nằm trong `configs/env_config.yaml.example`.

Dữ liệu explanation trên máy train:

| Nguồn | Vị trí | Ghi chú đã xác minh |
|---|---|---|
| CheXmask OriginalResolution | `/mnt/drive1tb/datasets/chexmask/MIMIC-CXR-JPG.csv` | Header thật là `dicom_id` (không phải `Image ID`); dùng hai phổi, Dice mean ≥0.7 |
| MS-CXR v1.1.0 | `/mnt/drive1tb/datasets/ms-cxr/MS_CXR_Local_Alignment_v1.1.0.csv` | bbox pixel ảnh gốc; **không dùng cột `split`** |
| Cache đề xuất | `/mnt/drive1tb/datasets/explanation_masks/` | `masks_<split>.npy` + `index_<split>.json`, private |

Cột `split` MS-CXR không khớp manifest project: đã thấy 166 bbox họ gọi là
`train` nằm trong test của project. Luôn join theo `dicom_id` rồi để manifest
project quyết định split. Với PhysioNet restricted files, dùng `wget --user ...
--ask-password`: server chỉ nhận Basic auth sau challenge 401; `curl -n` gửi
preemptive và trả 403 không giúp phân biệt credential sai với thiếu quyền.

Build cache (CPU, trên máy có data mount):

```bash
python preporcessing/build_explanation_masks.py --inspect
python preporcessing/build_explanation_masks.py \
  --split all \
  --chexmask-csv /mnt/drive1tb/datasets/chexmask/MIMIC-CXR-JPG.csv \
  --ms-cxr-csv /mnt/drive1tb/datasets/ms-cxr/MS_CXR_Local_Alignment_v1.1.0.csv \
  --output-dir /mnt/drive1tb/datasets/explanation_masks
```

Geometry là Resize cạnh ngắn 512 → CenterCrop 448 → nearest 112². Smoke thật
với `--split val --limit 200` cho 193 mask hợp lệ: 189 lung, 4 bbox; lung phủ
18,2–52,9% (trung vị 32,6%), bbox union phủ 3,5–18,2%. Đây là kiểm chứng CPU
cache, **không phải** kiểm chứng GPU loss/evaluator.

## Quick start

Các lệnh dưới đây là entrypoint hiện có. Chúng cần environment, dữ liệu và checkpoint tương ứng; chưa được xác nhận bằng GPU run trên commit hiện tại.

### 1. VM preflight

```bash
python scripts/vm_preflight.py
python scripts/vm_preflight.py --stage 1
```

Preflight không tải model weights; nó kiểm tra Python, CUDA/GPU, RAM/disk/shared memory, imports, paths và Hugging Face auth.

### 2. Stage 1 smoke test và training

Stage 1 hỗ trợ `run.truncate_train/val/test` cho smoke test. Config production
chạy 10 epoch, early stopping patience 5 (bất động với eval bắt đầu ở epoch [5]) và chọn checkpoint theo
macro-AUPRC; logits validation được lưu để calibrate threshold F1 sau đó.

```bash
CUDA_VISIBLE_DEVICES=0 python -m pretraining.train \
  --cfg-path pretraining/configs/mimic_cxr_full.yaml \
  --options run.batch_size_train=6 run.batch_size_eval=6 run.accum_grad_iters=11
```

Sau khi train, calibrate threshold chỉ trên prediction của validation từ
`checkpoint_best` (các bệnh có dưới 20 positive giữ threshold 0.5):

```bash
python scripts/calibrate_thresholds.py \
  --predictions pretraining/outputs/<run>/result/val_predictions_epoch_best.npz \
  --objective f1 --uncertain-policy ignore_uncertain --min-positive 20 \
  --output pretraining/outputs/<run>/result/f1_thresholds.json
```

### 3. Stage 2 smoke test và training

```bash
CUDA_VISIBLE_DEVICES=0 python training/run_medgemma_qlora.py \
  --train-limit 500 --val-limit 10 --test-limit 10 --no-upload \
  --output-dir training/outputs/smoke

CUDA_VISIBLE_DEVICES=0 python training/run_medgemma_qlora.py \
  --pipeline-mode medgemma_direct --no-upload \
  --output-dir training/outputs/medgemma_direct_full
```

Prompt v2/Q-Former route cần Stage 1 checkpoint và chỉ sinh FINDINGS:

```bash
CUDA_VISIBLE_DEVICES=0 python training/run_medgemma_qlora.py \
  --pipeline-mode meta_cxr_qformer --section-mode findings_only \
  --prompt-config configs/stage2_prompt_v2.yaml \
  --checkpoint-root pretraining/outputs --no-upload
```

### 4. Evaluation

```bash
python scripts/calibrate_thresholds.py \
  --predictions <validation_predictions.npz> --split validation \
  --output <thresholds.json>

python scripts/evaluate_stage1.py \
  --predictions <test_predictions.npz> --thresholds <thresholds.json> \
  --output-dir <stage1_eval_dir>

CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_explanation.py \
  --checkpoint <checkpoint_best.pth> \
  --cfg-path pretraining/configs/mimic_cxr_full.yaml --split test \
  --mask-cache-dir /mnt/drive1tb/datasets/explanation_masks \
  --output-dir /mnt/drive1tb/private-results/xai --export-figures 12

python scripts/evaluate_stage2.py \
  --predictions <generated_reports.jsonl> \
  --metrics bleu,rouge,meteor,cider,bertscore \
  --skip-clinical-metrics --output-dir <stage2_eval_dir>
```

## Hỗ trợ nhiều GPU

Máy train chỉ có một GPU nên đây không còn là workflow được hỗ trợ. Stage 1 vẫn
chạy plain (một tiến trình); code DDP còn trong
LAVIS fork nhưng không có config nào dùng và chưa từng được test. Stage 2
**không hỗ trợ DDP** và không dùng `device_map` rộng để thay thế.

## Testing

CPU checks đã chạy thật trong Phase 3 (2026-08-14):

```bash
CUDA_VISIBLE_DEVICES="" python -m pytest tests/ -q \
  --ignore=tests/test_blip2_negative_sampling.py \
  --ignore=tests/test_encoder_ablation.py
CUDA_VISIBLE_DEVICES="" python -m compileall -q \
  stage2 training scripts runtime safety tests medgemma_inference
```

Với hai file cần torchvision bị ignore theo lệnh chuẩn, kết quả thật là **541
passed, 5 failed, 1 skipped**. Bảy test metric XAI mới đều pass. Năm failure là
baseline có sẵn: `test_native_independence` ×4 thiếu
`configs/env_config.yaml`, và `test_stage1_eval_hook` ×1 thiếu torchvision; không
phát sinh từ Phase 3. Test CPU không thay thế smoke Stage-1/Stage-2/XAI trên GPU.

## Kết quả và cảnh báo metric

### Stage 1 — test split, `run_20260820_ft` (2026-08-21) — PHIÊN BẢN HIỆN TẠI

Mở băng phần đỉnh của cả hai vision encoder (31.85M / 181.3M tham số đóng băng, ở
`init_lr_enc` 1e-5) và hạ mọi kappa về 1. 10 epoch, `rc=0`, 15h33m, 0 restart,
0 kernel fault. Best epoch **9 — epoch cuối**, và val loss **vẫn đang giảm** ở đó.

`study_presence` + `marginal_presence`, ngưỡng calibrate trên val (plateau, minpos 5):

| Metric | Giá trị | 95% CI |
|---|---:|:---:|
| `macro_auroc` | **0.7643** | [0.7545, 0.7740] |
| `positive_macro_f1` | **0.3542** | [0.3418, 0.3647] |
| `macro_auprc` | 0.3203 | [0.3108, 0.3362] |
| `positive_macro_recall` | 0.5373 | [0.5184, 0.5555] |
| `positive_macro_precision` | 0.2931 | [0.2816, 0.3038] |

**So với phiên bản trước, bootstrap ghép cặp trên cùng 3,269 study:**
ΔAUROC **+0.0201** [+0.0136, +0.0263] · ΔF1 **+0.0144** [+0.0053, +0.0236] ·
Δprecision **+0.0154** [+0.0073, +0.0238] · Δrecall **+0.0368** [+0.0205, +0.0535].
**Cả bốn đều có ý nghĩa và cùng chiều** — mô hình tốt hơn thật, không phải dịch điểm
vận hành. AUROC tăng trên **14/14 nhãn**. Trần precision nhấc từ ~0.40 lên **~0.52**.

Chi phí: **+37%** thời gian mỗi epoch. ⚠ kappa và mở băng đi chung một run nên **chưa
tách được đóng góp của từng cái**; lập luận gián tiếp là kappa không đổi được AUROC.

Chi tiết đầy đủ: `Eval/stage1_test_02/README.md` (git-ignored).

#### ⚠ Train thêm epoch KHÔNG giúp — đã đo, `run_20260821_ext` (2026-08-21)

Val loss vẫn đang giảm ở epoch 9 nên đã thử resume với `run.max_epoch=15`. Epoch
10–14 chạy **7h32m**, `rc=0`, 0 restart, 0 kernel fault, và **không thu được gì**.

- Không epoch nào vượt epoch 9 theo `val_loss`; `checkpoint_best` không bị ghi đè
  lần nào. Train loss vẫn giảm (1.848 → 1.836) trong khi khoảng cách train–val nới
  từ +0.0363 lên +0.0517.
- Trên **val**, epoch 14 hơn epoch 9 macro AUROC **+0.0082** [+0.0015, +0.0148],
  P = 0.995 — CI không cắt 0. Nhưng trên **test** thì thành **−0.0005**
  [−0.0043, +0.0031]. **Không lặp lại được.**
- Test, bootstrap ghép cặp cùng 3,269 study: ΔAUROC −0.0005 · ΔF1 −0.0067 ·
  Δprecision +0.0015 — **cả ba CI đều cắt 0**. Chỉ có Δrecall **−0.0826** và
  Δspecificity **+0.0244** là có ý nghĩa, và đó chỉ là ngưỡng calibrate lại dịch
  điểm vận hành. AUROC cải thiện trên **6/14 nhãn** (tung đồng xu); lần mở băng
  encoder cải thiện 14/14.

**Kết luận: phiên bản 02 / epoch 9 là mô hình Stage-1 cuối cùng.** Ba bài học
đáng mang đi: (1) CI val sát 0 trên ~1,800 mẫu thì coi như chưa có kết luận;
(2) kiểm tra macro với micro trước khi tin một mức tăng macro — ở đây val macro
AUROC tăng còn micro AUROC *giảm*, tức "lợi ích" nằm ở vài nhãn hiếm nhiễu nhất;
(3) tách nhỏ điểm số — bỏ mention gate ra thì epoch 14 **thua** epoch 9 trên `q`
(0.7304 vs 0.7354), nên chênh lệch chỉ là may mắn ở tích `m × q`.

Chi tiết đầy đủ: `Eval/stage1_test_03/README.md` (git-ignored).

#### ❌ `run_20260821_deep` — mở băng sâu hơn KHÔNG giúp gì (đo 2026-08-22)

Mở băng thêm ResNet50 `layer3` và CLIP block 8–9: **31.85M → 53.12M** trên tổng
181.3M tham số encoder (158 tham số, 8 pattern). Đây là thay đổi **duy nhất** so
với `run_20260820_ft` — kappa, batch 16×4, `init_lr_enc` 1e-5, 10 epoch và mọi
trọng số loss giữ nguyên — nên là **ablation sạch về độ sâu mở băng**.

Run chạy hết sạch: 10/10 epoch, `rc=0`, **14h03m**, 0 lần restart, 0 kernel
fault, `max mem` **9,839 MiB** (đúng con số 9,847 đo trước khi phóng).

Bootstrap **ghép cặp** trên cùng 3,269 study test, 2,000 lần lấy mẫu, framing
`study_presence` + `marginal_presence`, mỗi run dùng ngưỡng calibrate trên val
của chính nó:

| | `run_20260820_ft` (nông) | `run_20260821_deep` | Δ, 95% CI |
|---|---:|---:|:---:|
| `macro_auroc` | 0.7643 | 0.7692 | +0.0049 [−0.0003, +0.0100] |
| `micro_auroc` | 0.8166 | 0.8187 | **+0.0021 [+0.0001, +0.0040]** |
| `positive_macro_f1` | 0.3542 | 0.3518 | −0.0023 [−0.0128, +0.0088] |
| `positive_macro_precision` | 0.2931 | 0.3008 | +0.0077 [−0.0031, +0.0204] |
| `positive_macro_recall` | 0.5373 | 0.4436 | **−0.0937 [−0.1106, −0.0769]** |
| `macro_specificity` | 0.8020 | 0.8395 | **+0.0375 [+0.0343, +0.0407]** |

**`macro_auroc` không vượt qua 0.** Hai thay đổi lớn duy nhất có ý nghĩa là
recall giảm và specificity tăng — đó là **điểm vận hành dịch chuyển**, không phải
mô hình tốt hơn; cùng chữ ký với `run_20260821_ext`, và ngược hẳn với lần mở băng
nông nơi cả bốn chỉ số cùng đi lên. `micro_auroc` vượt 0 nhưng chỉ +0.0021, quá
nhỏ để hành động. Per-label AUROC: deep thắng **10/14** nhãn, mean +0.0049
(binomial P ≈ 0.09) — so với **14/14** của lần mở băng nông. `macro_auprc`
0.3203 → 0.3269.

⚠ **`run_20260820_ft` / epoch 9 vẫn là mô hình Stage-1 cuối cùng.** `val_loss`
của run deep thấp hơn ở **mọi** epoch được chấm (best 1.8739 so với 1.8843) mà
vẫn không cho mô hình test tốt hơn — thêm một lý do đừng chọn checkpoint chỉ
bằng val loss ở đây.

⚠ F1 tụt chủ yếu do calibration chứ không phải mô hình: ngưỡng `Fracture` ra
**0.407** trên deep so với 0.235 trên nông (chỉ 18 positive trên val), sang test
thành precision 0.043 / recall 0.011, trong khi AUROC gần như y hệt (0.6670 vs
0.6657). Với nhãn hiếm, hãy coi F1 dao động là nhiễu ngưỡng cho tới khi AUROC
đồng ý.

⚠ `init_lr_enc` 1e-5 áp chung cho cả tầng nông lẫn tầng sâu, và đây là thứ phải
sửa đầu tiên nếu ai muốn thử lại. Chuẩn mực là layer-wise LR decay và repo hiện
chưa biểu diễn được (chỉ có một nhóm optimizer cho encoder). Nên kết quả trên là
bằng chứng chống lại **cấu hình sâu này**, không phải chống lại độ sâu nói chung
— muốn thử lại cần thêm nhóm LR theo pattern, tức sửa code chứ không phải sửa
config.

Thời gian: 14h03m so với 15h33m của bản nông. ⚠ Đừng đọc thành "sâu hơn thì
nhanh hơn" — hai run gặp điều kiện dataloader khác nhau và không run nào là phép
đo throughput có kiểm soát.

### Stage 1 — test split, `run_20260819_xmpoff` (2026-08-20) — phiên bản 01

Run 10 epoch trên toàn bộ MIMIC-CXR, `checkpoint_best` chọn ở epoch 6 theo `val_loss`.
Test split (3,269 study) được giữ kín và chấm **đúng một lần**; ngưỡng calibrate **chỉ
trên validation**.

Số chính, dưới framing `study_presence` + `marginal_presence` (xem mục Evaluation):

| Metric | Giá trị | 95% CI |
|---|---:|:---:|
| `macro_auroc` | **0.7441** | [0.7341, 0.7540] |
| `positive_macro_f1` | **0.3397** | [0.3275, 0.3515] |
| `macro_auprc` | 0.3004 | [0.2919, 0.3139] |
| `positive_macro_recall` | 0.5006 | [0.4811, 0.5206] |
| `positive_macro_precision` | 0.2776 | [0.2657, 0.2888] |
| `micro_auroc` | 0.8183 | – |

So với baseline tầm thường trên **cùng** framing: `all_positive` 0.2280,
`all_negative` 0.0000, `majority_class` 0.0000.

Ngưỡng calibrate trên val bằng `--selection plateau --plateau-fraction 0.95
--min-positive 5` (chọn bằng CV trong val, xem mục Evaluation). So với `argmax` +
`min-positive 20`, bootstrap **ghép cặp** 2,000 lần cho ΔF1 **+0.0174**
[+0.0102, +0.0243], Δrecall **+0.0747**, Δprecision **−0.0211** — cả ba có ý nghĩa.
Đây là **đánh đổi 3.5:1 nghiêng về recall**, không phải quà miễn phí.

⚠ **Precision không mua được bằng ngưỡng.** Ép sàn precision 0.70 trên val chỉ giao
0.4074 trên test và vứt 84% recall; precision bão hòa quanh 0.40. Trần đó thuộc về
**mô hình**. Năm nhãn có precision dưới 0.25 — chưa dùng lâm sàng được.

Bốn nhãn khỏe nhất (AUROC): Support Devices 0.8783, Pleural Effusion 0.8682,
Pneumothorax 0.8283, Edema 0.8245. Yếu nhất: Enlarged Cardiomediastinum 0.6126.

**Mention gate đóng góp thật:** `m × q_pos` hơn `q_pos` trần về AUROC ở **14/14 nhãn**,
trung bình **+0.0831** (`macro_auroc` 0.6767 → 0.7441, `macro_specificity`
0.6359 → 0.8214). Lớn nhất ở đúng những nhãn mà "có được nhắc tới không" mang phần lớn
tín hiệu: No Finding +0.2603, Pleural Other +0.2501, Fracture +0.1798.

⚠ **Chưa có kết quả Stage 2** cho checkpoint này, nên không có metric NLG
(BLEU/ROUGE/METEOR/CIDEr/BERTScore). RadGraph/RadCliQ/CheXbert/RadFact **không cài
được** trong repo — báo là *unavailable*, không bao giờ báo là 0.

⚠ **Checkpoint này không dùng được cho Stage-2 chế độ `meta_cxr_qformer`** (soft token):
`lambda_itc/itm/lm` = 0 nên đường ảnh của Q-Former bị skip toàn bộ trong Stage 1.
`medgemma_direct` không ảnh hưởng.

Chi tiết đầy đủ (bảng từng nhãn, so sánh 5 arm, đóng góp của gate, lệnh tái tạo) nằm ở
`Eval/stage1_test/README.md` — thư mục đó **git-ignored** vì file `.npz` chứa định danh
study của MIMIC-CXR.

### ❌ Stage 2 — cắm Stage 1 vào prompt KHÔNG giúp (đo 2026-09-07/08)

Đây là **kết quả âm**, và nó được giữ lại nguyên vẹn vì phép đo là có kiểm soát:
cùng checkpoint, cùng cohort, cùng decoding, và **được nhân bản độc lập bởi hai
cài đặt khác nhau**.

#### Bối cảnh: hai arm

| Arm | Kiến trúc | Huấn luyện |
|---|---|---|
| **A** | `medgemma_direct` — chỉ vision tower của MedGemma | 0,8621 epoch (bị dừng), không có validation, `best_val_loss` = `+inf` |
| **C** | `meta_cxr_native_qformer_guided` — ảnh **cộng** 32 Q-Former soft token **cộng** cue P/N/U từ MHCAC | trọn 1 epoch, 82h28m, `val_loss` **1,0019** |

Arm C là kiến trúc được thiết kế ban đầu của dự án, và nó được train **nhiều
hơn** arm A.

#### 1. Arm C kém hơn arm A trên cohort khớp

2.772 study test mà **cả hai arm đều sinh ra output** (arm A lọc PA/AP rồi lấy
mẫu ngẫu nhiên; nhánh Stage-1 đi theo thứ tự dataset lọc bằng `generation_mask`
— hai cohort khác nhau, phải giao nhau mới so được). Bootstrap ghép cặp, 1.000
lần lấy mẫu:

| | Arm A | Arm C | Δ (C−A), CI95 |
|---|---:|---:|:---|
| BLEU-4 | 0,0763 | 0,0605 | — |
| ROUGE-L | 0,2360 | 0,2161 | **−0,0198 [−0,0233; −0,0165]** |
| METEOR | 0,2547 | 0,2548 | +0,0001 [−0,0048; +0,0048] |
| CIDEr | 0,0584 | 0,0164 | **−0,0420 [−0,0518; −0,0334]** |
| BERTScore-F1 | 0,7959 | 0,7587 | **−0,0372 [−0,0403; −0,0343]** |

Ba trên bốn chỉ số có CI cho thấy Arm C kém hơn; METEOR chưa xác lập khác biệt.
Hai adapter có lịch huấn luyện khác nhau, nên phép so này chưa tách riêng tác
động kiến trúc, lượng huấn luyện và quá trình tối ưu.

#### 2. Cues cũ có ảnh hưởng bất lợi; soft token chưa bị chứng minh là nguyên nhân

Can thiệp lúc suy luận trên một checkpoint arm C cố định, 100 study val, Δ ghép
cặp so với đầu vào đầy đủ:

| Can thiệp | BERTScore-F1 | CIDEr |
|---|:---|:---|
| **Bỏ cue P/N/U** | **+0,0107 [+0,0033; +0,0191]** ✅ | **+0,0093 [+0,0023; +0,0193]** ✅ |
| Zero hoá soft token | +0,0042 [−0,0074; +0,0173] | −0,0020 [−0,0126; +0,0113] |
| Bỏ cả hai | +0,0034 [−0,0084; +0,0160] | +0,0029 [−0,0104; +0,0183] |

**Bỏ cues cũ cải thiện BERTScore và CIDEr trong phép thử này.** Không được gọi
soft token hoàn toàn trung tính: zero hoá làm ROUGE-L giảm 0,0098 với CI loại
trừ 0, dù thay đổi BERTScore/CIDEr chưa xác lập.

#### 3. Cue kém vì `q` trả lời sai câu hỏi

79,5% ô nhãn CheXpert để trống và bị mask khỏi loss, nên đầu classification chỉ
học được *"NẾU bác sĩ có nhắc, thì dương hay âm?"* — nó chưa từng thấy ví dụ
"không xuất hiện trong báo cáo". Đo trên chính file dự đoán của
`run_20260820_ft`, macro 13 nhãn, framing `study_presence`:

| Quy tắc dựng cue | Precision | Recall | Cue/study |
|---|---:|---:|---:|
| `conditional_positive` (arm C đã chạy) | **0,1887** | 0,8097 | **8,46** |
| `mention_gated` (m ≥ 0,60) | 0,2357¹ | 0,4364¹ | 2,30¹ |
| `marginal_positive` (ngưỡng theo nhãn, fit val) | **0,4069** | 0,3135 | **1,46** |

¹ chỉ trên val. Thực tế mỗi study chỉ có **~1,6** bệnh — quy tắc đang chạy khẳng
định gấp **hơn năm lần** thực tế.

Với bệnh hiếm nó suy biến thành hằng số: `Fracture`, `Pleural Other`,
`Lung Lesion` đều cho recall 1,000 với precision đúng bằng prevalence — tức
"luôn luôn nói có". Ngược lại `Support Devices`, bệnh phổ biến nhất (34,4%),
**chưa từng một lần** được gắn Positive.

#### 4. Marginal cues trước sửa lỗi chưa chứng minh giúp generation

Ba điều kiện, một checkpoint arm C cố định, **đúng 100 study val cùng thứ tự**
(đã kiểm chứng `sample_key` trùng khớp), greedy 160 token:

| Quy tắc | BLEU-1 | BLEU-4 | ROUGE-L | METEOR | CIDEr | BERTScore-F1 |
|---|---:|---:|---:|---:|---:|---:|
| `conditional_positive` | 0,2115 | 0,0695 | 0,2358 | 0,2879 | 0,0100 | 0,7698 |
| `marginal_positive` | 0,2129 | 0,0695 | 0,2379 | 0,2847 | 0,0076 | 0,7772 |
| **`none`** | **0,2192** | **0,0723** | 0,2374 | **0,2900** | **0,0176** | **0,7813** |

Bootstrap ghép cặp từng study, 2.000 lần:

| So sánh | BERTScore-F1 | CIDEr |
|---|:---|:---|
| `none` − `conditional` | **+0,0115 [+0,0025; +0,0217]** ✅ | +0,0076 [−0,0049; +0,0243] |
| `marginal` − `conditional` | +0,0073 [−0,0008; +0,0166] ❌ | −0,0024 [−0,0084; +0,0022] |
| `none` − `marginal` | +0,0042 [−0,0041; +0,0118] ❌ | +0,0100 [−0,0009; +0,0279] |

**Đính chính ngữ nghĩa:** dòng `none` trong bảng là kết quả trước sửa lỗi:
nhóm rỗng vẫn sinh câu tóm tắt dự đoán bình thường, chưa xoá hẳn structured block.
Không gán những số này cho `none` đã sửa. Chỉ CI BERTScore của
`none − conditional` lịch sử loại trừ 0; cải thiện từ marginal chưa được xác lập.

#### Nhân bản độc lập

Dòng `conditional_positive` tái lập kết quả của một phép thử viết độc lập
**đến bốn chữ số thập phân** (0,7698 / 0,2115 / 0,2358 / 0,0100), và
`none − conditional` (+0,0115 [+0,0025; +0,0217]) khớp với `no_cues − full`
của phép thử đó (+0,0107 [+0,0033; +0,0191]). Hướng thay đổi giống nhau,
nhưng probe xoá trực tiếp structured block còn `none` cũ giữ câu normal.

#### Kết luận

Ở checkpoint này, bỏ cues cũ cải thiện BERTScore trên 100 ca validation.
Marginal cues chưa chứng minh cải thiện sinh báo cáo; chưa thể suy rộng sang
mọi rule hoặc model được huấn luyện lại. Giả thuyết cues trùng thông tin ảnh
chưa phải kết luận về cơ chế, và NLG không thay thế đánh giá độ đúng lâm sàng.

#### ⚠️ CẢNH BÁO 2026-09-09: mọi số Stage-2 ở trên đo khi còn LỖI DỪNG SINH

MedGemma khai báo stop token `[1, 106]` với `106 = <end_of_turn>`, nhưng code
ghi đè bằng mỗi EOS của tokenizer (`1`). Model **không được phép dừng ở đúng
dấu kết thúc mà target huấn luyện dùng**. Đo trên 25 case val, 4 nhánh trong
một instance model:

| nhánh | tokens | cạn cap 160 | phát `<end_of_turn>` | token viết SAU đó |
|---|---:|---:|---:|---:|
| stop cũ `[1]` | 4.000 | **25/25** | 25/25 | **3.112** |
| stop đúng `[1,106]` | 888 | **0/25** | 25/25 | **0** |

**78% số token sinh ra là phần thừa viết sau khi model đã báo xong**, và 100%
output cạn cap. Đây là phép đếm, không phải ước lượng.

Sửa xong, CIDEr tăng ~25 lần: **0,0065 → 0,1683**, CI95 [+0,0727; +0,2620].
CIDEr là chỉ số chiết khấu khuôn mẫu — nó kẹt ở 0,006–0,06 suốt mọi thí nghiệm
Stage-2 của dự án, và phần lớn cái sàn đó là hiện vật của lỗi này.

⚠ Lỗi **đối xứng** giữa mọi nhánh từng so sánh, nên các kết luận **tương đối**
ở trên nhiều khả năng vẫn đúng. Cái không còn đứng là **mọi con số tuyệt đối**.
Kết luận "cue không giúp" cũng cần đo lại, vì nó được thiết lập trong chế độ mà
78% output là nhiễu lấp lên tín hiệu.

⚠ n=25, cùng một cohort val đã xem trước đó. Số đếm token là chính xác; các
delta NLG là phép thử cơ chế nhỏ, không phải đánh giá held-out. **Việc sinh lại
toàn bộ kết quả với stop đúng CHƯA được làm.**

#### Giới hạn — phải đọc kèm

- **n = 100 cho mục 2 và 4.** CI rộng. `marginal − conditional` ở +0,0073
  [−0,0008; +0,0166] là *suýt* có ý nghĩa: phải báo là **"chưa xác lập ở cỡ mẫu
  này"**, không phải "không có tác dụng".
- **Không phải ablation kiến trúc sạch.** Arm A 0,8621 epoch vs arm C 1 epoch.
- **Can thiệp lúc suy luận ≠ huấn luyện lại.** Model được train *có* cue; bỏ cue
  lúc sinh là lệch phân phối. Chưa chứng minh "train từ đầu không cue sẽ tốt hơn".
- **Đường ảnh của Q-Former chưa từng được huấn luyện** (`lambda_itc/itm/lm` = 0),
  nên 32 soft token là phép đọc BLIP-2 cố định. Kết quả trung tính của chúng là
  đúng như dự đoán cho một nhánh chưa train, **không** phải phán quyết về kiến trúc
  Q-Former.
- **Không có chỉ số lâm sàng nào ở đây.** CIDEr của mọi arm đều gần 0. BLEU/ROUGE/
  BERTScore đo chồng lặp bề mặt; "arm A tốt hơn arm C" **không** có nghĩa arm A tốt.
- **Sinh văn bản bị sụp lặp ở cả hai arm** (arm A 28,1%, arm C 44,3% output có
  5-gram lặp ≥3 lần, người viết 0,1%). Mọi số trên đều đo trong tình trạng đó.

#### Tái lập

```bash
python scripts/generate_stage2_reports.py \
    --pipeline-mode meta_cxr_native_qformer_guided \
    --prompt-config configs/experiment_native_qformer_guided.yaml \
    --adapter <ft_guided_full>/adapters/medgemma_qlora_meta_cxr_native_qformer_guided \
    --checkpoint-root <run_20260820_ft> --stage1-cache-dir <cache> \
    --split val --limit 100 --max-new-tokens 160 \
    --cue-rule marginal_positive \
    --threshold-path configs/stage2_cue_thresholds_marginal_pfit.json \
    --output-dir <private>/marginal
# đổi --cue-rule thành conditional_positive | none cho hai điều kiện còn lại
# (none không cần --threshold-path)
```

### Original paper reference results

Bài báo META-CXR gốc có báo cáo classification và report-generation metrics cho kiến trúc/dữ liệu của công trình đó. Các số trong paper chỉ là tham khảo lịch sử, **không phải kết quả của repository/commit hiện tại**. README này không sao chép bảng số để tránh trộn nguồn; xem bài báo được dẫn trong mục Citation.

## Tài liệu

- [Stage 2 pipeline modes](docs/STAGE2_PIPELINE_MODES.md)
- [Stage 2 prompt design](docs/stage2_prompt_design.md)
- [Stage 2 prompt audit](docs/stage2_prompt_audit.md)
- [Stage 2 prompt ablation](docs/stage2_prompt_ablation.md)
- [Temporal-target audit](docs/stage2_temporal_target_audit.md)
- [Evaluator validation](docs/evaluator_validation.md)
- [Evaluator audit](docs/evaluator_audit.md)
- [MedGemma runtime smoke status](docs/medgemma_real_runtime_smoke.md)
- [Final branch integration audit](docs/final_branch_integration_audit.md)
- [Final merge plan](docs/final_merge_plan.md)
- [Feature cache](docs/FEATURE_CACHE.md)
- [Notebook privacy](docs/notebook_privacy.md)
- [So sánh với repo gốc của bài báo](docs/so_sanh_voi_repo_goc.md)
- [Bàn giao plan → thực thi](docs/handoff/README.md)

## Quy trình làm việc với agent (một Claude lập kế hoạch, một Claude thực thi)

Người thực thi có thể là Claude Code **hoặc** Codex — cả hai đều đã cài và đều
đã chạy việc thật (Codex gần nhất là 2026-09-08). Vai trò không phụ thuộc vào
việc đó là ai.

⚠ Ghi chú sửa 2026-09-08: tài liệu từng ghi "Codex không còn được dùng" từ
2026-08-19. **Điều đó sai** và tồn tại ba tuần. Đừng cho rằng một công cụ không
dùng được chỉ vì tài liệu nói vậy — kiểm tra bằng `command -v`.

⚠⚠ **Một GPU, có thể có nhiều hơn một agent.** Ngày 2026-09-08 hai agent dùng
chung máy train cách nhau vài giờ mà không đụng nhau, vì agent thứ hai kiểm tra
trước khi phóng: `pgrep` tên job đã biết, `nvidia-smi` phải rảnh, và từ chối
chạy nếu thư mục output đã tồn tại. Luật vẫn là **một card, một run, một đường
output** — nhưng nó được bảo đảm bằng việc *kiểm tra*, không phải bằng việc giả
định mình đang ở một mình.

- **Phiên lập kế hoạch** (checkout này, không GPU) đọc code, thiết kế thay đổi, sửa
  source và cập nhật `CLAUDE.md` / `README.md` / `struct/`.
- **Phiên thực thi** (trên máy train) chạy: `git pull`, venv/pytest, preflight,
  smoke, Stage-1/Stage-2 training, `scripts/supervise_stage1.sh`, đọc log. Nó
  **không** tự đổi loss, YAML hay recipe ngoài kế hoạch.
- Chạy phiên thực thi bằng một trong hai cách: SSH trực tiếp (dùng
  `setsid nohup … &` cho việc dài), hoặc Claude Code headless trên máy train:

  ```bash
  ssh phuong@100.116.167.90 'bash -lc "cd ~/Meta-CXR && \
      claude -p \"<yêu cầu>\" --allowedTools \"Bash Read Write Edit Grep Glob\" \
      --model opus"' < /dev/null
  ```

  Ưu tiên `--allowedTools` hơn `--dangerously-skip-permissions`: nó đủ dùng cho mọi
  bước của một plan.

- ⚠ **Chỉ bấm launch MỘT lần.** Ngày 2026-08-19 lệnh bị chạy hai lần cách nhau 50
  giây: phiên thứ hai OOM vì phiên đầu đang giữ GPU, và vì dùng chung đường ghi
  output, báo cáo "abort" của nó đè lên báo cáo thật.
- Bàn giao bằng file, không bằng chat: `docs/handoff/PLAN-<ngày>-<chủ-đề>.md`.
  Bên lập kế hoạch viết plan, bên thực thi nối `## Execution report` vào **cùng
  file**. Xem [docs/handoff/README.md](docs/handoff/README.md).
- **Log lỗi phải được tóm tắt, không dán nguyên file:** lệnh + exit status, lỗi đầu
  tiên kèm ~20 dòng ngữ cảnh và frame traceback cuối, các số quan trọng (`s/it`,
  `max mem`, từng loss term, `epoch`/`iter`, VRAM nếu OOM), và đường dẫn log gốc
  trên host để hỏi `grep` cụ thể sau.
- `scripts/train_healthcheck.sh` là monitor **chỉ đọc** cho cả hai stage. Trước
  một thí nghiệm đã lên lịch, đặt `RUN_DIR`, `LOG` và `EXPECT_RUNNING=1`; process
  biến mất khi đó là ALERT (exit 3), không bị coi là IDLE. Monitor theo dõi mtime
  của cả checkpoint Stage 1 (`*.pth`) lẫn artifact cứu hộ Stage 2
  (`adapter_model.safetensors`, `trainer_state.pt`).
- Khi **Opus hết usage**, chuyển việc cho agent **Sonnet 5** thay vì chờ — Sonnet đủ
  để thực thi kế hoạch đã viết và triage log, rẻ hơn nhiều. Giữ Opus cho quyết định
  kiến trúc/recipe.
- ⚠ **Báo cáo của agent là bằng chứng, không phải sự thật** — đối chiếu với file
  thật trên đĩa (kể cả mtime) trước khi hành động theo nó.

Luật không đổi: mọi thay đổi hành vi phải kèm cập nhật tài liệu trong cùng commit,
và tuyệt đối không đưa dữ liệu bệnh nhân vào commit/handoff/tóm tắt. Chi tiết cho
agent nằm ở [AGENTS.md](AGENTS.md) và `CLAUDE.md`.

## Hạn chế hiện tại

- Stage 2 **training đã chạy trên GPU**: arm A (`medgemma_direct`, 0,8621 epoch —
  bị dừng, không có validation) và arm C (`meta_cxr_native_qformer_guided`, trọn
  1 epoch, `val_loss` 1,0019). Kết quả: xem
  [Stage 2 — cắm Stage 1 vào prompt KHÔNG giúp](#-stage-2--cắm-stage-1-vào-prompt-không-giúp-đo-2026-09-0708).
  Table 5 Stage-1 inference-only encoder ablation đã hoàn tất 4/4 trên full test
  split; xem `results/table5_encoder_ablation.*`.
- Explanation loss chưa từng chạy smoke/full training trên GPU;
  `scripts/evaluate_explanation.py` chưa từng nạp checkpoint/dataset hay chạy
  end-to-end. Không dùng metric/heatmap từ đường này trong luận văn trước smoke.
- Cache explanation mới chỉ được build/kiểm tra ở smoke val 200 study, chưa xác
  nhận full train/val/test cache.
- Stage 2 metric đã có (BLEU/ROUGE/METEOR/CIDEr/BERTScore trên test và val), nhưng
  **chưa có chỉ số lâm sàng nào** và CIDEr của mọi arm đều gần 0 — không suy diễn
  lexical metric thành độ đúng lâm sàng.
- **Sinh văn bản Stage 2 bị sụp lặp ở cả hai arm** — 28,1% (arm A) và 44,3% (arm C)
  output có 5-gram lặp ≥3 lần, so với 0,1% ở báo cáo bác sĩ thật. Mọi số Stage-2 đã
  ghi đều đo trong tình trạng đó. `--no-repeat-ngram-size 5` khắc phục được nhưng
  **mặc định tắt** để giữ khả năng tái lập.
- `val_loss` **không phát hiện được** sụp lặp: nó là teacher-forced, mỗi bước được
  mồi bằng tiền tố đúng, nên model chưa bao giờ ở chế độ sinh tự do nơi vòng lặp
  hình thành. Một run có `val_loss` tốt vẫn có thể sinh ra văn bản lặp.
- Split records hiện chưa mang prior linkage đầy đủ; temporal target policy mặc định vẫn là `keep`.
- `native_multiview` tồn tại trong Prompt v2, nhưng native manifest hiện chỉ luồn anchor image và để `auxiliary_views` rỗng; Stage 2 native multi-image chưa hoàn chỉnh end-to-end.
- METEOR, CIDEr và BERTScore phụ thuộc package tùy chọn; clinical metric adapters chưa được wire/validate.
- Stage 2 chưa hỗ trợ DDP; mỗi run dùng một GPU.
- Gradio `inference.py` vẫn là đường Vicuna legacy, chưa phải UI cho pipeline MedGemma mới.

## Acknowledgements và Citation

Repository kế thừa đáng kể từ META-CXR và bản fork LAVIS/BLIP-2, đồng thời sử dụng hoặc tích hợp MedGemma, MIMIC-CXR, BioViL-T, PubMedCLIP và các vision backbones khác. Hãy tuân thủ license, model terms và data-use agreement của từng upstream project.

META-CXR original work:

> D. Edirisinghe, W. Nimalsiri, M. Hennayake, D. Meedeniya and G. Lim, “Chest X-Ray Report Generation Using Abnormality Guided Vision Language Model,” *IEEE Access*, vol. 13, pp. 157651–157673, 2025. [doi:10.1109/ACCESS.2025.3606961](https://doi.org/10.1109/ACCESS.2025.3606961)

```bibtex
@article{edirisinghe2025metacxr,
  title     = {Chest X-Ray Report Generation Using Abnormality Guided Vision Language Model},
  author    = {Edirisinghe, D. and Nimalsiri, W. and Hennayake, M. and Meedeniya, D. and Lim, G.},
  journal   = {IEEE Access},
  volume    = {13},
  pages     = {157651--157673},
  year      = {2025},
  publisher = {IEEE},
  doi       = {10.1109/ACCESS.2025.3606961}
}
```

## License

Repository hiện không có file license riêng ở top level. Bản LAVIS vendored giữ BSD 3-Clause License tại [`model/lavis/LICENSE.txt`](model/lavis/LICENSE.txt). Điều này không tự động xác định license cho mọi phần còn lại của repository; cần kiểm tra điều khoản của từng upstream model/dataset trước khi sử dụng hoặc phân phối.
## Sửa contract cues (2026-09-08)

**Giảm nhiễu đầu vào bằng selective marginal cues (2026-09-09, opt-in).** CLI
`scripts/calibrate_cue_precision.py` fit trên validation, tối đa recall với
precision thực nghiệm >=0,70 và ít nhất 20 dự đoán. Nhãn không đạt sẽ abstain;
không ép điền P/N/U cho mọi bệnh và không dùng ngưỡng fallback.

Thử nghiệm CPU trên checkpoint Stage-1 hiện có (fit 1.808 val, kiểm tra 3.269
test), cùng cách tính **micro** và nhãn hiện diện trong báo cáo:

| Cues | Precision | Recall | Positive cues/ca |
|---|---:|---:|---:|
| Conditional cũ | 17,96% | 75,14% | 8,46 |
| Marginal P-fit cũ | 42,66% | 30,89% | 1,46 |
| Selective marginal | **68,00%** | 27,87% | **0,83** |

Chỉ Lung Opacity, Edema, Pleural Effusion và Support Devices đạt điều kiện fit;
9 nhãn còn lại không cung cấp cue, **không được hiểu là âm tính**. Ảnh vẫn là
đầu vào Stage 2 cho mọi finding. Sàn 70% trên val không bảo đảm 70% trên test
(CI95 precision test 66,27–69,89%). Đây là cải thiện độ chính xác cues theo nhãn
báo cáo, chưa chứng minh báo cáo sinh ra tốt hơn hay đúng lâm sàng hơn.

```bash
# Chạy trên training host, dùng validation predictions thật.
python scripts/calibrate_cue_precision.py \
  --predictions <private>/val_predictions_epoch_best.npz --split val \
  --precision-floor 0.70 --min-predicted 20 --output <private>/selective.json
# Training/generation cùng thêm:
# --cue-rule marginal_positive --threshold-path <private>/selective.json
# --prompt-config configs/experiment_native_qformer_guided.yaml
```

Nguyên nhân và phép thử tiếp theo: [handoff root-cause](docs/handoff/PLAN-2026-09-08-stage2-root-cause.md).

**Sửa token dừng chat.** MedGemma cấu hình stop IDs `[1, 106]`, nhưng code cũ
ghi đè thành tokenizer EOS `1`, làm mất `<end_of_turn>` (`106`) dù target
training kết thúc bằng token này. Constructor và generation nay giữ stop IDs
của model; chỉ fallback về tokenizer EOS nếu model không cấu hình. Summary ghi
lại stop IDs thực tế. Đây là lỗi dùng chung cho Arm A/C; cần đối chứng riêng
để đo ảnh hưởng đến lặp câu và không quy toàn bộ chênh lệch A/C cho nó.

Cả training và generation nhận `--cue-rule conditional_positive|mention_gated|marginal_positive|none`; training truyền cùng rule cho train/val/test và ghi vào summary/manifest. Mặc định vẫn là `conditional_positive` để giữ recipe cũ. Rule khác mặc định cần `--prompt-config` guided khớp visual mode; marginal dùng `--threshold-path configs/stage2_cue_thresholds_marginal_pfit.json`.

Ba trạng thái được tách rõ: `not_provided` (chủ động bỏ cues), `abstained` (không chọn được cue), `predicted` (có dự đoán P/N/U). Hai trạng thái đầu không sinh structured block hay câu normal. Âm tính cho một phần nhãn chỉ được nêu cụ thể; chỉ đủ 13 nhãn âm tính mới được tóm tắt normal. Ảnh, soft tokens và instruction được giữ nguyên. Cache cũ được bổ sung trạng thái khi đọc, không đổi embedding. Template hash thay đổi để nhận diện semantics mới.

Chạy rule khác trong output directory mới. Kiểm thử và audit cache validation nằm ở [handoff](docs/handoff/PLAN-2026-09-08-cue-contract.md); phép so generation sau sửa được ghi riêng trong [handoff root-cause](docs/handoff/PLAN-2026-09-08-stage2-root-cause.md).

Cache evaluation nay phân biệt stop IDs, cấu hình/template prompt và section mode,
để bản sửa không đọc lại báo cáo/điểm cũ. Cache Stage 1 vẫn dùng riêng như trước.

Đã kiểm thử CPU trên host sau các bản sửa: **1.013 passed, 2 skipped**, so với revision gốc
**974 passed, 2 skipped** — thêm 18 test contract, 17 test selective cues và 4 test token dừng/cache, không có failure mới. Ruff có
438 lỗi ở cả hai revision, không thêm lỗi. Audit cùng 1.415 ca validation:
`none` bỏ 1.415 câu normal không có cơ sở; marginal bỏ 654 câu (46,2%) khi
không có cue vượt ngưỡng. Đây là xác minh prompt, chưa phải cải thiện NLG/lâm sàng.
