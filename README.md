# META-CXR

Repository nghiên cứu cho bài toán hiểu ảnh X-quang ngực và sinh báo cáo. Stage 1 học biểu diễn thị giác theo study, hợp nhất nhiều view, tạo Q-Former tokens và dự đoán bất thường. Stage 2 dùng MedGemma để sinh nội dung báo cáo, với đường ảnh native hoặc Q-Former soft tokens. Repository cũng có evaluator cho classification và report generation.

> **Trạng thái hiện tại:** Stage-1 đã chạy đủ trên GPU. Explanation loss đã được
> A/B có kiểm soát 5 epoch (bật/tắt) và evaluator XAI đã chạy trên cả hai
> checkpoint — kết quả: **không có tác dụng đo được, và CAM ở mức ngẫu nhiên ở cả
> hai nhánh**, nên hướng này đã **tắt trong production từ 2026-08-17** (chi tiết ở
> [Explanation-aware learning](#explanation-aware-learning-đã-tắt-trong-production-từ-2026-08-17)).
> Table 5 Stage-1 inference ablation hoàn tất. Stage 2 vẫn cần GPU validation.
>
> Stage-1 đã có bằng chứng GPU thật (xem bảng dưới). Stage-2 thì chưa — phần
> trạng thái của Stage 2 vẫn chỉ là tình trạng tích hợp code tại commit hiện tại,
> không phải xác nhận đã train trên GPU hay đã tái lập metric mô hình.
>
> **Máy train:** một host duy nhất, `phuong@phuong-b760m-pro-rs-d4-wifi` (máy cá
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
| Stage 2 implementation | MedGemma QLoRA, native-image và Q-Former routes có trong code; chưa GPU-validated |
| Explanation masks | Full cache đã build và kiểm chứng (`explanation_masks_v2`, có `masks_bbox_*`). Không còn được training tiêu thụ; giữ cho evaluator |
| XAI evaluation | Đã chạy trên GPU với 2 checkpoint (test split). **Cảnh báo: saliency precision ở mức ngẫu nhiên** — luôn kèm baseline diện tích mask |
| CPU tests | Xem mục [Testing](#testing) cho output chạy thật của Phase 3 |
| GPU evidence | Stage-1 full run xong; A/B explanation loss bật/tắt 5 epoch xong (2026-08-16/17) kèm calibration + eval test + XAI cả hai nhánh |
| Checkpoint cũ | **Đã xoá toàn bộ 2026-08-14** (15 file, 39 GB) — các run đó đi sai hướng và không nạp được vào recipe hiện tại (Swin tắt → 98 token thay vì 147). Số liệu Table 5 còn trong `results/` nhưng không tái lập được |
| Full MIMIC-CXR training | ✅ **Đã chạy xong 2026-08-20** — `run_20260819_xmpoff`, 10/10 epoch, `rc=0`, 12h35m, 0.3505 s/it, 0 kernel fault. Best epoch 6 |
| Reproduced metrics | ✅ Stage 1 đã có kết quả test split (3,269 study), chấm một lần từ `checkpoint_best`, ngưỡng calibrate chỉ trên val — xem [Kết quả và cảnh báo metric](#kết-quả-và-cảnh-báo-metric). Stage 2 vẫn chưa có |
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

### Stage 1 — test split, `run_20260819_xmpoff` (2026-08-20)

Run 10 epoch trên toàn bộ MIMIC-CXR, `checkpoint_best` chọn ở epoch 6 theo `val_loss`.
Test split (3,269 study) được giữ kín và chấm **đúng một lần**; ngưỡng calibrate **chỉ
trên validation**.

Số chính, dưới framing `study_presence` + `marginal_presence` (xem mục Evaluation):

| Metric | Giá trị | 95% CI |
|---|---:|:---:|
| `macro_auroc` | **0.7441** | [0.7341, 0.7540] |
| `positive_macro_f1` | **0.3224** | [0.3109, 0.3326] |
| `macro_auprc` | 0.3004 | [0.2919, 0.3139] |
| `macro_specificity` | 0.8214 | – |
| `micro_auroc` | 0.8183 | – |

So với baseline tầm thường trên **cùng** framing: `all_positive` 0.2280,
`all_negative` 0.0000, `majority_class` 0.0000, `threshold_half` 0.3173.

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
`Test/stage1_test/README.md` — thư mục đó **git-ignored** vì file `.npz` chứa định danh
study của MIMIC-CXR.

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

Từ 2026-08-19, **Codex không còn được dùng** (hết hạn gia hạn). Vai trò không đổi,
chỉ đổi người thực thi — Claude Code đã được cài và xác thực sẵn trên máy train
(`~/.local/bin/claude`).

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
- Khi **Opus hết usage**, chuyển việc cho agent **Sonnet 5** thay vì chờ — Sonnet đủ
  để thực thi kế hoạch đã viết và triage log, rẻ hơn nhiều. Giữ Opus cho quyết định
  kiến trúc/recipe.
- ⚠ **Báo cáo của agent là bằng chứng, không phải sự thật** — đối chiếu với file
  thật trên đĩa (kể cả mtime) trước khi hành động theo nó.

Luật không đổi: mọi thay đổi hành vi phải kèm cập nhật tài liệu trong cùng commit,
và tuyệt đối không đưa dữ liệu bệnh nhân vào commit/handoff/tóm tắt. Chi tiết cho
agent nằm ở [AGENTS.md](AGENTS.md) và `CLAUDE.md`.

## Hạn chế hiện tại

- Stage 1 và Stage 2 **training** chưa được GPU smoke-tested trong lần tích hợp
  hiện tại. Riêng Table 5 Stage-1 inference-only encoder ablation đã hoàn tất 4/4
  trên full test split; xem `results/table5_encoder_ablation.*`.
- Explanation loss chưa từng chạy smoke/full training trên GPU;
  `scripts/evaluate_explanation.py` chưa từng nạp checkpoint/dataset hay chạy
  end-to-end. Không dùng metric/heatmap từ đường này trong luận văn trước smoke.
- Cache explanation mới chỉ được build/kiểm tra ở smoke val 200 study, chưa xác
  nhận full train/val/test cache.
- Full training pipeline và Stage 2 metric chưa được tái lập từ pipeline final.
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
