# META-CXR

Repository nghiên cứu cho bài toán hiểu ảnh X-quang ngực và sinh báo cáo. Stage 1 học biểu diễn thị giác theo study, hợp nhất nhiều view, tạo Q-Former tokens và dự đoán bất thường. Stage 2 dùng MedGemma để sinh nội dung báo cáo, với đường ảnh native hoặc Q-Former soft tokens. Repository cũng có evaluator cho classification và report generation.

> **Trạng thái hiện tại:** ba giai đoạn explanation-aware đã hoàn tất ở mức
> implementation/CPU test; cache val nhỏ đã kiểm chứng trên dữ liệu thật; Table 5
> Stage-1 inference ablation hoàn tất. Explanation loss và evaluator XAI mới
> **chưa từng chạy trên GPU**; training pipelines vẫn cần GPU smoke/full validation.
>
> Đây là trạng thái theo tài liệu tích hợp tại commit hiện tại, không phải xác nhận đã train trên GPU hay đã tái lập metric mô hình.
>
> **Máy train:** một host duy nhất, `phuong@minhphuong` (máy cá nhân của tác giả).
> Xác minh 2026-08-13: **1× RTX 5060 Ti 16 GB**, dữ liệu và checkpoint nằm trên
> `/mnt/drive1tb` (930 GB NTFS, **không auto-mount** — phải mount tay sau mỗi lần
> reboot). Không còn đường chạy cloud: các recipe GCP/L4/Kaggle/2×3090 đã bị gỡ
> ngày 2026-08-13 để tối ưu chi phí.

## Trạng thái hiện tại

| Thành phần | Trạng thái |
|---|---|
| Branch integration | Các nhánh tính năng đã được tích hợp tuyến tính vào `main`; xem [integration audit](docs/final_branch_integration_audit.md) |
| Stage 1 implementation | Study-level/multi-view, Q-Former, MHCAC và explanation loss tùy chọn có trong code; Table 5 inference-only ablation đã chạy, training/explanation path chưa GPU-validated |
| Stage 2 implementation | MedGemma QLoRA, native-image và Q-Former routes có trong code; chưa GPU-validated |
| Explanation masks | Đã build CPU thử 200 study val: 193 hợp lệ (189 lung, 4 bbox); full cache chưa được xác nhận |
| XAI evaluation | Metric NumPy + checkpoint script + PNG/NPZ implemented; script checkpoint/GPU chưa từng chạy |
| CPU tests | Xem mục [Testing](#testing) cho output chạy thật của Phase 3 |
| GPU evidence | Table 5 encoder ablation 4/4 hoàn tất trên full test; chưa phải training smoke/full validation |
| Full MIMIC-CXR training | Chưa được xác nhận với pipeline final |
| Reproduced metrics | Chưa có metric mô hình mới được tái lập từ pipeline final |

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

Các config Stage 1 chính bật BioViL-T, PubMedCLIP và SwinV2; RadDINO có implementation nhưng đang tắt trong các recipe này. MHCAC dự đoán 14 nhãn theo Positive/Negative/Uncertain, còn Q-Former tạo 32 query tokens.

## Stage 1

Stage 1 nhận mẫu theo study. Với `multi_view: true`, view ưu tiên PA/AP/lateral được chọn làm anchor và tối đa một view phụ được fuse trước projection. Nhánh student dùng ảnh để tạo abnormality predictions và Q-Former representations; report text chỉ tham gia teacher branch trong lúc train.

- Entrypoint: [`pretraining/train.py`](pretraining/train.py)
- Config production (một GPU, recipe duy nhất): [`pretraining/configs/mimic_cxr_full.yaml`](pretraining/configs/mimic_cxr_full.yaml)
- Checkpoint selection: `macro_auprc` trên validation; test được giữ ngoài quá trình chọn checkpoint.

### Explanation-aware learning (tùy chọn)

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
Eq. (5). Loss chạy riêng trên BioViL 14×14 và PubMedCLIP/Swin 7×7; encoder vẫn
đóng băng, nên nó nắn cách projection/MHCAC/head đọc feature chứ không đổi
feature encoder.

`model.loss.lambda_explanation` là cờ bật/tắt duy nhất. Config production hiện
vẫn an toàn ở `0.0` và cache path rỗng. Sau khi build cache và smoke GPU, bật:

```yaml
model:
  loss:
    lambda_explanation: 0.25
  explanation:
    top_k: 0.5
    warmup_start_epoch: 2
    warmup_epochs: 2
    streams: [biovil, pubmedclip, swin]
    mask_cache_dir: /mnt/drive1tb/datasets/explanation_masks
```

Warmup: epoch [0]–[1] = 0; [2] = 0.125; [3] = 0.1875; [4]+ = 0.25. Đặt
`lambda_explanation: 0.0` để tắt hoàn toàn CAM capture/double backprop.

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

### Pipeline hiện tại

Repository chưa công bố metric mới được tái lập từ pipeline final. Chưa có full-data model result hoặc Prompt v2 GPU result được xác nhận cho commit hiện tại; không có bằng chứng pipeline final vượt META-CXR gốc.

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
