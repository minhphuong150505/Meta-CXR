# META-CXR / Meta-CXR-Kaggle

Repository nghiên cứu cho bài toán hiểu ảnh X-quang ngực và sinh báo cáo. Stage 1 học biểu diễn thị giác theo study, hợp nhất nhiều view, tạo Q-Former tokens và dự đoán bất thường. Stage 2 dùng MedGemma để sinh nội dung báo cáo, với đường ảnh native hoặc Q-Former soft tokens. Repository cũng có evaluator cho classification và report generation.

> **Trạng thái hiện tại:** CPU integration complete; ready for GPU smoke testing.
>
> Đây là trạng thái theo tài liệu tích hợp tại commit hiện tại, không phải xác nhận đã train trên GPU hay đã tái lập metric mô hình.

## Trạng thái hiện tại

| Thành phần | Trạng thái |
|---|---|
| Branch integration | Các nhánh tính năng đã được tích hợp tuyến tính vào `main`; xem [integration audit](docs/final_branch_integration_audit.md) |
| Stage 1 implementation | Study-level/multi-view, Q-Former, MHCAC và DDP có trong code; chưa GPU-validated |
| Stage 2 implementation | MedGemma QLoRA, native-image và Q-Former routes có trong code; chưa GPU-validated |
| CPU tests | 465 tests passed theo [integration notes](docs/final_merge_plan.md); không chạy lại trong lần cập nhật README này |
| GPU smoke test | **Not yet GPU-validated** |
| Full MIMIC-CXR training | Chưa được xác nhận với pipeline final |
| Reproduced metrics | Chưa có metric mô hình mới được tái lập từ pipeline final |

## Những thay đổi so với META-CXR gốc

Repository kế thừa công trình META-CXR nhưng code hiện tại đã bổ sung:

- sampling theo study và Stage 1 multi-view với anchor/auxiliary view;
- đường Stage 2 MedGemma native độc lập với Stage 1, bên cạnh Q-Former ablations;
- Prompt v2 có cấu hình, version và hash;
- evaluator cho classification, report generation, error analysis và counterfactual checks;
- workflow VM/preflight và config Stage 1 cho một GPU hoặc 2-GPU DDP.

Các thay đổi này chưa kèm bằng chứng rằng pipeline mới tốt hơn kết quả của bài báo gốc.

## Kiến trúc tổng quan

```text
Chest X-ray study
    -> Stage 1 visual encoders
    -> anchor/auxiliary multi-view fusion
    -> Q-Former representations + abnormality classification
    -> Stage 2 MedGemma (native image hoặc Q-Former soft tokens)
    -> FINDINGS/report output
    -> classification and generation evaluators
```

Các config Stage 1 chính bật BioViL-T, PubMedCLIP và SwinV2; RadDINO có implementation nhưng đang tắt trong các recipe này. MHCAC dự đoán 14 nhãn theo Positive/Negative/Uncertain, còn Q-Former tạo 32 query tokens.

## Stage 1

Stage 1 nhận mẫu theo study. Với `multi_view: true`, view ưu tiên PA/AP/lateral được chọn làm anchor và tối đa một view phụ được fuse trước projection. Nhánh student dùng ảnh để tạo abnormality predictions và Q-Former representations; report text chỉ tham gia teacher branch trong lúc train.

- Entrypoint: [`pretraining/train.py`](pretraining/train.py)
- Config một GPU: [`pretraining/configs/mimic_cxr_full_l4.yaml`](pretraining/configs/mimic_cxr_full_l4.yaml)
- Config 2× RTX 3090: [`pretraining/configs/mimic_cxr_2x3090.yaml`](pretraining/configs/mimic_cxr_2x3090.yaml)
- Checkpoint selection: `f1_positive_macro` trên validation; test được giữ ngoài quá trình chọn checkpoint.

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

### Stage 2

- BLEU và ROUGE-L được implement trong repository;
- METEOR, CIDEr và BERTScore dùng package tham chiếu tùy chọn;
- per-sample error analysis, subgroup analysis và cờ possible temporal hallucination;
- bootstrap intervals cho các per-sample metric khả dụng.

Clinical adapters hiện chỉ khai báo CheXbert, RadGraph và CheXpert labeler. Chúng cần dependency/checkpoint riêng và chưa được wire/validate để trả metric; evaluator báo `unavailable` hoặc `not implemented`, không thay bằng điểm 0. Không nên suy diễn lexical metrics thành độ đúng lâm sàng. Xem [evaluator validation](docs/evaluator_validation.md) và [evaluator audit](docs/evaluator_audit.md).

## Cấu trúc repository

```text
Meta-CXR-Kaggle/
├── configs/                 environment, experiment và prompt configs
├── pretraining/             Stage 1 entrypoint và configs
├── training/                Stage 2, data I/O và evaluation implementation
│   └── evaluation/
├── stage2/                  Prompt v2 builder và policies
├── model/                   LAVIS fork và model integrations
├── mhcac/                   abnormality classification và view fusion
├── vision_encoders/         visual encoder implementations
├── scripts/                 preflight, calibration và evaluator CLIs
├── tests/                   CPU test suite
├── cloud/                   VM/cloud automation
└── docs/                    hướng dẫn và audit chi tiết
```

Không có thư mục top-level `evaluation/`; evaluator hiện nằm tại `training/evaluation/`.

## Cài đặt

```bash
git clone https://github.com/minhphuong150505/Meta-CXR-Kaggle.git
cd Meta-CXR-Kaggle
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

`requirements-stage2.txt` bao gồm Stage 1 requirements rồi bổ sung Accelerate, bitsandbytes, PEFT và các package Stage 2. Xem [VM training guide](docs/VM_TRAINING_FINAL.md) trước khi cài trên GPU host. MedGemma là gated model; dùng `HF_TOKEN` hoặc đăng nhập Hugging Face và không ghi credential vào repository.

## Cấu hình

```bash
cp configs/env_config.yaml.example configs/env_config.yaml
```

Điền đường dẫn local trong `configs/env_config.yaml`; không commit file local, token hoặc credential. [`configs/env_config.yaml.example`](configs/env_config.yaml.example) mô tả:

- root chứa trực tiếp `files/` của MIMIC-CXR-JPG;
- train/val/test CSV trong `processed/full_allviews/`;
- output/checkpoint directories;
- private GCS và Weights & Biases settings nếu dùng.

`image_path` trong processed CSV là đường dẫn tương đối dạng `files/p1X/.../<dicom>.jpg` và được nối với `mimic_cxr_jpg_root`; không đổi nó thành đường dẫn tuyệt đối.

## Dữ liệu

MIMIC-CXR là dữ liệu hạn chế truy cập theo DUA. Người dùng phải tự có quyền truy cập hợp lệ; ảnh, report text, processed splits, credentials và model artifacts không được phân phối trong repository. Pipeline hiện nhắm tới full p10–p19 splits, không phải notebook p10 cũ. Cấu trúc mount chi tiết nằm trong [VM training guide](docs/VM_TRAINING_FINAL.md) và file config mẫu.

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
chạy tối đa 20 epoch, early stopping patience 5 và chọn checkpoint theo
macro-AUPRC; logits validation được lưu để calibrate threshold F1 sau đó.

```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --standalone --nproc_per_node=1 \
  -m pretraining.train --cfg-path pretraining/configs/mimic_cxr_full_l4.yaml
```

Sau khi train, calibrate threshold chỉ trên prediction của validation từ
`checkpoint_best` (các bệnh có dưới 20 positive giữ threshold 0.5):

```bash
python scripts/calibrate_thresholds.py \
  --predictions pretraining/outputs/<run>/result/val_predictions_epoch_best.npz \
  --objective f1 --uncertain-policy ignore_uncertain --min-positive 20 \
  --output pretraining/outputs/<run>/result/f1_thresholds.json
```

Full 2-GPU command được cấu hình như sau, nhưng vẫn cần GPU smoke test trước:

```bash
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  -m pretraining.train --cfg-path pretraining/configs/mimic_cxr_2x3090.yaml
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

python scripts/evaluate_stage2.py \
  --predictions <generated_reports.jsonl> \
  --metrics bleu,rouge,meteor,cider,bertscore \
  --skip-clinical-metrics --output-dir <stage2_eval_dir>
```

Xem [VM training guide](docs/VM_TRAINING_FINAL.md) để biết resume, output layout và troubleshooting.

## Hỗ trợ nhiều GPU

| Thành phần | 1 GPU | 2-GPU DDP | Hai job độc lập |
|---|---|---|---|
| Stage 1 | Có trong code | Có trong code/config qua `torchrun`; chưa GPU-tested | Không phải workflow chính |
| Stage 2 | Một GPU cho mỗi run | **Không hỗ trợ** | Có thể chạy hai experiment riêng bằng `CUDA_VISIBLE_DEVICES=0` và `CUDA_VISIBLE_DEVICES=1` |

Stage 2 không dùng một `device_map` rộng để thay cho DDP. Hai job độc lập không chia sẻ gradient và không phải một distributed run.

## Testing

CPU checks được integration notes sử dụng:

```bash
CUDA_VISIBLE_DEVICES="" python -m pytest tests/ -q
CUDA_VISIBLE_DEVICES="" python -m compileall -q \
  stage2 training scripts runtime safety tests medgemma_inference
```

Con số 465 passed ở bảng trạng thái là baseline được ghi trong integration notes, không phải kết quả vừa chạy lại trong lần chỉnh tài liệu này. Test CPU không thay thế Stage 1/Stage 2 smoke test trên VM GPU.

## Kết quả và cảnh báo metric

### Pipeline hiện tại

Repository chưa công bố metric mới được tái lập từ pipeline final. Chưa có full-data model result hoặc Prompt v2 GPU result được xác nhận cho commit hiện tại; không có bằng chứng pipeline final vượt META-CXR gốc.

### Original paper reference results

Bài báo META-CXR gốc có báo cáo classification và report-generation metrics cho kiến trúc/dữ liệu của công trình đó. Các số trong paper chỉ là tham khảo lịch sử, **không phải kết quả của repository/commit hiện tại**. README này không sao chép bảng số để tránh trộn nguồn; xem bài báo được dẫn trong mục Citation.

## Tài liệu

- [VM training guide](docs/VM_TRAINING_FINAL.md)
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
- [Checkpoint workflow](docs/CHECKPOINT_WORKFLOW.md)
- [Feature cache](docs/FEATURE_CACHE.md)
- [Notebook privacy](docs/notebook_privacy.md)

## Hạn chế hiện tại

- Stage 1 và Stage 2 chưa được GPU smoke-tested trong lần tích hợp hiện tại.
- Full MIMIC-CXR training và metric pipeline final chưa được tái lập.
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
