> Source: `pretraining/`, `training/`, `scripts/`, `medgemma_inference/`, `preporcessing/`
> Status: ✅ ACTIVE
> Last verified against source: 2026-08-14

# Pipelines

Repository có **mười một** pipeline riêng biệt. Chúng không phải mười một cách chạy cùng
một thứ — mỗi cái có entrypoint, input, output và điều kiện tiên quyết khác nhau.

> GPU evidence hiện có chỉ bao phủ P10 inference-only encoder ablation. Nó không
> xác nhận P1/P2 training hay các pipeline GPU khác đã smoke/full validate.

| | Pipeline | Status | Cần Stage 1? | Cần GPU? |
|---|---|---|---|---|
| [P1](#p1--stage-1-pretraining) | Stage 1 pretraining | ✅ ACTIVE | — | ✅ |
| [P2](#p2--stage-2-medgemma-qlora) | Stage 2 MedGemma QLoRA | ✅ ACTIVE | tùy mode | ✅ |
| [P3](#p3--preprocessing--dựng-split) | Preprocessing / dựng split | ✅ ACTIVE | ❌ | ❌ |
| [P4](#p4--evaluation-stage-1) | Evaluation Stage 1 | ✅ ACTIVE | ❌ | ❌ |
| [P5](#p5--evaluation-stage-2) | Evaluation Stage 2 | ✅ ACTIVE | ❌ | ❌ |
| [P6](#p6--prompt-ablation-dry-run) | Prompt ablation (dry run) | 🧪 ABLATION | ❌ | ❌ |
| [P7](#p7--feature-precompute) | Feature precompute | 🟡 OPTIONAL | ❌ | ✅ |
| [P8](#p8--external-medgemma-inference-baseline) | External MedGemma inference | ✅ ACTIVE (baseline) | ❌ | ✅ |
| [P9](#p9--gradio-demo-vicuna-7b) | Gradio demo Vicuna-7B | ✅ ACTIVE (demo) | ✅ checkpoint | ✅ |
| [P10](#p10--stage-1-encoder-ablation-table-5) | Encoder ablation Table 5 | ✅ COMPLETE (4/4) | ✅ checkpoint | ✅ |
| [P11](#p11--evaluation-xai-grad-cam) | Evaluation XAI Grad-CAM | ✅ IMPLEMENTED, chưa GPU-run | ✅ checkpoint | ✅ |

---

## P1 — Stage 1 pretraining

**Mục đích:** học biểu diễn thị giác theo study và huấn luyện bộ phân loại 14×3,
đồng thời căn chỉnh ảnh↔text qua Q-Former.

### Entrypoint

> **Chạy ở đâu:** mọi lệnh dưới đây thực thi trên `phuong@minhphuong`, không phải
> ở checkout này (máy dev không có GPU và không có dataset). SSH vào máy đó,
> `cd ~/Documents/2026/KLTN/Code_github/META-CXR-full-smoke-git`, rồi
> `git pull origin main` **trước khi chạy**.

```bash
CUDA_VISIBLE_DEVICES=0 python -m pretraining.train \
    --cfg-path pretraining/configs/mimic_cxr_full.yaml \
    --options run.batch_size_train=6 run.batch_size_eval=6 run.accum_grad_iters=11
```

Chỉ còn một GPU nên không còn DDP; chạy plain, **không** qua `torch.distributed.run`.

### Config bắt buộc

`pretraining/configs/mimic_cxr_full.yaml` (production) — **10 epoch**,
`eval_start_epoch: 5` (epoch [0]–[4] train nhưng không chấm, cũng không thể được
tuyển làm best), early stop patience 5 (**bất động** với cấu hình này), bf16 AMP,
`save_freq: 5`, `warmup_steps: 300`.

Cộng `configs/env_config.yaml` cho đường dẫn máy. Thiếu file này →
`local_config.py` raise `FileNotFoundError`.

### Data

`MIMIC_CXR_Dataset` đọc ba CSV train/val/test. Sampling **một dòng cho một study**
(`model.data.study_sampling: true`), không phải một dòng cho một ảnh.

### Model

`Blip2Qformer` — xem [ARCHITECTURE.md §2](ARCHITECTURE.md#2-stage-1--chi-tiết-từng-khối).

### Checkpoint

| | |
|---|---|
| Ghi ra | `<output_dir>/<run_name>/checkpoint_best.pth`, `checkpoint_last.pth` |
| Chọn theo | `selection_metric: macro_auprc`, `selection_mode: max`, **chỉ trên validation** |
| Cadence | `save_freq: 5` → checkpoint theo epoch mỗi 5 epoch; `0` thì chỉ giữ best/last |
| Resume | `run.resume_ckpt_path` |

**Test split bị giữ hoàn toàn ngoài quá trình chọn checkpoint** và chỉ được đánh
giá một lần, từ `checkpoint_best`, sau khi train xong.

### Output phụ

Với `save_predictions: true`, logits validation được lưu ra `.npz` → cho phép
calibrate threshold offline (P4) mà không cần một GPU pass nữa.

### Smoke test

Đặt `run.truncate_train` / `truncate_val` / `truncate_test` trong YAML.

### Dependencies

`model/lavis/` (fork), `mhcac/`, `vision_encoders/`, `biovil_t/`, `local_config.py`,
`wandb`.

---

## P2 — Stage 2 MedGemma QLoRA

**Mục đích:** fine-tune MedGemma sinh báo cáo.

### Entrypoint

```bash
CUDA_VISIBLE_DEVICES=0 python training/run_medgemma_qlora.py [flags]
```

Smoke:
```bash
CUDA_VISIBLE_DEVICES=0 python training/run_medgemma_qlora.py \
    --train-limit 500 --val-limit 10 --test-limit 10 --no-upload \
    --output-dir training/outputs/smoke
```

### Năm kiến trúc, chọn bằng `--pipeline-mode`

| Mode | Stage 1 | Đường thị giác | Section hỗ trợ |
|---|---|---|---|
| `medgemma_direct` **(mặc định)** | ❌ | image tower + projector của MedGemma | findings_only, impression_only, findings_and_impression |
| `meta_cxr_qformer` | ✅ | Q-Former 32 soft token | **findings_only** |
| `meta_cxr_qformer_with_mhcac_prompt` | ✅ | soft token + text P/N/U | **findings_only** |
| `text_only_language_prior_ablation` | ❌ | **không có ảnh** | như native |
| `both_for_ablation` | — | chạy `medgemma_direct` rồi `meta_cxr_qformer` | — |

Thứ tự trong `both_for_ablation` **có chủ đích**: pipeline chính chạy trước, nên
một crash trong ablation vẫn để lại kết quả chính trên đĩa.

`text_only_language_prior_ablation` là mode **duy nhất** có
`requires_multimodal=False`. Nó là **sàn so sánh** — đo xem bao nhiêu phần báo cáo
tái tạo được chỉ từ language prior. Không bao giờ được báo cáo nó như "native
MedGemma" hay so sánh như thể nó đã nhìn ảnh.

### Alias cũ

`--image-mode {native,qformer,both}` vẫn hoạt động, map sang tên mới.
⚠ Launcher duy nhất còn dùng alias cũ này (`cloud/run_stage2.sh`) đã bị xóa 2026-08-13.

### Ví dụ đường Q-Former

```bash
CUDA_VISIBLE_DEVICES=0 python training/run_medgemma_qlora.py \
    --pipeline-mode meta_cxr_qformer --section-mode findings_only \
    --prompt-config configs/stage2_prompt_v2.yaml \
    --checkpoint-root pretraining/outputs --no-upload
```

### Ràng buộc

- **Single GPU.** Không DDP, không FSDP, không `device_map` rộng thay cho DDP.
  Muốn chạy hai experiment song song thì dùng hai job độc lập với
  `CUDA_VISIBLE_DEVICES=0` và `=1` — chúng **không** chia sẻ gradient.
- MedGemma là gated model → cần `HF_TOKEN` hoặc `huggingface-cli login`.
- Mode Q-Former + `--section-mode findings_and_impression` → **báo lỗi và dừng**,
  không âm thầm đổi target.

### Checkpoint

Chọn theo **validation cross-entropy**. Adapter LoRA + `trainer_state.pt` ghi ra
`--output-dir`. Test cohort được generate đúng một lần sau khi train.

### Dependencies

`training/dataio/`, `training/medgemma/`, `stage2/prompts/`, `training/stage2_utils.py`,
`training/train_eval_figure9_llm_variants_200.py`. **Chỉ khi mode cần Stage 1** mới
động tới `training/stage1/lavis_loader.py`.

---

## P3 — Preprocessing / dựng split

**Mục đích:** từ CSV thô + report `.txt` dựng ra ba split CSV mà training tiêu thụ.

```bash
python preporcessing/preprocess_mimic_cxr.py \
    --raw-dir <mimic-cxr-raw> \
    --reports-root <mimic-cxr-reports/files> \
    --output-dir <processed/full_allviews>
```

Cờ hữu ích: `--views frontal` (chỉ PA/AP), `--limit-studies N` (smoke).

**CPU-only, không cần ảnh.** Split phải patient- và study-disjoint.

Kiểm tra invariant của manifest sau khi dựng:
```bash
python -m training.dataio.validate_manifest --section-mode findings_and_impression
```

⚠ Manifest dựng trước 2026-07-21 thiếu cột `impression_clean` / `impression_valid` /
`impression_token_count` và **không phục vụ được** `--section-mode findings_and_impression`
(mặc định). `assert_columns` sẽ fail và nêu tên cột thiếu.

---

## P4 — Evaluation Stage 1

**Không cần GPU, không cần model, không cần dataset.** Mọi thứ tính lại từ `.npz`
đã lưu lúc inference. Đó là chủ ý: đổi threshold hoặc đổi uncertain policy không
được tốn một GPU-hour.

Hai bước, **theo đúng thứ tự này**:

```bash
# 1. Calibrate — CHỈ trên validation
python scripts/calibrate_thresholds.py \
    --predictions <val_predictions_epoch_best.npz> \
    --objective f1 --uncertain-policy ignore_uncertain --min-positive 20 \
    --output <f1_thresholds.json>

# 2. Score — trên test, dùng threshold đã calibrate
python scripts/evaluate_stage1.py \
    --predictions <test_predictions.npz> \
    --thresholds <f1_thresholds.json> \
    --output-dir <stage1_eval_dir>
```

`--min-positive 20`: bệnh lý có dưới 20 mẫu positive giữ nguyên threshold 0.5 —
calibrate trên quá ít mẫu chỉ là overfit vào nhiễu.

Chỉ số: precision/recall/F1 positive macro, per-pathology, AUROC, AUPRC, bootstrap
CI, confusion matrix 3 lớp, baseline all-negative.

Plot cần extra `eval-plots`.

---

## P5 — Evaluation Stage 2

```bash
python scripts/evaluate_stage2.py \
    --predictions <generated_reports.jsonl> \
    --metrics bleu,rouge,meteor,cider,bertscore \
    --skip-clinical-metrics --output-dir <stage2_eval_dir>
```

| Chỉ số | Trạng thái |
|---|---|
| BLEU, ROUGE-L | Tự implement trong repo, chỉ cần numpy |
| METEOR, CIDEr, BERTScore | Cần extra `eval-generation` |
| CheXbert, RadGraph, RadCliQ, RadFact | **Cố ý không cài được** — research code sau license riêng |

`training/evaluation/clinical.py` raise `MissingOptionalDependency` (nêu tên
package) hoặc `NotImplementedError` (package có nhưng adapter chưa validate với
điểm tham chiếu công bố).

**Chỉ số lâm sàng thiếu được báo là "unavailable", không bao giờ là 0.** Và
BLEU/ROUGE không được trình bày như độ chính xác lâm sàng.

Kèm theo: per-sample error analysis, subgroup analysis, cờ possible temporal
hallucination, bootstrap CI.

---

## P6 — Prompt ablation (dry run)

**Không load model, không sinh metric.** Nó render từng prompt variant cho mỗi
record và ghi metadata + aggregate ở mức prompt.

```bash
python scripts/run_prompt_ablation.py \
    --prompt-configs configs/prompt_ablation/P5_visual_primary.yaml \
    --max-samples 1000 --output-dir outputs/prompt_ablation
```

Chín cấu hình `configs/prompt_ablation/P1..P9.yaml`:

| | |
|---|---|
| P1 | legacy style |
| P2 | positive + uncertain, bỏ negative |
| P3 | positive + uncertain + negative quan trọng |
| P4 | thêm thông tin view |
| P5 | visual primary |
| P6 | qformer visual only |
| P7 | confidence bins |
| P8 | compact normal |
| P9 | full negative control |

Hai script cùng họ: `scripts/export_stage2_prompt_samples.py` (debug JSONL — ⚠
**có chứa findings text**, đặt `--output` ở nơi riêng tư),
`scripts/prompt_length_statistics.py`, `scripts/audit_temporal_targets.py`.

`scripts/_stage2_fixtures.py` cung cấp record tổng hợp **không phải MIMIC-CXR**,
để các script này chạy được khi không có dữ liệu. Số liệu từ fixture chỉ để minh
họa, không bao giờ là kết quả model.

---

## P7 — Feature precompute

**Mục đích:** tính trước đầu ra encoder đóng băng để Stage 1 khỏi chạy lại chúng
mỗi epoch.

```bash
python -m pretraining.precompute_features \
    --cfg-path pretraining/configs/mimic_cxr_full.yaml \
    --options model.encoders.biovil=true model.encoders.pubmedclip=true
```

Cache đi vào `run.feature_cache_dir`. `Blip2Qformer._encode_image_streams` sẽ bỏ
qua forward encoder khi thấy cache — **các projection có thể train vẫn chạy**, nên
training giống hệt.

⚠ Dựng cache với `model.data.study_sampling=false` để auxiliary view cũng có mặt.
Thiếu DICOM nào trong cache thì `_row_visual` raise `KeyError` nêu rõ tên.

⚠ Feature cache là dẫn xuất trực tiếp từ ảnh MIMIC → chịu **cùng lệnh cấm
redistribute**. `.gitignore` chặn `features/`, `feature_cache/`, `*.npy`, `*.npz`.

---

## P8 — External MedGemma inference (baseline)

**Mục đích:** sinh FINDINGS bằng checkpoint MedGemma do **bên thứ ba** fine-tune,
làm số đối chứng chính thức ([D-005](DECISIONS.md#d-005--track-inference-checkpoint-ngoài-là-baseline-chính-thức)).

```bash
python -m medgemma_inference.run_pretrained_findings \
    --config configs/experiments/pretrained_medgemma_findings_first.yaml \
    --split validation --max-samples 100
```

### Provenance — phải nêu ở mọi nơi báo cáo kết quả này

Checkpoint `erjui/medgemma-4b-srrg-findings` được fine-tune từ `google/medgemma-4b-it`
trên csrrg_ift (dẫn xuất từ MIMIC-CXR + CheXpert+) bởi **bên thứ ba**, không phải
project này, và **không** trên split của repository này.

### Đặc điểm

- **Inference only.** Không dựng optimizer, không tính gradient, không gọi
  `model.train()`. Enforce bằng `tests/test_inference_only_invariants.py`.
- **Không chạy được qua CLI fine-tuning.** `resolve_pipeline_modes` chủ động
  raise nếu ai đó thử — một run inference ngoại lai không được lọt vào đường
  fine-tuning.
- **Resume an toàn.** `prediction_writer.py` flush + fsync từng dòng; dòng dở
  dang bị cắt khi resume. `progress.py` kiểm tra run identity: nếu model, revision,
  generation setting, split hay dataset fingerprint đổi, nó **từ chối append** —
  vì trộn output của hai cấu hình vào một file kết quả thì không metric nào phát
  hiện ra được.
- **Budget theo wall-clock.** `runtime/budget.py`, mang `prior_elapsed_seconds`
  để resume không reset trần chi phí.
- **Lazy load.** Model chỉ được dựng khi còn việc chưa xong → run đã resume hoàn
  toàn tốn 0 GPU time. Guard Impression chạy **trước** mọi thứ, nên cấu hình sai
  fail trong mili-giây thay vì sau khi tải xong 4B weight.

### Phase 2 — bị vô hiệu hóa

`pretrained_medgemma_impression_phase2` **được khai báo nhưng bị runtime guard
chặn**. `model/pretrained_medgemma/impression_reporter.py` cố ý trơ: import nó
không được tải checkpoint, không dựng processor, không cấp VRAM, không import
transformers. Nó tồn tại để interface Phase 2 được chốt và review, không phải để
chạy.

---

## P9 — Gradio demo Vicuna-7B

**Status:** ✅ ACTIVE theo [D-002](DECISIONS.md#d-002--đường-vicuna-7b-legacy-vẫn-là-demo-active).
Đây là đường **legacy về mặt kiến trúc** (Vicuna, không phải MedGemma) nhưng vẫn
là đường demo đang dùng.

```bash
./build_container.sh && ./run_container.sh     # Docker, Gradio ở :7860
# hoặc trực tiếp:
bash inference.sh
```

`Dockerfile:5` → `ENTRYPOINT ["/bin/bash", "inference.sh"]`
`inference.sh:7` → `python3 inference.py --cfg-path pretraining/configs/blip2_pretrain_stage1_emb.yaml`

| | |
|---|---|
| LLM | Vicuna-7B + LoRA adapter từ `checkpoints/` |
| Config | `pretraining/configs/blip2_pretrain_stage1_emb.yaml` |
| Determinism | `SEED = 16` đặt cứng ở đầu `inference.py` |
| CheXpert labeler | Cần Java — `JAVA_HOME`/`JAVA_PATH` từ `local_config.py` |

⚠ `inference.py:312` hardcode `device_map={"": 0}` — đây là chỗ duy nhất còn ghim
GPU 0 trong repo.

⚠ Đường này **chưa được migrate sang MedGemma**. Nó không dùng
`stage2/prompts/PromptBuilder`; nó là một đường prompt riêng.

---

## P10 — Stage-1 encoder ablation Table 5

**Status:** ✅ COMPLETE — bốn cấu hình inference-only đã chạy trên full 3.216-study
test split.

```bash
python -m pretraining.train \
  --cfg-path pretraining/configs/ablation/all_three.yaml

python scripts/evaluate_stage1.py \
  --predictions <test_predictions.npz> \
  --thresholds configs/stage1_thresholds_f1_val.json \
  --output-dir <eval-dir>
```

Thay `all_three.yaml` bằng `biovil.yaml`, `pubmedclip.yaml` hoặc `swin.yaml`.
Các config không có train/validation split; `active_encoders` chỉ zero span sau
shared projection khi model ở `eval()`. Tên encoder lạ hoặc dùng ablation trong
training đều fail-closed.

Kết quả canonical: [`results/table5_encoder_ablation.json`](../results/table5_encoder_ablation.json.doc.md).
Threshold được calibrate duy nhất trên validation; cùng một file threshold được
dùng cho bốn dòng. Đây là evaluation của checkpoint đã tồn tại, **không phải**
bốn model encoder đơn được retrain.

---

## P11 — Evaluation XAI Grad-CAM

**Khác P4:** P4 classification cố ý model-free; P11 phải nạp checkpoint và giữ
autograd sống vì Grad-CAM lấy đạo hàm theo activation. `model.eval()` tắt dropout,
nhưng không có optimizer/backward update.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_explanation.py \
  --checkpoint <checkpoint_best.pth> \
  --cfg-path pretraining/configs/mimic_cxr_full.yaml \
  --split test --mask-cache-dir /mnt/drive1tb/datasets/explanation_masks \
  --ms-cxr-csv /mnt/drive1tb/datasets/ms-cxr/MS_CXR_Local_Alignment_v1.1.0.csv \
  --output-dir /mnt/drive1tb/private-results/xai \
  --save-cams --export-figures 12
```

Output metric tách `lung` anatomical prior và `bbox` expert annotation cho từng
encoder stream. Annotation coverage chỉ có ở bbox. Cột split MS-CXR không được
đọc; project manifest định tuyến val/test. PNG/NPZ là dữ liệu bệnh nhân và không
được đưa vào Git.

**Validation status:** metric/test/import/compile/lint chạy trên CPU dev; script
checkpoint/GPU **chưa từng chạy**.

---

## Phân biệt production / baseline / ablation / legacy

Để không nhầm khi viết báo cáo:

| Loại | Pipeline |
|---|---|
| **Production** | P1 (Stage 1), P2 với `medgemma_direct` |
| **Baseline đối chứng** | P8 (external checkpoint), `text_only_language_prior_ablation` |
| **Ablation** | P2 với `meta_cxr_qformer*`, P6 (prompt), P10 (encoder; complete) |
| **Hỗ trợ** | P3, P4, P5, P7, P11 |
| **Demo** | P9 |
| **Legacy đã chết** | notebooks 01/02/03 — xem [LEGACY_AND_OPTIONAL.md](LEGACY_AND_OPTIONAL.md) |

---

## Liên kết

- [ENTRYPOINTS.md](ENTRYPOINTS.md) — bảng tra lệnh đầy đủ
- [ARCHITECTURE.md](ARCHITECTURE.md) — các khối bên trong từng pipeline
- [DATA_FLOW.md](DATA_FLOW.md) — dữ liệu đi từ đâu tới đâu

← [Về HOME](../../HOME.md)
