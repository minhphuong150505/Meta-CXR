# Meta-CXR — Source Code Guide

> Điểm bắt đầu **duy nhất**. Mọi thứ khác đều tới được từ đây.
> Last verified against source: 2026-08-12 · source through commit `6d5c463` · branch `main`

---

## Start Here

Bạn đang mở knowledge base kỹ thuật của **Meta-CXR** — framework vision-language
sinh báo cáo X-quang ngực tự động.

`struct/` là bản đồ **song song** với source code, không phải bản sao của nó.
Nó trả lời những câu hỏi mà đọc code không trả lời được: *tại sao* component này
tồn tại, *ai* gọi nó, *cái gì* vỡ nếu bạn sửa nó, và *cái gì* trong repo đã chết.

> **`struct/` được track trong Git** theo
> [D-012](project/_meta/DECISIONS.md#d-012--đưa-struct-vào-repository).
> Mọi thay đổi source có ảnh hưởng hành vi phải cập nhật knowledge base này trong
> cùng commit.

### Máy train hiện tại

Theo xác nhận của user ngày 2026-08-12, project hiện dùng **một máy train** với
định danh đăng nhập `phuong@minhphuong` ([D-011](project/_meta/DECISIONS.md#d-011--máy-train-hiện-tại)).
Các tài liệu GCP L4 và VM 2× RTX 3090 là recipe tham khảo/khả năng được hỗ trợ,
không phải mô tả phần cứng hiện tại. Chưa có xác nhận về GPU, VRAM, RAM hay data
mount của `minhphuong`; luôn chạy `scripts/vm_preflight.py` trước một run dài.

---

## What Meta-CXR Does

**Nhận ảnh X-quang ngực → sinh phần FINDINGS của báo cáo y khoa.**

Điểm khác biệt: nó **phân loại bất thường trước**, rồi dùng kết quả đó dẫn đường
cho mô hình ngôn ngữ — thay vì ném thẳng ảnh vào LLM. Đây là ý tưởng
"abnormality-guided".

| | |
|---|---|
| **Input** | Một *study* MIMIC-CXR: 1 anchor view + tối đa 1 auxiliary view, ảnh 448×448 |
| **Output** | 14 bệnh lý × 3 lớp {Positive, Negative, Uncertain} · và text báo cáo |
| **Dữ liệu** | MIMIC-CXR-JPG p10–p19 — ⚠ PhysioNet credentialed, DUA cấm redistribute |

> ⚠ **Phạm vi GPU evidence hiện có rất hẹp.** Stage-1 inference-only encoder
> ablation Table 5 đã chạy đủ 4/4 trên full test split và có artifact tracked.
> Điều đó **không** chứng minh Stage-1/Stage-2 training pipeline hiện tại đã được
> GPU smoke/full-train validate, cũng không phải metric Stage 2 tái lập.

---

## High-Level Pipeline

```text
Chest X-ray study (448×448)
        │
        ▼
┌─ STAGE 1 ────────────────────────── pretraining/train.py ─────────────┐
│  3 vision encoder ĐÓNG BĂNG   BioViL-T 1408 · PubMedCLIP 768 · SwinV2 │
│         ▼                                                              │
│  View fusion (per-encoder, trên output THÔ)                            │
│         ▼                                                              │
│  SharedVisualTokenProjector → 1408, nối theo trục token                │
│         ├──────────────► MHCAC  → logits [B,14,3]                      │
│         │                  student = ảnh  |  teacher = ảnh+text (train) │
│         └──────────────► Q-Former 32 query → ITC + ITM + LM            │
└────────────────────────────────────────────────────────────────────────┘
        │  checkpoint_best.pth
        ▼   (mode mặc định của Stage 2 BỎ QUA hoàn toàn nhánh này)
┌─ STAGE 2 ──────────────── training/run_medgemma_qlora.py ─────────────┐
│  MedGemma 1.5 4B-it · QLoRA NF4 · single GPU                          │
│  medgemma_direct (mặc định) | meta_cxr_qformer | +mhcac_prompt | …    │
└────────────────────────────────────────────────────────────────────────┘
        ▼
   FINDINGS (± IMPRESSION)  →  evaluate_stage1.py / evaluate_stage2.py
```

**Hai Stage cố ý tách rời.** Đây là ràng buộc thiết kế nặng nhất repository, và
nó được enforce bằng test (`tests/test_native_independence.py`), không phải bằng
quy ước.

---

## How to Read This Documentation

### Nếu bạn hoàn toàn mới — đọc theo đúng thứ tự này

1. [**Project Overview**](project/_meta/PROJECT_OVERVIEW.md) — bài toán, dữ liệu, ràng buộc
2. [**Architecture**](project/_meta/ARCHITECTURE.md) — các khối và cách chúng nối
3. [**Pipelines**](project/_meta/PIPELINES.md) — 10 pipeline, chạy thế nào
4. [**Data Flow**](project/_meta/DATA_FLOW.md) — CSV → tensor → output, kèm shape
5. [**Entrypoints**](project/_meta/ENTRYPOINTS.md) — gõ lệnh gì
6. [**Source Tree**](#source-code-tree) — bắt đầu click xuống code

### Nếu bạn sắp sửa code

1. [**Active vs Legacy**](project/_meta/ACTIVE_COMPONENTS.md) — thứ bạn định sửa có đang chạy không?
2. [**Call Graph**](project/_meta/CALL_GRAPH.md) — ai gọi nó, nó gọi ai
3. `_index.md` của thư mục → `.doc.md` của file → `.methods/` của hàm
4. Mục **Modification risk** trong trang method — cái gì vỡ nếu bạn đổi

### Nếu bạn không hiểu một thuật ngữ

→ [**Glossary**](project/_meta/GLOSSARY.md)

### Nếu bạn thắc mắc "tại sao lại thế này"

→ [**Decisions**](project/_meta/DECISIONS.md) — bộ nhớ bền vững của project

---

## Meta Documentation

| Trang | Trả lời câu hỏi |
|---|---|
| [Project Overview](project/_meta/PROJECT_OVERVIEW.md) | Project này giải quyết gì? |
| [Architecture](project/_meta/ARCHITECTURE.md) | Các khối ghép với nhau ra sao? |
| [Pipelines](project/_meta/PIPELINES.md) | Có những đường chạy nào? |
| [Data Flow](project/_meta/DATA_FLOW.md) | Dữ liệu biến đổi thế nào? Shape gì? |
| [Call Graph](project/_meta/CALL_GRAPH.md) | Hàm nào gọi hàm nào? Ai gọi tôi? |
| [Entrypoints](project/_meta/ENTRYPOINTS.md) | Gõ lệnh gì để chạy? |
| [Active Components](project/_meta/ACTIVE_COMPONENTS.md) | Cái gì đang thực sự chạy? |
| [Legacy & Optional](project/_meta/LEGACY_AND_OPTIONAL.md) | Cái gì đã chết? Cái gì chỉ bật theo config? |
| [Decisions](project/_meta/DECISIONS.md) | Tại sao lại quyết định như vậy? |
| [Glossary](project/_meta/GLOSSARY.md) | MHCAC là gì? Soft token là gì? |

---

## Source Code Tree

Nhãn: `✅` active · `🟡` conditional · `🧪` ablation/experimental · `🧰` utility ·
`❓` unknown · `🕰` legacy · `⚠` cần chú ý

```text
Meta-CXR-source/
│
├── pretraining/                    ✅ Stage 1
│   ├── train.py                    ✅ ★ ENTRYPOINT Stage 1
│   ├── precompute_features.py      🟡 feature cache
│   ├── configs/
│   │   ├── mimic_cxr_full_l4.yaml       ✅ ★ PRODUCTION
│   │   ├── mimic_cxr_2x3090.yaml        ✅ 2-GPU DDP
│   │   ├── blip2_pretrain_stage1_emb.yaml ✅ demo Gradio
│   │   ├── blip2_pretrain_stage1.yaml   🕰
│   │   ├── mimic_cxr_2gpu.yaml          🕰 ⚠ warmup_steps 32000
│   │   ├── ablation/                     ✅ Table 5 inference-only (4/4 complete)
│   │   └── encoder_comparison/
│   │       └── 07_all_three.yaml        🧪
│   └── outputs/                    (generated — chỉ .gitkeep)
│
├── model/
│   ├── lavis/                      ✅ fork LAVIS đã sửa (24 file, 10.8k LOC)
│   │   ├── models/blip2_models/
│   │   │   ├── blip2_qformer.py         ✅ ★ TRUNG TÂM Stage 1 (1497)
│   │   │   ├── Qformer.py               ✅ (1221)
│   │   │   ├── blip2.py                 ✅
│   │   │   └── modeling_llama_imgemb.py ✅ (975, đường Vicuna)
│   │   ├── data/
│   │   │   ├── ReportDataset.py         ✅ ★ MIMIC_CXR_Dataset (1130)
│   │   │   └── mimic_cxr_utils.py       ✅ build_study_index
│   │   ├── runners/runner_base.py       ✅ ★ train loop (1182)
│   │   ├── tasks/image_text_pretrain.py ✅ eval hook
│   │   ├── common/                      ✅ config, registry, optims, dist_utils…
│   │   ├── datasets/  processors/       ✅
│   │   └── blip_models/blip_outputs.py  ✅
│   └── pretrained_medgemma/        ✅ checkpoint NGOÀI (baseline)
│       ├── findings_loader.py           ✅
│       ├── findings_reporter.py         ✅
│       ├── impression_reporter.py       ⛔ DISABLED có chủ đích
│       ├── output_schema.py             ✅
│       └── errors.py                    ✅
│
├── mhcac/                          ✅ phân loại bất thường
│   ├── mhcac_12.py                 ✅ ★ bản DUY NHẤT được wire (448)
│   ├── loss.py                     ✅ ★ toàn bộ 11 loss (628)
│   ├── view_fusion.py              🟡 multi_view: true
│   ├── utils.py                    🕰 (624 — chỉ notebook legacy gọi)
│   ├── aggregator.py               🕰 ⚠ chỉ còn string trong freeze-list
│   └── mhcac.py, mhcac_2..11.py    🕰 11 variant, zero reference
│
├── vision_encoders/
│   ├── shared_visual_tokens.py     ✅ ★ điểm chiếu DUY NHẤT
│   ├── pubmedclip/pubmed_clip.py   ✅
│   ├── swin/swin_encoder.py        ✅
│   ├── rad_dino/rad_dino_encoder.py 🟡 raddino: false ở mọi config
│   ├── biovil_t/                   🕰 ⚠ BẢN SAO — bản dùng thật là biovil_t/ ở root
│   └── medclip/medclip.py          🕰 import bị comment
│
├── biovil_t/                       ✅ BioViL-T (bản ĐƯỢC import)
│   ├── model.py  encoder.py  modules.py  resnet.py
│   └── transformer.py  pretrained.py  types.py  device.py
│
├── training/                       ✅ Stage 2 + dataio + evaluation
│   ├── run_medgemma_qlora.py       ✅ ★ ENTRYPOINT Stage 2 (527)
│   ├── train_eval_figure9_llm_variants_200.py ✅ ★ ĐỘNG CƠ Stage 2 (1746)
│   │                                  ⚠ tên file gây hiểu nhầm
│   ├── pipeline_modes.py           ✅ stdlib-only, resolve kiến trúc
│   ├── stage2_utils.py             ✅
│   ├── run_context.py              ✅   torch_io.py  ✅
│   ├── dataio/
│   │   ├── manifest.py             ✅ CHỈ pandas — giữ ranh giới Stage 1/2
│   │   └── validate_manifest.py    🧰 CLI kiểm tra leakage
│   ├── stage1/
│   │   └── lavis_loader.py         🟡 ★ CỬA DUY NHẤT sang LAVIS/Stage-1
│   ├── medgemma/
│   │   ├── soft_tokens.py          🟡 ⚠ chỗ dễ sai nhất repo
│   │   └── capabilities.py         ✅
│   ├── trainer/                    ❓ D-001 — chỉ test import
│   │   ├── checkpointing.py  state.py
│   └── evaluation/                 ✅ 17 module
│       ├── classification_metrics.py ✅ (662)   schemas.py ✅
│       ├── threshold_calibration.py  ✅         uncertain_policy.py ✅
│       ├── generation_metrics.py     ✅         error_analysis.py ✅
│       ├── baselines.py  bootstrap.py  subgroup_analysis.py  ✅
│       ├── report_writer.py  clinical.py        ✅
│       ├── visualization.py          🟡 cần extra eval-plots
│       ├── config.py                 ❓ D-001
│       └── counterfactual.py  perturbations.py  ❓ D-001
│
├── stage2/prompts/                 ✅ Prompt v2, stdlib-only
│   ├── builder.py                  ✅ ★ PromptBuilder — điểm vào DUY NHẤT
│   ├── schemas.py  policies.py  templates.py
│   └── ontology.py  records.py  validation.py
│
├── medgemma_inference/             ✅ baseline checkpoint NGOÀI
│   ├── run_pretrained_findings.py  ✅ ★ ENTRYPOINT
│   ├── runner.py  config.py
│   └── prediction_writer.py  progress.py
│
├── runtime/                        ✅ stdlib-only
│   ├── budget.py                   ✅ tính tiền theo wall-clock
│   └── device.py                   ✅ không hardcode cuda:0
│
├── safety/                         phần lớn ❓ D-001
│   ├── claims.py                   ✅ CÓ caller thật (error_analysis.py)
│   └── pipeline.py  verifiers.py  reconciler.py   ❓ chỉ test
│
├── scripts/                        🧰 CLI
│   ├── vm_preflight.py             🧰 chạy TRƯỚC mọi run dài
│   ├── calibrate_thresholds.py     ✅   evaluate_stage1.py ✅
│   ├── evaluate_stage2.py          ✅   check_notebook_privacy.py ✅ pre-commit
│   ├── run_prompt_ablation.py      🧪   export_stage2_prompt_samples.py 🧪 ⚠ chứa findings
│   ├── prompt_length_statistics.py 🧪   audit_temporal_targets.py 🧪
│   └── _stage2_fixtures.py         🧪 dữ liệu tổng hợp, KHÔNG phải MIMIC
│
├── preporcessing/                  ✅ (sic — tên thư mục sai chính tả, giữ nguyên)
│   ├── preprocess_mimic_cxr.py     ✅ ★ dựng split CSV
│   └── mimic_report_parser.py      ✅ trích FINDINGS/IMPRESSION
│
├── configs/
│   ├── env_config.yaml.example     ✅ (env_config.yaml git-ignored)
│   ├── stage2_prompt_v2.yaml       ✅ opt-in Prompt v2
│   ├── stage1_thresholds_f1_val.json ✅ threshold validation cho Table 5
│   ├── experiments/pretrained_medgemma_findings_first.yaml ✅
│   ├── prompt_ablation/P1..P9.yaml 🧪
│   └── kaggle_datasets.yaml        🕰 giờ chủ yếu là file chính sách
│
├── cloud/                          🧰 launcher GCP
│   ├── run_stage1.sh  run_stage2.sh ✅
│   ├── env.sh                      ✅ ⚠ identity đến từ cloud/env.local.sh (untracked)
│   ├── setup_vm.sh  push_from_local.sh  🧰
│   ├── run_encoder_comparison.sh   🕰 compatibility alias → run_stage1
│   ├── run_medgemma_*.sh           🕰 3 compatibility alias → run_stage2
│   ├── run_paper_assets.sh         ⚠ broken: thiếu paper_assets.py
│   └── lib/common.sh               🧰 require_private_bucket = chốt chặn dữ liệu
│
├── results/                        ✅ Table 5 encoder ablation (4/4 complete)
│   └── table5_encoder_ablation.{md,json,csv}
│
├── tests/                          ✅ 31 file — enforce invariant kiến trúc
│   └── fixtures/notebooks/*.fixture
│
├── utils/                          ⚠ POTENTIALLY_UNUSED — zero import toàn repo
│   ├── prompter.py  callbacks.py  datacollator.py
│   └── split_emb.py                ⚠ chạy ngay khi import, path không tồn tại
│
├── notebooks/                      🕰 cả 3 tự gắn nhãn "DO NOT RUN"
│   ├── 01_generate_mimic_cxr_cleaned_csv.ipynb
│   ├── 02_preprocess_mimic_cxr_p10_splits.ipynb
│   └── 03_train_meta_cxr_2xT4_kaggle.ipynb
│
├── docs/                           ⚠ ~15/27 file là BIÊN BẢN, không phải spec
│   ├── VM_TRAINING_FINAL.md        ✅ nên theo
│   ├── STAGE2_PIPELINE_MODES.md    ✅ nên theo
│   ├── SETUP_GUIDE.md  CHECKPOINT_WORKFLOW.md  FEATURE_CACHE.md  ✅
│   ├── notebook_privacy.md         ✅   cloud/  ✅
│   └── *_audit.md  *_baseline.md  final_*.md   🕰 chỉ đọc như lịch sử
│
├── checkpoints/                    (generated — chỉ .gitkeep; LoRA Vicuna ở đây)
│
├── inference.py                    ✅ demo Gradio Vicuna-7B (670) ⚠ device_map ghim GPU 0
├── inference.sh                    ✅   Dockerfile ✅ ENTRYPOINT → inference.sh
├── build_container.sh              ✅   run_container.sh ✅
├── local_config.py                 ✅ ★ nạp configs/env_config.yaml
├── threshold.json                  ⚠ artifact lịch sử, không được load ngầm
├── pyproject.toml                  🧰 CỐ Ý zero runtime dependency
├── requirements-stage1.txt         ✅   requirements-stage2.txt ✅ (additive, bao gồm stage1)
├── requirements.txt                ✅ alias mặc định → requirements-stage1.txt
├── .pre-commit-config.yaml         ✅ notebook privacy guard
├── .gitignore                      ✅ chặn dữ liệu/artifact nhạy cảm; không chặn struct/
└── README.md                       ✅ tiếng Việt, chi tiết, authoritative
```

---

## Navigation — click xuống code

### `pretraining/` — Stage 1
[📁 Directory documentation](project/pretraining/_index.md)

- [`train.py`](project/pretraining/train.py.doc.md) ★ entrypoint
- [`precompute_features.py`](project/pretraining/precompute_features.py.doc.md)

#### `pretraining/configs/`
[📁 Directory documentation](project/pretraining/configs/_index.md)

### `model/` — LAVIS fork + checkpoint ngoài
[📁 Directory documentation](project/model/_index.md)

#### `model/lavis/`
[📁 Directory documentation](project/model/lavis/_index.md)

- [`models/blip2_models/blip2_qformer.py`](project/model/lavis/models/blip2_models/blip2_qformer.py.doc.md) ★
- [`data/ReportDataset.py`](project/model/lavis/data/ReportDataset.py.doc.md) ★
- [`runners/runner_base.py`](project/model/lavis/runners/runner_base.py.doc.md) ★
- [`tasks/image_text_pretrain.py`](project/model/lavis/tasks/image_text_pretrain.py.doc.md)
- [`models/blip2_models/Qformer.py`](project/model/lavis/models/blip2_models/Qformer.py.doc.md)
- [`models/blip2_models/modeling_llama_imgemb.py`](project/model/lavis/models/blip2_models/modeling_llama_imgemb.py.doc.md)

#### `model/pretrained_medgemma/`
[📁 Directory documentation](project/model/pretrained_medgemma/_index.md)

### `mhcac/` — phân loại bất thường
[📁 Directory documentation](project/mhcac/_index.md)

- [`mhcac_12.py`](project/mhcac/mhcac_12.py.doc.md) ★
- [`loss.py`](project/mhcac/loss.py.doc.md) ★
- [`view_fusion.py`](project/mhcac/view_fusion.py.doc.md)

### `vision_encoders/`
[📁 Directory documentation](project/vision_encoders/_index.md)

### `biovil_t/`
[📁 Directory documentation](project/biovil_t/_index.md)

### `training/` — Stage 2 + evaluation
[📁 Directory documentation](project/training/_index.md)

- [`run_medgemma_qlora.py`](project/training/run_medgemma_qlora.py.doc.md) ★
- [`train_eval_figure9_llm_variants_200.py`](project/training/train_eval_figure9_llm_variants_200.py.doc.md) ★
- [`pipeline_modes.py`](project/training/pipeline_modes.py.doc.md)
- [`stage2_utils.py`](project/training/stage2_utils.py.doc.md)

#### `training/dataio/` · `training/stage1/` · `training/medgemma/` · `training/trainer/` · `training/evaluation/`
[📁 dataio](project/training/dataio/_index.md) ·
[📁 stage1](project/training/stage1/_index.md) ·
[📁 medgemma](project/training/medgemma/_index.md) ·
[📁 trainer](project/training/trainer/_index.md) ·
[📁 evaluation](project/training/evaluation/_index.md)

### `stage2/prompts/` — Prompt v2
[📁 Directory documentation](project/stage2/prompts/_index.md)

### `medgemma_inference/` — baseline checkpoint ngoài
[📁 Directory documentation](project/medgemma_inference/_index.md)

### `runtime/` · `safety/`
[📁 runtime](project/runtime/_index.md) · [📁 safety](project/safety/_index.md)

### `scripts/` · `preporcessing/` · `configs/` · `cloud/` · `tests/` · `utils/`
[📁 scripts](project/scripts/_index.md) ·
[📁 preporcessing](project/preporcessing/_index.md) ·
[📁 configs](project/configs/_index.md) ·
[📁 cloud](project/cloud/_index.md) ·
[📁 tests](project/tests/_index.md) ·
[📁 utils](project/utils/_index.md)

### `results/` — artifact nghiên cứu đã track
[📁 Directory documentation](project/results/_index.md)

### File ở root
- [`inference.py`](project/inference.py.doc.md) — demo Gradio Vicuna
- [`local_config.py`](project/local_config.py.doc.md) — nạp env config
- [`Dockerfile`](project/Dockerfile.doc.md) · [`inference.sh`](project/inference.sh.doc.md)
- [`build_container.sh`](project/build_container.sh.doc.md) · [`run_container.sh`](project/run_container.sh.doc.md)
- [`threshold.json`](project/threshold.json.doc.md) · [`pyproject.toml`](project/pyproject.toml.doc.md)
- [`requirements-stage1.txt`](project/requirements-stage1.txt.doc.md) ·
  [`requirements-stage2.txt`](project/requirements-stage2.txt.doc.md) ·
  [`requirements.txt`](project/requirements.txt.doc.md) ·
  [`.pre-commit-config.yaml`](project/.pre-commit-config.yaml.doc.md)

### Không có documentation riêng
`notebooks/` và các file legacy khác → xem [Legacy & Optional](project/_meta/LEGACY_AND_OPTIONAL.md)
([D-004](project/_meta/DECISIONS.md#d-004--di-sản-kagglep10-chỉ-liệt-kê)).

---

## Ba điều phải biết trước khi sửa bất cứ thứ gì

### 1. Ranh giới Stage 1 / Stage 2 là bất khả xâm phạm
Mọi import LAVIS/Stage-1 **chỉ** được nằm trong `training/stage1/lavis_loader.py`.
Thêm một `import model.lavis` ở module scope trong `training/` sẽ làm
`medgemma_direct` không khởi động nổi trên máy không có LAVIS — và làm hỏng tính
độc lập của ablation. `tests/test_native_independence.py` sẽ chặn bạn.

### 2. Dữ liệu MIMIC-CXR không bao giờ vào Git
Ảnh, report text, split CSV, feature cache, prediction JSONL, credential, weight.
Notebook đã chạy là đường rò rỉ dễ nhất. `scripts/check_notebook_privacy.py` chạy
như pre-commit hook — **đừng bypass**.

Điều này áp dụng cho cả `struct/`: không viết `subject_id`, `study_id`, `dicom_id`,
đường dẫn ảnh thật hay report text vào bất kỳ trang documentation nào.

### 3. Hai virtual environment, không phải một
Stage 2 dùng lock file additive bao gồm Stage 1 rồi thêm QLoRA packages. Cloud
setup vẫn khuyến nghị tách hai venv để cách ly workflow/dependency nặng.

```bash
python3 -m venv .venv-stage1 && pip install -r requirements-stage1.txt
python3 -m venv .venv-stage2 && pip install -r requirements-stage2.txt
```

`requirements.txt` chỉ include `requirements-stage1.txt`; dùng nó tương đương cài
environment Stage 1, không phải Stage 2.

---

## Lệnh nhanh

```bash
# Bắt buộc trước tiên
cp configs/env_config.yaml.example configs/env_config.yaml   # rồi điền path

# Test CPU (479 test; 5 fail + 1 file không collect được nếu thiếu torchvision)
CUDA_VISIBLE_DEVICES="" python -m pytest tests/ -q

# Preflight trước mọi run GPU
python scripts/vm_preflight.py --stage 1

# Stage 1
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --standalone --nproc_per_node=1 \
    -m pretraining.train --cfg-path pretraining/configs/mimic_cxr_full_l4.yaml

# Stage 2 (smoke)
CUDA_VISIBLE_DEVICES=0 python training/run_medgemma_qlora.py \
    --train-limit 500 --val-limit 10 --test-limit 10 --no-upload \
    --output-dir training/outputs/smoke

# Kiểm tra invariant manifest
python -m training.dataio.validate_manifest --section-mode findings_and_impression
```

Đầy đủ: [ENTRYPOINTS.md](project/_meta/ENTRYPOINTS.md)

---

## Giữ `struct/` không bị lỗi thời

`struct/` là **bộ nhớ dài hạn** của project. Nó chỉ có giá trị khi còn đúng.

Sau khi sửa source, kiểm tra checklist ảnh hưởng:

```text
[ ] cây HOME.md              [ ] call graph        [ ] entrypoints
[ ] _index.md thư mục        [ ] data flow         [ ] configs
[ ] .doc.md file             [ ] architecture      [ ] tests
[ ] .methods/ hàm            [ ] active/legacy     [ ] decisions
```

Chỉ cập nhật phần **thực sự** bị ảnh hưởng. Đừng viết lại toàn bộ `struct/` mỗi lần.

**Source code luôn là chân lý cuối cùng.** Nếu `struct/` mâu thuẫn với code, hãy
đọc code rồi sửa `struct/` — không phải ngược lại.

---

*Knowledge base này được dựng ngày 2026-08-12 bằng audit toàn repository, không
phải bằng suy đoán từ README. Mọi khẳng định đều truy được về đường dẫn + số dòng.
Chỗ nào chưa xác minh được đều ghi rõ `⚠`.*
