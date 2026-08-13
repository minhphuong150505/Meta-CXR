> Source: kết quả trace import/caller trên toàn repository
> Status: ✅ ACTIVE
> Last verified against source: 2026-08-12

# Active Components

Bản đồ những gì **thực sự đang chạy**. Mọi mục ở đây có caller cụ thể, kiểm chứng
được — không có mục nào dựa trên giả định.

Thứ không nằm ở đây: xem [LEGACY_AND_OPTIONAL.md](LEGACY_AND_OPTIONAL.md).

**Nhãn:** `✅ ACTIVE` · `🟡 CONDITIONAL` (chạy khi config bật) · `🧰 UTILITY` ·
`🧪 ABLATION` · `❓ UNKNOWN`

---

## Stage 1 — đường chạy production

```text
pretraining/train.py                                          ✅ ENTRYPOINT
 └── model/lavis/common/config.py            Config            ✅
 └── model/lavis/common/dist_utils.py        init_distributed  ✅
 └── model/lavis/tasks/image_text_pretrain.py                  ✅
 └── model/lavis/data/ReportDataset.py       MIMIC_CXR_Dataset ✅
 │    └── model/lavis/data/mimic_cxr_utils.py build_study_index ✅
 │    └── explanation mask cache + synchronized affine          🟡 mask_cache_dir
 └── model/lavis/models/blip2_models/blip2_qformer.py          ✅ ★ trung tâm
 │    ├── biovil_t/                          BioViL-T          ✅
 │    │    ├── model.py  encoder.py  modules.py  resnet.py
 │    │    └── transformer.py  pretrained.py  types.py  device.py
 │    ├── vision_encoders/pubmedclip/pubmed_clip.py            ✅
 │    ├── vision_encoders/swin/swin_encoder.py                 ✅
 │    ├── vision_encoders/rad_dino/rad_dino_encoder.py         🟡 raddino: false
 │    ├── vision_encoders/shared_visual_tokens.py              ✅
 │    ├── mhcac/view_fusion.py                ViewFusionModule 🟡 multi_view: true
 │    ├── mhcac/mhcac_12.py                   MHCAC            ✅
 │    ├── mhcac/explanation.py                Grad-CAM loss    🟡 lambda_explanation
 │    ├── mhcac/loss.py                       11 loss          ✅
 │    ├── model/lavis/models/blip2_models/blip2.py  Blip2Base  ✅
 │    ├── model/lavis/models/blip2_models/Qformer.py           ✅
 │    └── model/lavis/models/blip_models/blip_outputs.py       ✅
 └── model/lavis/runners/runner_base.py       RunnerBase       ✅
      ├── model/lavis/common/optims.py        LR scheduler     ✅
      ├── model/lavis/common/registry.py                       ✅
      └── training/evaluation/classification_metrics.py        ✅ (import trễ)
```

### Trong `mhcac/`, chỉ ba file active

| File | Trạng thái | Bằng chứng |
|---|---|---|
| `mhcac_12.py` | ✅ | `blip2_qformer.py:23` |
| `loss.py` | ✅ | `blip2_qformer.py:35` |
| `view_fusion.py` | 🟡 | `blip2_qformer.py:41`, chỉ dựng khi `multi_view: true` |
| 11 file còn lại + `utils.py` + `aggregator.py` | 🕰 | [LEGACY_AND_OPTIONAL.md](LEGACY_AND_OPTIONAL.md) |

### Trong `vision_encoders/`

| Thư mục | Trạng thái |
|---|---|
| `pubmedclip/`, `swin/`, `shared_visual_tokens.py` | ✅ |
| `rad_dino/` | 🟡 CONDITIONAL — có wire, nhưng mọi config đặt `raddino: false` |
| `biovil_t/` (bản sao) | 🕰 LEGACY — bản được import là `biovil_t/` top-level |
| `medclip/` | 🕰 LEGACY — import bị comment |

---

## Stage 2 — đường chạy production

```text
training/run_medgemma_qlora.py                                ✅ ENTRYPOINT
 ├── training/pipeline_modes.py           resolve modes       ✅ stdlib-only
 ├── training/run_context.py              Stage1Context       ✅
 ├── training/dataio/manifest.py          build_records       ✅ CHỈ pandas
 │    └── training/dataio/validate_manifest.py                🧰 (cũng là CLI)
 ├── training/stage2_utils.py                                 ✅
 ├── training/torch_io.py                 load_torch_checkpoint ✅
 ├── stage2/prompts/                      PromptBuilder       ✅ stdlib-only
 │    ├── builder.py  schemas.py  policies.py  templates.py
 │    └── ontology.py  records.py  validation.py
 ├── training/medgemma/soft_tokens.py     SoftTokenEmbeddingWrapper 🟡 mode Q-Former
 ├── training/medgemma/capabilities.py    kiểm tra multimodal ✅
 ├── training/train_eval_figure9_llm_variants_200.py          ✅ ★ ĐỘNG CƠ
 │                                          (tên file gây hiểu nhầm — D-006)
 └── training/stage1/lavis_loader.py                          🟡 CỬA DUY NHẤT sang Stage 1
      └── (chỉ khi --pipeline-mode cần Stage 1)
```

---

## Evaluation

```text
scripts/calibrate_thresholds.py                               ✅
scripts/evaluate_stage1.py                                    ✅
scripts/evaluate_stage2.py                                    ✅
 │
 └── training/evaluation/
      ├── schemas.py                  ✅   ClassificationPredictions, load_generation_records
      ├── classification_metrics.py   ✅   P/R/F1, AUROC, AUPRC (662 dòng)
      ├── uncertain_policy.py         ✅   ignore_uncertain / …
      ├── threshold_calibration.py    ✅   chỉ chạy trên validation
      ├── baselines.py                ✅   all-negative baseline
      ├── bootstrap.py                ✅   khoảng tin cậy
      ├── generation_metrics.py       ✅   BLEU, ROUGE-L (tự implement)
      ├── error_analysis.py           ✅   → safety/claims.py
      ├── subgroup_analysis.py        ✅
      ├── report_writer.py            ✅   markdown + json
      ├── clinical.py                 ✅   luôn báo unavailable/not-implemented
      ├── visualization.py            🟡   import trễ, cần extra eval-plots
      ├── config.py                   ❓   D-001
      ├── counterfactual.py           ❓   D-001
      └── perturbations.py            ❓   D-001
```

---

## External MedGemma baseline (P8)

```text
medgemma_inference/run_pretrained_findings.py                 ✅ ENTRYPOINT
 ├── medgemma_inference/config.py         validate YAML       ✅
 ├── medgemma_inference/runner.py                             ✅
 │    └── runtime/budget.py    BudgetState, BudgetExceeded    ✅
 ├── medgemma_inference/prediction_writer.py  JSONL fsync     ✅
 ├── medgemma_inference/progress.py       run identity        ✅
 ├── model/pretrained_medgemma/findings_loader.py             ✅
 │    └── runtime/device.py    plan_device                    ✅
 ├── model/pretrained_medgemma/findings_reporter.py           ✅
 ├── model/pretrained_medgemma/output_schema.py               ✅
 ├── model/pretrained_medgemma/errors.py                      ✅
 ├── model/pretrained_medgemma/impression_reporter.py         ⛔ DISABLED có chủ đích
 └── configs/experiments/pretrained_medgemma_findings_first.yaml ✅
```

---

## Demo Vicuna (P9)

```text
Dockerfile → inference.sh → inference.py                      ✅ D-002
 ├── pretraining/configs/blip2_pretrain_stage1_emb.yaml       ✅
 ├── model/lavis/models/blip2_models/modeling_llama_imgemb.py ✅
 ├── model/lavis/data/ReportDataset.py                        ✅
 │     create_chest_xray_transform_for_inference, ExpandChannels
 ├── checkpoints/                        LoRA adapter (không track) ✅
 ├── build_container.sh  run_container.sh                     ✅
 └── local_config.py    JAVA_HOME / JAVA_PATH cho CheXpert    ✅
```

⚠ `utils/prompter.py`, `callbacks.py`, `datacollator.py` **không** được
`inference.py` import, dù docs cũ nói vậy — xem [LEGACY_AND_OPTIONAL.md](LEGACY_AND_OPTIONAL.md).

---

## Preprocessing & tooling

| Component | Nhãn | Vai trò |
|---|---|---|
| `preporcessing/preprocess_mimic_cxr.py` | ✅ | Dựng split CSV |
| `preporcessing/mimic_report_parser.py` | ✅ | Trích FINDINGS/IMPRESSION |
| `preporcessing/build_explanation_masks.py` | 🟡 | Dựng private CheXmask/MS-CXR cache khi chạy explanation-aware |
| `local_config.py` | ✅ | Nạp `configs/env_config.yaml` |
| `scripts/vm_preflight.py` | 🧰 | Kiểm tra trước run dài |
| `scripts/check_notebook_privacy.py` | ✅ | Pre-commit hook chặn rò rỉ |
| `pretraining/precompute_features.py` | 🟡 | Chỉ khi `run.feature_cache_dir` |
| `scripts/run_prompt_ablation.py` + `_stage2_fixtures.py` | 🧪 | P6 |
| `scripts/export_stage2_prompt_samples.py` | 🧪 | ⚠ output chứa findings text |
| `scripts/prompt_length_statistics.py` | 🧪 | |
| `scripts/audit_temporal_targets.py` | 🧪 | |
| `tests/` (35 file) | ✅ | Enforce invariant kiến trúc, gồm explanation-mask và inference-only encoder ablation |

---

## Tổng kết theo trạng thái

| Nhãn | Số file (không tính `model/lavis/`) | Ghi chú |
|---|---|---|
| ✅ ACTIVE | ~95 | Có caller production xác nhận |
| 🟡 CONDITIONAL | ~8 | `rad_dino`, `view_fusion`, `soft_tokens`, `visualization`, `precompute_features`, `lavis_loader` |
| 🧪 ABLATION / EXPERIMENTAL | ~10 | Script prompt + 5 cloud wrapper |
| ❓ UNKNOWN | ~8 | D-001 |
| 🕰 LEGACY | ~25 | D-003, D-004 |
| ⚠ POTENTIALLY_UNUSED | 4 | `utils/` |

---

## Ba invariant giữ bản đồ này đúng

Nếu một trong ba invariant sau bị phá, bản đồ trên **sai ngay lập tức**:

1. **Ranh giới Stage 1 / Stage 2** — mọi import LAVIS chỉ ở
   `training/stage1/lavis_loader.py`. Bảo vệ bởi `tests/test_native_independence.py`.
2. **Inference-only** — `medgemma_inference/` và `model/pretrained_medgemma/`
   không được dựng optimizer / tính gradient / gọi `model.train()`. Bảo vệ bởi
   `tests/test_inference_only_invariants.py`.
3. **Teacher-only text** — MHCAC không bao giờ nhận text ở nhánh student. Bảo vệ
   bởi `tests/test_stage1_objectives.py::test_mhcac_text_is_teacher_only_and_student_shape_matches_inference`.

---

← [Về HOME](../../HOME.md) · [LEGACY_AND_OPTIONAL.md](LEGACY_AND_OPTIONAL.md) · [DECISIONS.md](DECISIONS.md)
