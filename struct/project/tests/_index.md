> Source: `tests/` (48 file Python, gồm `conftest.py` và `tests/explainability/`)
> Status: ✅ ACTIVE
> Last verified against source: 2026-08-30

# `tests/`

## Purpose

Test suite CPU. Nhưng vai trò thật của nó lớn hơn "kiểm thử":

> **`tests/` là thứ duy nhất enforce các invariant kiến trúc của repository.**

Ranh giới Stage 1 / Stage 2, tính chất inference-only, teacher-only text — không
có gì trong runtime bảo vệ chúng. Chỉ có test.

Phạm vi documentation: nhóm theo component, không một trang cho mỗi file
([D-010](../_meta/DECISIONS.md#d-010--tests-document-theo-nhóm-component)). Bù lại,
mọi `.doc.md` của source đều có mục `## Tests` link ngược về đây.

## Cách chạy

```bash
CUDA_VISIBLE_DEVICES="" python -m pytest tests/ -q \
  --ignore=tests/test_blip2_negative_sampling.py \
  --ignore=tests/test_encoder_ablation.py
python -m pytest tests/test_stage2_prompts.py -q            # một file
python -m pytest tests/test_stage2_prompts.py -q -k negative_policy   # một test
```

Kết quả Phase 3 ngày 2026-08-14, với hai file cần full LAVIS stack được ignore:
**546 passed, 5 failed, 1 skipped**. Riêng `test_explanation_metrics.py`: 7 passed; `test_blank_label_masking.py`: 5 passed.
Năm failure đúng baseline bên dưới.

⚠ **Trên máy CPU không có torchvision/transformers, 5 test fail và 2 file phải
ignore trước collection.** Đây là trạng thái đã biết, không phải hỏng:

| Không chạy được | Lý do |
|---|---|
| `test_native_independence` (4 test) | thiếu private `configs/env_config.yaml` |
| `test_stage1_eval_hook` (1 test) | import `model.lavis` |
| `test_blip2_negative_sampling` (cả file) | cần torchvision để collect |
| `test_encoder_ablation` (cả file) | cần torchvision để collect |

## `conftest.py` — không có nó thì suite không collect nổi

`tests/conftest.py` đăng ký `model` và `model.lavis` là **package path-only**, để
import một submodule **không** thực thi `model/lavis/__init__.py` — file đó kéo
theo toàn bộ stack GPU. Nó cũng stub `timm.models.hub` khi thiếu timm.

> Khi một test CPU cần import GPU-only, **stub nó trong `conftest.py`** — đừng
> `pip install` vào venv CPU. Cài torchvision/transformers sẽ kéo về vài GB CUDA
> và nâng cấp torch.

Fixture opt-in `report_dataset_module` bổ sung stub Pillow rất hẹp cho
`ReportDataset.py`: chỉ test đường cache/transform explanation mới dùng nó. Stub
không hoạt động khi fixture không được gọi, nên không che các baseline failure
do thiếu full training stack.

---

## Nhóm 1 — Invariant kiến trúc ★ quan trọng nhất

| Test | Invariant được bảo vệ |
|---|---|
| `test_native_independence.py` (215) | **Mọi import LAVIS/Stage-1 chỉ nằm trong `training/stage1/lavis_loader.py`.** Thông điệp fail chỉ đúng cách sửa. Bảo vệ tính độc lập của `medgemma_direct` và tính sạch của ablation |
| `test_inference_only_invariants.py` (258) | `medgemma_inference/` và `model/pretrained_medgemma/` **không** dựng optimizer, không tính gradient, không gọi `model.train()`. Cho phép rõ ràng cho `pretraining/`, `mhcac/`, `vision_encoders/`, `model/lavis/` |
| `test_notebook_privacy.py` (279) | Guard chống rò rỉ MIMIC qua notebook. Fixture ở `tests/fixtures/notebooks/` (clean, credential_like, executed_output, kaggle_ids, synthetic_identifier) |

**Nếu bạn phá một trong ba invariant này, đó không phải "test fail" — đó là lỗi
kiến trúc hoặc lỗi tuân thủ dữ liệu.**

## Nhóm 2 — Stage 1: model, loss, multi-view

| Test | Kiểm gì |
|---|---|
| `test_stage1_objectives.py` | **Teacher-only text** — student không bao giờ thấy text; shape student khớp inference. Loss từ `mhcac/loss.py` |
| `test_blank_label_masking.py` | Ô trống CheXpert -> IGNORE_LABEL(-100), không phải 0; sentinel sống sót int8; ô bị mask không sinh gradient; loss bằng đúng việc bỏ hẳn nhãn; Grad-CAM không coi ô trống là dương tính |
| `test_explanation_loss.py` | Logit Difference Squared, disease-positive gate, Grad-CAM FP32, top-k mềm có gradient, double backprop, warmup, resize mask và vòng đời capture stream |
| `test_explanation_mask_pipeline.py` | RLE row-major round-trip, union phổi, bbox override, Dice gate, memmap lazy, affine dùng chung params, geometry 512→448 và no-cache regression |
| `test_view_fusion.py` | `ViewFusionBlock` là **identity chính xác tại step 0** (zero-init `W_O` + FFN cuối) → checkpoint single-view load không hỏng |
| `test_multiview_losses.py` | `MultiPositiveContrastiveLoss`, `view_consistency_loss` |
| `test_shared_visual_tokens.py` (209) | Thứ tự stream chuẩn hóa, `spans` đúng, `without()` zero-out mà không đổi shape, gradient chảy đúng luồng |
| `test_encoder_ablation.py` (72) | `active_encoders` zero đúng span, all-three giữ đường gốc, tên lạ fail-closed, ablation bị cấm khi training |
| `test_blip2_negative_sampling.py` | Hard negative sampling ⚠ cần torchvision |
| `test_stage1_eval_hook.py` (183) | Eval hook ⚠ cần `model.lavis` |
| `test_mimic_data_pipeline.py` | Study sampling — nạp `mimic_cxr_utils.py` theo path |
| `test_training_core.py` | Scheduler giữ `lr_scale`, đuôi accumulation |
| `test_eval_start_epoch.py` (20 test) | Cửa sổ warm-up `run.eval_start_epoch`: epoch [0]–[4] không eval và **không thể được tuyển làm best**; patience chỉ đếm epoch đã chấm (kẹp `max(best_epoch, eval_start_epoch)`), nếu không run chết ngay ở epoch chấm đầu tiên và log ra như hội tụ. Cũng khẳng định config hiện tại **không thể** early stop |

## Nhóm 3 — Stage 2: mode, soft token, capability

| Test | Kiểm gì |
|---|---|
| `test_pipeline_modes.py` | Resolve mode; `meta_cxr_qformer` vs `..._with_mhcac_prompt` **phân biệt được**; `uses_mhcac_prompt` đúng |
| `test_soft_token_injection.py` | `SoftTokenEmbeddingWrapper` — **THAY THẾ chứ không cộng**, index theo hàng đúng |
| `test_multimodal_capability.py` (330) | Model phải thật sự đa phương thức; fail-closed |
| `test_stage2_utils.py` | Helper Stage 2 |
| `test_run_context.py` | `Stage1Context` |
| `test_trainer_resume.py` (309) | ❓ `CheckpointManager` — resume giữa epoch, RNG snapshot, subdir last/best. **Chỉ test này dùng `training/trainer/`** |

## Nhóm 4 — Prompt v2

| Test | Kiểm gì |
|---|---|
| `test_stage2_prompts.py` (408) | Parity train↔inference; chính sách negative/uncertain; prompt prefix bị mask; `qformer_visual_only` **không nhận label**; `configs/stage2_prompt_v2.yaml` parse được |

## Nhóm 5 — Dataio & manifest

| Test | Kiểm gì |
|---|---|
| `test_manifest.py` (185) | `build_records`, `assert_columns` nêu tên cột thiếu, `assert_no_leakage` |
| `test_section_metrics.py` | Section target; `split_generated_report` |

## Nhóm 6 — Evaluation

| Test | Kiểm gì |
|---|---|
| `test_classification_metrics.py` (377) | P/R/F1, per-pathology, uncertain policy |
| `test_threshold_calibration.py` (256) | Calibrate + baseline + bootstrap |
| `test_generation_metrics.py` (314) | BLEU/ROUGE, error analysis |
| `test_evaluation_integration.py` (379) | End-to-end evaluator |
| `test_clinical_metrics.py` | ⚠ **Chỉ số thiếu báo unavailable, KHÔNG trả 0** |
| `test_counterfactual.py` (320) | ❓ `counterfactual` + `perturbations` — chỉ test này dùng |
| `test_evaluation_config.py` | ❓ `evaluation/config.py` — chỉ test này dùng |
| `test_explanation_metrics.py` | Ba XAI metric thuần NumPy: top-k nhị phân exact-cardinality, all-saliency, Eq. (9) từng box; enforce tách lung/bbox và unavailable ≠ 0 |

## Nhóm 7 — Baseline ngoài & safety

| Test | Kiểm gì |
|---|---|
| `test_pretrained_findings.py` (553) | P8: loader fail-closed, resume, budget, guard Impression, provenance |
| `test_safety_pipeline.py` (297) | ❓ `SafetyPipeline`, `RuleBasedClaimReconciler`, `require_grounding` |

## Nhóm 8 — Explainability Stage 2

`tests/explainability/` — namespace package, **cố ý không có `__init__.py`**
(xem bẫy `iterative-stratification` trong `CLAUDE.md`). 91 test, toàn bộ chạy
trên máy CPU không có transformers/torchvision, `rc=0`.

| Test | Kiểm gì |
|---|---|
| `test_rollout.py` (30) | Rollout Chefer bằng ma trận dựng tay có đáp án tính tay. Pin thứ tự **clamp-rồi-mean** (đảo lại cho ra 0.0 thay vì 2.0, cùng shape, không báo lỗi); pin thứ tự hợp thành lớp; span không có khối lượng → zeros **không phải** phân phối đều |
| `test_projection.py` (33) | Ô vuông tổng hợp 4 góc × 2 lưới, kiểm cả khối lượng góc phần tư lẫn pixel đỉnh; số học `14*32 == 7*64 == 448`; từ chối chiếu soft token Q-Former; `normalize_map` pin số học vào `_normalize_cam` của Stage 1 |
| `test_sentence_attribution.py` (28) | Offset câu khớp `split_sentences`; gộp token→câu theo chồng lấn ký tự; `parse_coverage` gộp theo **câu** không theo study; nhãn theo **allowlist** |

⚠ Tầng GPU (đẩy ảnh thật qua encoder thật, test triệt tiêu soft token) **chưa
viết** — cần máy train.

## Parent

[`struct/project/`](../../HOME.md#source-code-tree)

## Dependencies

`pytest`, `numpy`, `pandas`. Phần lớn test **không cần** torch. Các test cần torch
sẽ skip hoặc fail rõ ràng trên máy CPU thuần.

## Status

```text
✅ ACTIVE
```

## Notes

- ⚠ **Bốn test file là caller DUY NHẤT của code production.** Đó là dấu hiệu
  [D-001](../_meta/DECISIONS.md#d-001--hạ-tầng-đã-viết-nhưng-chưa-nối-vào-pipeline):
  `test_trainer_resume`, `test_safety_pipeline`, `test_evaluation_config`,
  `test_counterfactual`.

- **`tests/fixtures/notebooks/*.ipynb.fixture`** dùng đuôi `.fixture` để không bị
  `check_notebook_privacy.py` quét — chúng cố ý chứa pattern giống dữ liệu thật.

- 12 test file có `if __name__ == "__main__"` để chạy lẻ khi debug. Đường chuẩn
  vẫn là `pytest`.

## Related documentation

[ACTIVE_COMPONENTS.md](../_meta/ACTIVE_COMPONENTS.md#ba-invariant-giữ-bản-đồ-này-đúng) ·
[LEGACY_AND_OPTIONAL.md](../_meta/LEGACY_AND_OPTIONAL.md) · [D-010](../_meta/DECISIONS.md#d-010--tests-document-theo-nhóm-component)

← [Về HOME](../../HOME.md)
