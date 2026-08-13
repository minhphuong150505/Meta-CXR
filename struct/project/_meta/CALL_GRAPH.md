> Source: `pretraining/train.py`, `model/lavis/runners/runner_base.py`, `model/lavis/tasks/`, `training/run_medgemma_qlora.py`
> Status: ✅ ACTIVE
> Last verified against source: 2026-08-12

# Call Graph

Đường đi thực thi ở mức critical path. **Không** liệt kê mọi hàm — chỉ những mắt
xích mà nếu không hiểu thì không lần được luồng.

Trang này phục vụ cả hai hướng đọc: **top-down** (từ entrypoint xuống) và
**bottom-up** (từ một hàm ngược lên xem ai gọi nó).

---

## 1. Stage 1 — top-down

```text
python -m torch.distributed.run … -m pretraining.train --cfg-path <yaml>
│
└─ pretraining/train.py :: main()                                      ← ENTRYPOINT
   │
   ├─ parse_args()                                → argparse
   ├─ Config(args)                                → model/lavis/common/config.py
   │     └─ merge CHỈ 3 block: run / model / datasets   ⚠ data: phải nằm trong model:
   ├─ init_distributed_mode(cfg)                  → common/dist_utils.py
   │     └─ đọc RANK / WORLD_SIZE / LOCAL_RANK do torchrun đặt
   ├─ setup_seeds(cfg)                            → seed = cfg.run_cfg.seed + get_rank()
   ├─ setup_logger()
   ├─ wandb.init(...)                             ← CHỈ rank 0; rank khác dùng mode="disabled"
   │
   ├─ tasks.setup_task(cfg)                       → ImageTextPretrainTask
   │
   ├─ MIMIC_CXR_Dataset(split="train"|"val"|"test")   → model/lavis/data/ReportDataset.py
   │     ├─ _init_study_index()                   → mimic_cxr_utils.build_study_index()
   │     └─ _init_feature_cache()                 → chỉ khi run.feature_cache_dir được đặt
   │
   ├─ task.build_model(cfg)
   │     └─ registry.get_model_class("blip2").from_config(cfg.model_cfg)
   │           └─ Blip2Qformer.from_config()      → blip2_qformer.py:1298
   │                 ├─ đọc encoders / multi_view / view_fusion / loss / mhcac
   │                 ├─ Blip2Qformer.__init__()
   │                 │     ├─ init_vision_encoder("biovil", …)  → biovil_t/
   │                 │     ├─ init_Qformer(32, vis_num_feat, 2) → Qformer.py
   │                 │     ├─ Pubmedclip / SwinEncoder / RadDinoEncoder
   │                 │     ├─ ViewFusionModule × (số encoder bật)  → mhcac/view_fusion.py
   │                 │     ├─ SharedVisualTokenProjector          → vision_encoders/shared_visual_tokens.py
   │                 │     ├─ AbnormalityClassificationModel      → mhcac/mhcac_12.py
   │                 │     └─ ClassificationLoss                  → mhcac/loss.py
   │                 └─ load_checkpoint_from_config(cfg)    ← BLIP-2 pretrained weights
   │
   └─ RunnerBase(cfg, job_id, task, model, datasets).train(wandb_run)
```

### Vòng train

```text
RunnerBase.train(wandb_run)                        runner_base.py:648
│
├─ (nếu resume_ckpt_path)  _load_checkpoint()
│
└─ FOR epoch in range(max_epoch):
   │
   ├─ train_epoch(epoch)                           runner_base.py:787
   │  └─ task.train_epoch(...)                     tasks/base_task.py:131
   │     └─ _train_inner_loop(...)                 tasks/base_task.py:192
   │        │
   │        └─ FOR micro_step in dataloader:
   │           ├─ lr_scheduler.step()              ← MỖI OPTIMIZER STEP, không phải mỗi microbatch
   │           │     └─ giữ nguyên tỉ lệ lr_scale của từng param group
   │           ├─ task.train_step(model, samples)  base_task.py:92
   │           │     └─ model(samples)  ──────────►  Blip2Qformer.forward()   ★ xem §2
   │           ├─ scaler.scale(loss).backward()    ← AMP bfloat16
   │           └─ mỗi accum_grad_iters:
   │                 clip_grad_norm_(max_grad_norm) → optimizer.step() → zero_grad()
   │
   ├─ validate(epoch, best, best_epoch, wandb_run) runner_base.py:545
   │  └─ eval_epoch("val", epoch)                  runner_base.py:806
   │     └─ ImageTextPretrainTask.evaluation()     tasks/image_text_pretrain.py:53
   │        ├─ model(samples) → logits
   │        ├─ _build_predictions(logits, labels, keys)
   │        ├─ training.evaluation.classification_metrics.*   ← import TRỄ, trong hàm
   │        └─ _save_predictions()                 → .npz nếu save_predictions: true
   │
   ├─ _metric_improved(value, best)                ← selection_metric = macro_auprc
   │     ├─ cải thiện  → _save_checkpoint(is_best=True)  → checkpoint_best.pth
   │     └─ không      → early_stop_counter += 1
   │
   ├─ _save_checkpoint(is_last=True)               → checkpoint_last.pth
   ├─ mỗi save_freq epoch: _save_checkpoint(epoch) → checkpoint_<epoch>.pth
   └─ early_stop_counter >= early_stop_patience → break
   │
   ▼ sau vòng lặp
   evaluate(cur_epoch="best")                      runner_base.py:776
   └─ _reload_best_model() → eval_epoch("test")    ← TEST chạy đúng MỘT LẦN, từ checkpoint_best
```

**Ba điểm dễ hiểu sai:**

1. `lr_scheduler.step()` gọi **một lần cho mỗi optimizer update**, không phải mỗi
   microbatch. Vì thế `warmup_steps: 300` nghĩa là 300 optimizer step. Config
   recipe 2×T4 cũ (đã xóa) để `32000` — ramp không bao giờ xong.
2. `validate()` chạy trên **toàn bộ** validation split, không phải subset.
3. Test split **không tham gia** chọn checkpoint ở bất kỳ bước nào.

---

## 2. `Blip2Qformer.forward()` — trái tim Stage 1

```text
Blip2Qformer.forward(samples)                      blip2_qformer.py:820
│
├─ gom cached / aux_cached từ samples["<enc>_feat"], samples["aux_<enc>_feat"]
│
├─ _encode_image_streams(image, cached, aux_image, aux_cached, aux_mask, …)   :469
│  │
│  ├─ _encode_aux_streams(aux_image, aux_cached)                              :400
│  │     └─ torch.no_grad():  visual_encoder / pubmedclip / swin / raddino
│  │        ⚠ chỉ FEATURE bị detach — W_K/W_V của fusion nằm NGOÀI block này
│  │
│  ├─ FOR mỗi encoder bật:
│  │     ├─ _stash_prefusion(name, anchor, aux_streams)     ← chỉ khi λ_mpc>0 hoặc λ_vc>0
│  │     └─ _fuse(name, anchor, …)  ──► ViewFusionModule[name].forward()
│  │
│  └─ SharedVisualTokenProjector(raw_streams) ──► SharedVisualTokens
│
├─ _batch_mask(samples, "classification_mask", …)  ← fallback: has_chexpert_label
├─ _batch_mask(samples, "generation_mask", …)
│
├─ Qformer.bert(query_embeds=query_tokens, encoder_hidden_states=image_embeds)
│     └─ vision_proj → normalize → image_features [B,32,256]
├─ tokenizer(text) → Qformer.bert(text) → text_proj → text_features [B,256]
│
├─ _image_text_contrastive(image_features, text_features, generation_mask)    :640
│     ├─ _gather_with_local_grad()          ← all-gather giữ grad ở rank cục bộ
│     ├─ nối itc_image_queue / itc_text_queue   ⚠ queue_filled = 0 khi KHÔNG training
│     └─ cross_entropy 2 chiều → loss_itc
│
├─ _image_text_matching(image_embeds, text_tokens, …)                         :719
│     ├─ _hard_negative_sampling_weights(sim, valid, positive_idx)            :52
│     │     └─ FP32, loại positive TRƯỚC softmax  ← tránh CUDA assert ở BF16
│     ├─ torch.multinomial → negative image / text
│     └─ Qformer.bert(triple pos/neg-img/neg-txt) → itm_head → cross_entropy
│
├─ _language_modeling(text_tokens, query_tokens, query_output, generation_mask):793
│     └─ Qformer(decoder, past_key_values=query_output.past_key_values, labels)
│
├─ _update_itc_queue(image_features, text_features, generation_mask)          :604
│     └─ @torch.no_grad, chỉ khi self.training
│
├─ ★ mhcac(shared_visual, text_embeddings=None,  labels, sample_mask)   ← STUDENT  :907
│     └─ AbnormalityClassificationModel.forward()      mhcac/mhcac_12.py:354
│        ├─ embedding_alignment(visual_tokens, text_embeddings, expert_tokens)
│        ├─ FOR name, span in sorted(spans):        ← slice theo encoder
│        │     ├─ cnn_downsampler (biovil) HOẶC _resize_patch_sequence
│        │     └─ pos_enc(stream)                   ← positional encoding RIÊNG mỗi encoder
│        ├─ 6 × ExpertTokenCrossAttention
│        ├─ expert_loss(...) → orth_loss, contrastive_loss, sparsity_loss
│        └─ 14 × classifiers → logits [B,14,3]
│
├─ cls_loss_fn(student_logits, cls_labels, sample_mask)   → mhcac/loss.py
│
├─ IF teacher_mask.any():
│     ├─ ★ mhcac(shared_visual, text_embeddings=text_output.last_hidden_state)  ← TEACHER  :925
│     ├─ cls_loss_fn(teacher_logits, …, sample_mask=teacher_mask)
│     └─ soft_target_kl_loss(student, teacher.detach(), …)
│
├─ IF multi_view và aux_mask.any():
│     ├─ mpc_loss_fn(anchor_raw, aux_raw, aux_mask)   → MultiPositiveContrastiveLoss
│     └─ ★ mhcac(anchor_shared, …)  ← lần gọi THỨ BA, chỉ anchor, cho view_consistency  :965
│
└─ total_loss = Σ λᵢ · lossᵢ   (11 số hạng)  → BlipOutput
```

⚠ **MHCAC được gọi tối đa 3 lần trong một forward**: student, teacher, anchor-only.
Đây là chi phí tính toán đáng kể và là thứ cần biết trước khi sửa MHCAC.

---

## 3. Stage 2 — top-down

```text
python training/run_medgemma_qlora.py [flags]
│
└─ training/run_medgemma_qlora.py :: main()                                :404
   │
   ├─ parse_args()                                                          :52
   ├─ resolve_pipeline_modes(selection)          → training/pipeline_modes.py:170
   │     ├─ từ chối EXTERNAL_INFERENCE_MODES     ← raise, không cho chạy P8 qua đây
   │     ├─ both_for_ablation → [MEDGEMMA_DIRECT, META_CXR_QFORMER]
   │     └─ alias cũ native/qformer/both → tên mới
   │
   ├─ modes_require_stage1(modes)
   │     ├─ False → build_native_records()                                   :364
   │     │            └─ dataio/manifest.py::build_records()   ← CHỈ pandas
   │     │                 ├─ assert_no_leakage()
   │     │                 └─ assert_columns()   ← fail nêu tên cột thiếu
   │     │
   │     └─ True  → build_stage1_records()
   │                  └─ ★ training/stage1/lavis_loader.py     ← ĐIỂM DUY NHẤT được import LAVIS
   │                       ├─ build_cfg()
   │                       ├─ build_stage1_model()
   │                       │     ├─ load_torch_checkpoint()    → training/torch_io.py
   │                       │     ├─ filter_state_dict_for_model()
   │                       │     └─ load_state_dict_materializing_meta()
   │                       └─ make_stage1_loader()             → MIMIC_CXR_Dataset
   │
   ├─ (nếu --prompt-config) stage2.prompts.load_prompt_config()              :410
   │
   ├─ train_mode(mode, records, …)                                           :211
   │     └─ fig9.VariantLLM(...)      → train_eval_figure9_llm_variants_200.py
   │           ├─ load MedGemma 4-bit NF4 + language-only LoRA
   │           │     (CLI mặc định r=16, α=32; không dùng all-linear)
   │           ├─ PromptBuilder(prompt_config).build(context)  → stage2/prompts/builder.py
   │           ├─ SoftTokenEmbeddingWrapper                     → training/medgemma/soft_tokens.py
   │           │     └─ validate_soft_token_batch()             → training/stage2_utils.py
   │           ├─ train loop → chọn checkpoint theo validation cross-entropy
   │           ├─ generate(bad_words_ids=soft_token_bad_words_ids(...))
   │           └─ compute_nlg(preds, refs)
   │
   ├─ resumable_adapter(path, image_mode)                                    :180
   └─ upload_safe_run(root, adapter_dirs, gcs_output)                        :186
```

### Ranh giới độc lập — vẽ lại cho rõ

```text
        ┌──────────────────────────────────────────┐
        │  run_medgemma_qlora.py                   │
        │  pipeline_modes.py                       │   KHÔNG import LAVIS
        │  dataio/manifest.py                      │   ở module scope
        │  medgemma/soft_tokens.py, capabilities.py│
        │  stage2/prompts/*                        │
        └────────────────┬─────────────────────────┘
                         │ chỉ gọi từ TRONG nhánh đã quyết định cần Stage 1
                         ▼
        ┌──────────────────────────────────────────┐
        │  training/stage1/lavis_loader.py         │   ★ CỬA DUY NHẤT
        └────────────────┬─────────────────────────┘
                         ▼
              model/lavis/*, mhcac/*, vision_encoders/*
```

Enforce bằng `tests/test_native_independence.py`. Nếu bạn thêm một
`import model.lavis` ở module scope trong `training/`, test sẽ fail và nêu đúng
câu: *"Move the import into training/stage1/lavis_loader.py"*.

---

## 4. Evaluation — top-down

```text
scripts/calibrate_thresholds.py :: main()
└─ training/evaluation/schemas.py::ClassificationPredictions   ← đọc .npz
   └─ threshold_calibration.calibrate(...)
      └─ uncertain_policy.POLICIES[...]       ← ignore_uncertain / …
         → thresholds.json

scripts/evaluate_stage1.py :: main()
├─ ClassificationPredictions (test .npz)
├─ evaluation.classification_metrics.evaluate_classification()
├─ evaluation.baselines.compute_baselines() / baseline_table()
├─ evaluation.bootstrap.*                    ← khoảng tin cậy
├─ evaluation.subgroup_analysis.*
├─ evaluation.report_writer.*                → markdown + json
└─ (nếu --plots) evaluation.visualization    ← import TRỄ, cần extra eval-plots

scripts/evaluate_stage2.py :: main()
├─ evaluation.schemas.load_generation_records()   ← đọc .jsonl
├─ evaluation.generation_metrics.*                ← BLEU/ROUGE tự implement
├─ evaluation.error_analysis.analyse_sample()
│     └─ safety.claims.*                          ← DUY NHẤT chỗ safety/ được dùng thật
├─ evaluation.bootstrap.*
├─ evaluation.subgroup_analysis.*
└─ (nếu không --skip-clinical-metrics) evaluation.clinical
      └─ raise MissingOptionalDependency / NotImplementedError
         ⚠ báo "unavailable", KHÔNG BAO GIỜ trả 0
```

---

## 5. Bottom-up — ai gọi cái này?

Tra ngược cho các thành phần hay bị sửa:

| Thành phần | Được gọi bởi |
|---|---|
| `AbnormalityClassificationModel.forward` | `Blip2Qformer.forward` × 3 (student `:907`, teacher `:925`, anchor `:965`) |
| `ViewFusionModule.forward` | `Blip2Qformer._fuse` (`:458`), một lần cho mỗi encoder bật |
| `SharedVisualTokenProjector.forward` | `_encode_image_streams` (`:542`) và nhánh view-consistency (`:964`) |
| `ClassificationLoss.__call__` | `Blip2Qformer.forward` (`:913` student, `:933` teacher) |
| `soft_target_kl_loss` | `Blip2Qformer.forward` (`:936`) |
| `MIMIC_CXR_Dataset` | `pretraining/train.py:main`, `training/stage1/lavis_loader.py:make_stage1_loader` |
| `PromptBuilder` | `fig9.VariantLLM` (`:928`), `scripts/run_prompt_ablation.py`, `export_stage2_prompt_samples.py`, `prompt_length_statistics.py` |
| `SoftTokenEmbeddingWrapper` | `fig9` (`:100` / `:108`, dual-import shim) |
| `dataio.manifest.build_records` | `run_medgemma_qlora.py:27`, `fig9:116/122`, `medgemma_inference/run_pretrained_findings.py:33` |
| `training.torch_io.load_torch_checkpoint` | `stage1/lavis_loader.py:32`, `trainer/checkpointing.py:23`, `fig9:102/113` |
| `safety.claims.*` | `training/evaluation/error_analysis.py:30` **(caller duy nhất ngoài test)** |
| `runtime.budget.BudgetState` | `medgemma_inference/runner.py:23` |
| `runtime.device.plan_device` | `model/pretrained_medgemma/findings_loader.py:24` |
| `training/evaluation/classification_metrics` | `scripts/evaluate_stage1.py`, và **`tasks/image_text_pretrain.py:223` (import trễ trong hàm)** |

### Không có caller production

| Thành phần | Chỉ được gọi bởi |
|---|---|
| `trainer.CheckpointManager` / `TrainingState` | `tests/test_trainer_resume.py` |
| `safety.SafetyPipeline` / `verifiers` / `reconciler` | `tests/test_safety_pipeline.py` |
| `evaluation.config.*` | `tests/test_evaluation_config.py` |
| `evaluation.counterfactual` / `perturbations` | `tests/test_counterfactual.py` |
| `utils/prompter.py`, `callbacks.py`, `datacollator.py` | **không ai** |

Xem [D-001](DECISIONS.md#d-001--hạ-tầng-đã-viết-nhưng-chưa-nối-vào-pipeline) và
[D-002](DECISIONS.md#d-002--đường-vicuna-7b-legacy-vẫn-là-demo-active).

---

## 6. Dual-import shim — đọc trước khi sửa `training/`

Mọi module trong `training/` mang pattern này:

```python
try:
    from stage2_utils import stable_fingerprint          # khi chạy như script
except ImportError:
    from training.stage2_utils import stable_fingerprint  # khi chạy qua python -m
```

Lý do: `run_medgemma_qlora.py:26` làm `sys.path.insert(0, dirname(abspath(__file__)))`,
nên `training/` trở thành import root khi chạy trực tiếp. **Giữ nguyên pattern này
trong file mới ở `training/`** — bỏ nó đi sẽ làm một trong hai cách gọi hỏng.

---

← [Về HOME](../../HOME.md) · [ARCHITECTURE.md](ARCHITECTURE.md) · [DATA_FLOW.md](DATA_FLOW.md) · [ENTRYPOINTS.md](ENTRYPOINTS.md)
