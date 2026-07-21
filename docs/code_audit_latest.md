# Code Audit — latest source

Audited: 2026-07-21
Base branch: `feat/medgemma-direct-default`
Base HEAD: `b174ca39cc60c583c8b5043e0bd8cf03ffbbbb6c` (2026-07-21 17:54:45 +0700)
Working branch: `refactor/clean-medgemma-xai-pipeline`
Working tree at audit start: **clean** (nothing stashed, nothing discarded)
Remote sync: 0 ahead / 0 behind `origin/feat/medgemma-direct-default`

Baseline test run before any change:

```
$ /home/phuong/venv/bin/python -m pytest tests/ training/test_stage2_utils.py -q
67 passed in 3.18s
```

## Scope note

This audit covers the code that was actually read, not the whole tree. Files read
end-to-end or in the relevant sections: `training/run_medgemma_qlora.py`,
`training/pipeline_modes.py`, `training/stage2_utils.py`,
`training/train_eval_figure9_llm_variants_200.py` (structure + `SoftTokenEmbeddingWrapper`,
`VariantLLM.__init__`, `encode_train_example`, `collate_train`, `_forward_batch`,
`train_fine`, `evaluate_variant`), `training/dataio/manifest.py`,
`model/lavis/models/blip2_models/blip2_qformer.py` (`_encode_image_streams` and its
call sites), `mhcac/mhcac_12.py` (`CrossModalEmbeddingAlignment`), plus repo-wide greps
for the anti-patterns listed in the brief. `model/lavis/` beyond `blip2_qformer.py`,
`inference.py`, `biovil_t/`, `vision_encoders/` and the notebooks were **not** read in
full; findings there are limited to what the greps showed and are marked as such.

---

## Findings

| Sev | Component | File / symbol | Problem | Consequence | Reproduce | Fix direction | Test to add |
|---|---|---|---|---|---|---|---|
| **Critical** | Stage-1 visual merge | `blip2_qformer.py:469` vs `:837-841` | There is no single `shared_visual_tokens`. `_encode_image_streams` returns the three streams **separately** (`cnn_patches` 1408, `vit_patches` 768, `swin_patches` swin_dim) *and* a concatenated `qformer_image_embeds`. META-Former consumes the concat; MHCAC consumes the three raw streams and re-projects them through its **own** `CrossModalEmbeddingAlignment` (`mhcac_12.py:66-104`). Two independent projection/merge paths. | The two branches are trained on different visual representations. Any claim that both consume "the same shared representation" is unsupported by the code. | Read `blip2_qformer.py:764` and `:837-841`; MHCAC never receives `image_embeds`. | **Blocked on a decision — see "Open architectural question" below.** The paper text supports *both* readings. | `test_shared_visual_tokens.py`: assert both branches derive from one tensor (only meaningful after the decision) |
| **High** | Stage-2 evaluation | `train_eval_figure9_llm_variants_200.py:1442` | `compute_nlg(preds, refs)` scores the **joined** `"FINDINGS: … IMPRESSION: …"` string. `manifest.split_generated_report` exists and is unit-tested but is **never called in production** (grep: only tests reference it). | With the default `findings_and_impression`, a model that writes a good FINDINGS and an empty IMPRESSION is scored the same way as one that splits effort correctly. Brief §X and §XVIII require separate Findings / Impression / full-report metrics. | `grep -rn split_generated_report --include=*.py .` → only `manifest.py` def + 3 test files | Split pred and ref via `split_generated_report`, emit `findings_*`, `impression_*`, `full_*` metric blocks. | metrics contain all three blocks; empty IMPRESSION is penalised |
| **High** | Global mutable state | `run_medgemma_qlora.py:375-378` | `fig9.RUN_NAME = …`, `fig9.STAGE1_CONFIG_PATH_OVERRIDE = …`, `fig9.STAGE1_CHECKPOINT_PATH_OVERRIDE = …`, `fig9.THRESHOLDS = …` — the entry point mutates another module's globals. `evaluate_variant` then reads `RUN_NAME` at `:1387,1395,1450`. | Config is not snapshot-able, two runs in one process interfere, and the eval fingerprint depends on import-time state rather than an argument. Explicitly forbidden by brief §VII. | Read the four assignments; then `grep -n "RUN_NAME" train_eval_figure9_llm_variants_200.py` | Frozen dataclass config passed by constructor/arg. | config immutability + no module-global reads |
| **High** | Native/Stage-1 coupling | `run_medgemma_qlora.py:26,47` | `sys.path.insert` then `import train_eval_figure9_llm_variants_200 as fig9`, which at `:94-95` inserts two more paths and imports the LAVIS/torch stack. `medgemma_direct` therefore imports Stage-1 even though it uses none of it. The in-file comment at `:43-46` already acknowledges this. | Brief §XXVI lists "Native MedGemma import Stage 1" as a blocking defect. Also makes CPU-only testing of the native path impossible. | `python -c "import training.run_medgemma_qlora"` pulls LAVIS | Extract loader/collator/trainer/evaluator out of the Figure-9 module; native path imports none of it. | native entry point imports without torch/LAVIS |
| **High** | God script / god class | `train_eval_figure9_llm_variants_200.py` (1613 lines); `VariantLLM` (`:629-1289`, ~660 lines) | One class loads the model, attaches LoRA, builds prompts, collates, trains, generates, evaluates, saves and is read by the uploader. It is the main dependency of the full-data pipeline. | Brief §VI/§XXVI. Untestable without a GPU; every change risks the whole Stage-2 path. | `wc -l`; `grep -n "^    def " …` | Split into loader / LoRA config / reporter / collator / injector / trainer / evaluator / checkpoint manager. | per-unit tests for each extracted piece |
| **Medium** | Metric-based selection | `run_medgemma_qlora.py:455` + `train_fine:1172-1176` | Checkpoint selection is `validation_cross_entropy` only. | Brief §XVIII forbids selecting on loss/BLEU/ROUGE alone and requires a configurable `primary_selection_metric`. | read `summary["selection_metric"]` | Config-driven selection metric with a clinical option. | selection metric is honoured |
| **Medium** | Missing safety/XAI layer | *(absent)* | No claim parser, verifier, grounding, uncertainty, reconciler, counterfactual audit, or clinical/factuality metrics exist anywhere in the tree. | Brief §XVI–§XVIII are unimplemented, not partially implemented. | `find . -name "*safety*" -o -name "*claim*"` → nothing | Build interfaces + baseline implementations; do not fake outputs. | claim schema, reconciliation rules, abstention |
| **Medium** | Distributed | whole repo | No DDP/FSDP/Accelerate anywhere. `run_medgemma_qlora.py` docstring says single-GPU by design. `VariantLLM` sets `device_map={"": torch.cuda.current_device()}` (`:699-701`) — correct for 1 GPU, unusable under DDP. | Brief §XIV two-GPU requirement unmet. | grep for `accelerate`/`DistributedDataParallel` → none | Accelerate + DDP entry point; keep `device_map` off the distributed path. | config smoke test (runtime unverifiable here) |
| **Medium** | Legacy `.cuda()` | `precompute_features.py:99,150`; `model/lavis/datasets/data_utils.py:47` | Hard-coded `.cuda()` in the Stage-1 feature-cache path. | Ignores `CUDA_VISIBLE_DEVICES` device selection; breaks any non-default-device use. | grep shown above | Route through an explicit `device` argument. | device-selection unit test |
| **Medium** | Wildcard imports | `pretraining/train.py:31-35`; `model/lavis/__init__.py:15-18`; `precompute_features.py:49` | `from … import *`. In LAVIS these are load-bearing (they populate the registry), so they cannot simply be deleted. | Brief §VII/§XXVI. Namespace pollution; unclear provenance. | grep shown above | Replace with explicit `import … # noqa` registry-registration calls, or an explicit `register_all()`. | registry populated without `*` |
| **Low** | Package layout | `training/test_stage2_utils.py` | A test file lives inside the training package. | Brief §XXVI. Ships test code with the package. | `ls training/` | Move to `tests/unit/`. | suite still green after move |
| **Low** | Tooling absent | repo root | No `pyproject.toml`; `ruff`/`mypy` not installed in `/home/phuong/venv`. | §XX/§XXI commands cannot run today. | `which ruff mypy` → nothing | Add `pyproject.toml`; install dev extras. | `ruff check` clean |
| **Low** | Dir naming | `preporcessing/` | Misspelled, referenced by docs/commands. | §XXII rename with shim. | `ls` | Rename + compatibility shim + migration note. | old import path still works with `DeprecationWarning` |

## Things that are already correct — preserve these

These were verified in the source and must survive any refactor (brief §XIII):

- **Soft-token fail-fast is real.** `SoftTokenEmbeddingWrapper.forward` (`:596-615`) calls
  `validate_soft_token_batch`; the old `min(batch_idx, n-1)` clamp is gone and the
  comment at `:604-606` records why. `stage2_utils.validate_soft_token_batch:138-156`
  raises on both batch mismatch and wrong placeholder count.
- **Exact prompt masking with fail-fast.** `masked_label_ids` (`stage2_utils.py:114-135`)
  raises when the full sequence does not start with the prompt tokens, instead of
  silently training on user/image tokens.
- **Target truncation fails closed.** `encode_train_example:981-982` raises
  `"target was completely truncated"` rather than training on an all-`-100` row.
- **Partial accumulation window handled.** `accumulation_window_size` (`:159-168`) +
  `train_fine:1159-1160` divide the tail window by its real size.
- **Per-epoch resumable checkpoints.** `train_fine:1179-1204` saves optimizer, scheduler,
  epoch, global_step, best_val, bad_epochs, data-generator state, CPU and CUDA RNG state.
  (Note: `CLAUDE.md` still claims "train_fine only saves the adapter after the last epoch"
  — that statement is **stale**; the code saves every epoch.)
- **Threshold argmax fallback.** `select_threshold_class:42-65` always returns a class.
- **Language-only LoRA discovery by name.** `language_lora_target_names:171-196` — a type
  check would find nothing under bitsandbytes `Linear4bit`.
- **Private-bucket validation treats missing metadata as unsafe.** `private_bucket_violations:214-261`.
- **Upload allow-list.** `run_medgemma_qlora.upload_safe_run:175-197` enumerates filenames
  explicitly and never walks the run root.
- **Predictions carry no identifiers by default.** `safe_prediction_row:68-77`;
  `evaluate_variant` writes to a `.sensitive_predictions` cache that the uploader skips.
- **Test split is held out of selection.** `run_medgemma_qlora.py:456` records
  `test_used_for_selection: false`; test generation happens once, after training.
- **Leakage is re-checked on the CSVs the run consumes**, not only at preprocessing time
  (`run_medgemma_qlora.build_native_records:343`).
- **`medgemma_direct` data path is already fully decoupled** — `training/dataio/manifest.py`
  imports only pandas and touches no Stage-1 code. The remaining coupling is the *model*
  path (the `fig9` import), not the data path.

## Open architectural question — blocks the §IV.A work

The brief (§IV.A) requires one `shared_visual_tokens` feeding both MHCAC and META-Former.
The current code does not do that. But the published paper supports the current design as
much as the required one, so I will not change it on my own reading:

- **For the brief's reading** — abstract: *"These multi-encoder features serve as a **shared
  representation** and are utilized in two key components … the META-Former module … and the
  MHCAC module"* (`docs/thesis/thesis.md:7`).
- **For the current code** — the paper's own ablations require the streams to stay separable:
  *"We evaluate the classification performance by **selectively activating one encoder at a
  time** during inference **within the MHCAC module**"* (`thesis.md:98`), and Figure 8 shows
  *"attention heatmaps from the MHCAC module … **across all three encoders**"* (`thesis.md:133`).
  A single pre-merged token sequence makes both of those impossible to produce.
  "Multi-**Head** Cross-Attention" plausibly means one attention head group per encoder.

Note the current code does **not** commit the failure mode the brief names ("MHCAC chỉ nhận
một encoder") — MHCAC receives all enabled encoders, just unmerged.

**Cost of changing it is unusually low right now**: `find . -name "*.pth"` returns nothing and
`CLAUDE.md` records that no Stage-1/Stage-2 run has executed on a GPU, so there is no trained
checkpoint to invalidate. If this is going to change, now is the cheapest moment.

## Not yet verified

- Everything in `model/lavis/` other than `blip2_qformer.py`'s encode path.
- `inference.py` (670 lines) — only grepped; it still runs Vicuna-7B and uses
  `device_map={"": 0}` at `:312`.
- Anything requiring a GPU, MIMIC images, or the split CSVs: none are present in this
  environment, so no training, generation, or metric number in this repo has been reproduced.
