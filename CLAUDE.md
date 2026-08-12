# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this checkout is

`Meta-CXR-source/` is the **current** source repo (remote `git@github.com:minhphuong150505/Meta-CXR.git`,
branch `main`). It is a sibling of the older `../META-CXR/` checkout. The parent
`../CLAUDE.md` describes that older layout — where it disagrees with this file
(no `stage2/`, `safety/`, `runtime/`, `scripts/`, `training/evaluation/`,
`medgemma_inference/`; a `Meta-CXR-Kaggle` remote; Stage-2 test counts), **this
file wins for work done inside this directory.**

Current training target, confirmed by the user on 2026-08-12, is one host with
login identity `phuong@minhphuong`. Its GPU count/model, VRAM, RAM and data mounts
have not been verified. Treat the L4 and 2×3090 documents as supported recipes,
not as descriptions of the current host; run `scripts/vm_preflight.py` there
before choosing a training config.

`README.md` is authoritative and detailed (written in Vietnamese) — read it before
making claims about pipeline status. GPU evidence is limited: the tracked Table 5
Stage-1 **inference-only encoder ablation** completed 4/4 configurations on the
full test split, but this does not validate current Stage-1/Stage-2 training or
reproduce Stage-2 metrics. Do not generalize that evaluation result into a claim
that either training pipeline is GPU-validated.

## Commands

```bash
# CPU test suite (479 tests). 5 fail on a box without torchvision/transformers:
# test_native_independence (4) and test_stage1_eval_hook (1) import model.lavis.
# test_blip2_negative_sampling cannot even be collected without torchvision.
CUDA_VISIBLE_DEVICES="" python -m pytest tests/ -q
python -m pytest tests/test_stage2_prompts.py -q          # one file
python -m pytest tests/test_stage2_prompts.py -q -k negative_policy   # one test

# Syntax check for the stdlib-only packages
CUDA_VISIBLE_DEVICES="" python -m compileall -q \
    stage2 training scripts runtime safety tests medgemma_inference

# Lint (config in pyproject.toml; ruff lint only, no formatter pass)
ruff check .

# Pre-commit — the notebook privacy guard is mandatory, see "Data handling"
pip install pre-commit && pre-commit install

# Preflight before any GPU run (checks CUDA, RAM, disk, shm, paths, HF auth)
python scripts/vm_preflight.py --stage 1

# Stage 1
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --standalone --nproc_per_node=1 \
    -m pretraining.train --cfg-path pretraining/configs/mimic_cxr_full_l4.yaml
# 2×3090 DDP variant: --nproc_per_node=2 --cfg-path .../mimic_cxr_2x3090.yaml
# Smoke: set run.truncate_train / truncate_val / truncate_test in the YAML.

# Stage 2 (single-GPU only — no DDP anywhere in Stage 2)
CUDA_VISIBLE_DEVICES=0 python training/run_medgemma_qlora.py \
    --train-limit 500 --val-limit 10 --test-limit 10 --no-upload --output-dir training/outputs/smoke

# Evaluation — calibrate on validation only, then score the test split
python scripts/calibrate_thresholds.py --predictions <val.npz> --objective f1 \
    --uncertain-policy ignore_uncertain --min-positive 20 --output <thresholds.json>
python scripts/evaluate_stage1.py --predictions <test.npz> --thresholds <thresholds.json> --output-dir <dir>
python scripts/evaluate_stage2.py --predictions <reports.jsonl> \
    --metrics bleu,rouge,meteor,cider,bertscore --skip-clinical-metrics --output-dir <dir>

# Manifest invariants (split leakage, required columns, section targets)
python -m training.dataio.validate_manifest --section-mode findings_and_impression
```

Environments: the cloud setup recommends **two separate venvs** to isolate the
Stage-1 workflow from the heavier Stage-2 QLoRA extras. The current lock files are
additive, not conflicting: `requirements-stage2.txt` includes
`requirements-stage1.txt` and adds accelerate/bitsandbytes/PEFT. `pyproject.toml`
deliberately declares zero runtime dependencies; it exists for tooling config
only, and the repo is not a src-layout package (modules import by path from the
repo root).

First run of anything: `cp configs/env_config.yaml.example configs/env_config.yaml`
and fill in paths — `local_config.py` raises `FileNotFoundError` otherwise.

## Architecture

Stage 1 (representation + classification) and Stage 2 (report generation) are
**deliberately decoupled**, and preserving that decoupling is the single most
load-bearing design constraint in the repo.

### Stage 1 — `pretraining/train.py`

Entrypoint registers LAVIS components via star imports, then hands off to the
vendored fork in `model/lavis/`. Data flow:

```
study (not image) → anchor + ≤1 auxiliary view
  → frozen encoders (BioViL-T 1408 + PubMedCLIP 768 + SwinV2; RadDINO off)
  → per-encoder ViewFusionModule on RAW pre-projection output  (mhcac/view_fusion.py)
  → FC projections → 1408, concatenated on the token axis
  → MHCAC (mhcac/mhcac_12.py): 14 abnormalities × {Positive, Negative, Uncertain}
       student = image only (this is inference); teacher = image + report text,
       TRAIN ONLY, distilled into the student. Text attention is teacher-gated.
  → Q-Former, 32 query tokens, cross-attn every 2nd block; ITC (1024-sample
       negative queue) + ITM + LM
```

- Sampling is **one row per study**, not per image (`study_sampling: true`).
- `ViewFusionBlock` zero-inits `W_O` and the last FFN Linear, so it is an exact
  identity at step 0 and a single-view checkpoint loads without regression.
  Studies with no auxiliary view are gated to zero, not dropped from the batch.
- `mhcac/loss.py` holds every loss; `ClassificationLoss` takes a `sample_mask` so
  unlabelled rows contribute nothing. `soft_target_kl_loss` detaches the teacher.
- Production config `mimic_cxr_full_l4.yaml`: 20 epochs, early stop patience 5,
  `selection_metric: macro_auprc` on **validation only**, bf16 AMP, `save_freq: 5`,
  `warmup_steps: 300` counted in **optimizer updates, not microbatches**.
  Thresholds are calibrated post-hoc from `checkpoint_best` validation logits.
  The test split is held out of checkpoint selection entirely.
- Note the `data:` block must sit **inside** `model:` — `Config` merges only
  `run`/`model`/`datasets`.
- `mimic_cxr_2gpu.yaml` is legacy (`multi_view: false`, `warmup_steps: 32000`
  which never finishes its ramp). Don't copy from it.

### Stage 2 — `training/run_medgemma_qlora.py`

`google/medgemma-1.5-4b-it`, 4-bit NF4 QLoRA, single process / single GPU,
checkpoint selected by validation cross-entropy.

`training/pipeline_modes.py` is stdlib-only and names each architecture explicitly
(the old `--image-mode {native,qformer,both}` was renamed because it described an
implementation detail and made the hybrid easy to mislabel as "native MedGemma"):

| Mode | Stage 1? | Visual path |
|---|---|---|
| `medgemma_direct` (default) | no | MedGemma's own image tower + projector |
| `meta_cxr_qformer` | yes | Q-Former soft tokens |
| `meta_cxr_qformer_with_mhcac_prompt` | yes | soft tokens + structured P/N/U text |
| `text_only_language_prior_ablation` | no | none — the only mode with `requires_multimodal=False` |
| `both_for_ablation` | — | runs direct then qformer sequentially |

Two further modes are inference-only against external checkpoints:
`pretrained_medgemma_findings_first` (routes through `medgemma_inference.run_pretrained_findings`,
not the fine-tuning CLI) and `pretrained_medgemma_impression_phase2` (declared but
disabled by a runtime guard).

**The independence invariant, enforced by `tests/test_native_independence.py`:**
every LAVIS/Stage-1 import lives in `training/stage1/lavis_loader.py` and nowhere
else. A Stage-2 entrypoint must never import it at module scope — only inside the
branch that has already decided it needs Stage 1. `training/dataio/manifest.py`
reads the split CSVs with pandas alone for the same reason.

Q-Former modes are **`findings_only`**; the native route also supports
`impression_only` and `findings_and_impression` (the default). The run errors out
rather than silently substituting one section for the other.

Soft-token conditioning **substitutes** projected Q-Former vectors at
`<qformer_soft_token>` positions — it does not sum into them
(`training/medgemma/soft_tokens.py`). Getting the per-row indexing wrong is
silent: loss still falls, but every study is described using another study's
image. Hence per-row shape validation and fail-closed behavior.

### Prompt v2 — `stage2/prompts/`

`PromptBuilder` is the single prompt entry point for train **and** inference, and
touches no model, tokenizer or torch, so parity is byte-for-byte testable. It
emits `PromptPart`s plus a prompt version, config hash and template hash recorded
in artifact metadata. Five visual modes in `schemas.py`: `native_anchor_only`,
`native_anchor_guided`, `native_multiview`, `qformer_visual_only`, `qformer_guided`.
Only guided modes see structured Stage-1 predictions, and they are phrased as
fallible auxiliary cues, never ground truth — `qformer_visual_only` receives no
labels at all, which is what keeps the ablation uncontaminated.

Opt-in via `--prompt-config configs/stage2_prompt_v2.yaml`; without the flag the
legacy prompt is used. The prompt prefix is masked out of training labels, and
the soft token is added to `bad_words_ids` at generation.
`configs/prompt_ablation/P1..P9.yaml` drive `scripts/run_prompt_ablation.py`.

### Evaluation — `training/evaluation/`, driven by `scripts/evaluate_stage*.py`

There is **no top-level `evaluation/` directory**; older docs that cite one are
stale. The core (classification metrics, AUROC/AUPRC, threshold calibration,
bootstrap, BLEU/ROUGE, error analysis) needs only numpy, so it runs wherever the
tests do. Plots need the `eval-plots` extra; METEOR/CIDEr/BERTScore need
`eval-generation`.

Clinical metrics (CheXbert, RadGraph, RadCliQ, RadFact) are **deliberately not
installable extras** — they're research code behind separate licences, not
reproducible pins. `training/evaluation/clinical.py` raises
`MissingOptionalDependency` naming the package, or `NotImplementedError` if the
package is present but the adapter was never validated against published
reference scores. **A missing clinical metric is reported as unavailable, never
as a score of 0**, and lexical metrics must not be presented as clinical accuracy.

### `safety/` and `runtime/` (both stdlib-only)

`safety/pipeline.py` orchestrates draft report → parsed claims → verification →
final report or abstention; it holds no verification logic itself so a real
phrase-grounding model can be swapped in via the same protocol. Its output record
carries no `subject_id`/`study_id`/`dicom_id`/path/reference text, so it is safe
to persist. `parse_coverage` is surfaced on purpose: a pipeline that parsed 2 of
12 sentences has not checked the report however clean its numbers look.

`runtime/budget.py` bills wall-clock time against an hourly rate (a stalled GPU
costs the same as a busy one) and carries `prior_elapsed_seconds` so resumes
cannot reset the ceiling. It only ever stops a run — it never downgrades the
model or enables extra sections. `runtime/device.py` resolves device/dtype from
config or the machine; nothing hardcodes `cuda:0`.

## Data handling — non-negotiable

MIMIC-CXR is PhysioNet credentialed data under a DUA that forbids redistribution.
This remote is public.

- Never commit images, report text, processed split CSVs, feature caches,
  prediction JSONL, credentials, or model weights. `.gitignore` covers these
  broadly (`data/`, `Report/`, `*.npz`, `*.jsonl`, `checkpoints/**`, `*.pth`, …).
- **Executed notebooks are the easy leak** — their outputs embed `subject_id`,
  `study_id` and report text. `scripts/check_notebook_privacy.py` runs as a
  pre-commit hook; do not bypass it.
- `configs/env_config.yaml` is git-ignored here (unlike the old checkout). Edit
  `configs/env_config.yaml.example` for anything shared.
- `cloud/env.sh` holds no real project/bucket names. Export them from an
  untracked `cloud/env.local.sh` before running the launchers.
- `image_path` in the processed CSVs is **relative** (`files/p1X/pXXXXXXXX/sYYYYYYY/<dicom>.jpg`)
  and is joined onto `mimic_cxr_jpg_root`, which must point at a directory
  directly containing `files/`. Do not rewrite these to absolute paths.

## Conventions

- `tests/conftest.py` registers `model` and `model.lavis` as **path-only**
  packages so submodules resolve without executing `model/lavis/__init__.py`,
  which would drag in the whole GPU stack. Without it the suite cannot be
  collected on a CPU box. It also stubs `timm.models.hub` when timm is absent.
  When a CPU test needs a GPU-only import, stub it here — do not pip-install into
  the CPU venv.
- `model/lavis/` is a modified fork of Salesforce LAVIS and is excluded from ruff:
  reformatting it would make every future upstream diff unreadable. Same for
  `mhcac/mhcac_8..11.py` (legacy variants; only `mhcac_12.py` is wired).
- `preporcessing/` is misspelled in the tree. Leave it.
- Modules under `training/` carry a dual import shim
  (`try: from stage2_utils ... except ImportError: from training.stage2_utils ...`)
  so they work both as scripts and via `python -m`. Match it in new files there.
- `inference.py` is the legacy Vicuna-7B + LoRA Gradio path and has **not** been
  migrated to MedGemma. Two functionally identical copies of BioViL-T exist
  (`biovil_t/`, `vision_encoders/biovil_t/`).
- The many `docs/*audit*.md` / `*_baseline.md` files are point-in-time records of
  past integration work, not living specs. `docs/VM_TRAINING_FINAL.md` and
  `docs/STAGE2_PIPELINE_MODES.md` are the ones to actually follow.

## Source Documentation Synchronization

`struct/` is the persistent source-code knowledge base for this repository and is
tracked in Git. Behavioral source changes and their affected `struct/` pages must
be committed together.

**Before modifying source code:**

1. Read `struct/HOME.md`.
2. Read the documentation of the components you are about to touch —
   `struct/project/<dir>/_index.md`, then `<file>.py.doc.md`, then
   `<file>.py.methods/<fn>.md`.
3. Read `struct/project/_meta/DECISIONS.md` when architectural decisions are
   relevant. It records which components are active, legacy, conditional, or
   still unclassified, and why. Do not re-derive those conclusions.
4. Check `struct/project/_meta/ACTIVE_COMPONENTS.md` and `LEGACY_AND_OPTIONAL.md`
   before assuming a file is dead. "No static import" does not mean "unused" —
   this repo uses CLI entrypoints, registries, YAML config, shell scripts and
   config-gated branches.

**After modifying source code:**

1. Update the corresponding directory documentation (`_index.md`).
2. Update the file documentation (`<file>.doc.md`).
3. Update affected method documentation (`<file>.methods/`).
4. Update caller/callee relationships, in both directions.
5. Update config documentation if behavior changed.
6. Update `ARCHITECTURE.md` / `DATA_FLOW.md` / `CALL_GRAPH.md` if necessary.
7. Add documentation for new files and functions.
8. Remove or re-label documentation for deleted or renamed components.
9. Update the source tree in `struct/HOME.md` when repository structure changes.
10. Verify all relative Markdown links still resolve.

Only update what actually changed. Do not rewrite all of `struct/` on every edit.

A code change that changes behavior is not complete until the relevant `struct/`
documentation is synchronized.

**Source code remains the final source of truth.** If `struct/` conflicts with the
code, inspect the code and update `struct/` — never the other way round. Known
doc-vs-code conflicts are already recorded in
`struct/project/_meta/LEGACY_AND_OPTIONAL.md` under "Potential issues".

**Never write patient data into `struct/`** — no `subject_id`, `study_id`,
`dicom_id`, real image paths, or report text, in any documentation page.
