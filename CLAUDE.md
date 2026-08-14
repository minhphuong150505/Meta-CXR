# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this checkout is

`Meta-CXR-source/` is the **current** source repo (remote `git@github.com:minhphuong150505/Meta-CXR.git`,
branch `main`). It is a sibling of the older `../META-CXR/` checkout. The parent
`../CLAUDE.md` describes that older layout — where it disagrees with this file
(no `stage2/`, `safety/`, `runtime/`, `scripts/`, `training/evaluation/`,
`medgemma_inference/`; Stage-2 test counts), **this file wins for work done
inside this directory.**

## The training host — there is only one, and it is local

All training runs on **`phuong@minhphuong`**, the user's own machine. Verified
over SSH on 2026-08-13:

| | |
|---|---|
| GPU | 1× NVIDIA **RTX 5060 Ti, 16 GB** (`nvidia-smi`), driver 580.173.02, CUDA 13.0 |
| `/` | 58 GB, ~5 GB free — tight, do not install into it casually |
| `/home` | 185 GB, ~19 GB free |
| Data + checkpoints | `/mnt/drive1tb` — partition `nvme1n1p2`, **930 GB NTFS**, **not in `/etc/fstab`** |
| Repo checkout there | `~/Documents/2026/KLTN/Code_github/META-CXR-full-smoke-git` |

Two consequences worth remembering:

- `/mnt/drive1tb` does **not** auto-mount — it is not in `/etc/fstab`. After a
  reboot every path in `configs/env_config.yaml` dangles until it is mounted by
  hand. If a run fails with missing CSVs or images, check the mount before
  debugging anything else. Do not reboot mid-run.
- The host has no passwordless sudo, so an agent on SSH cannot mount it. Ask
  the user.
- **It is a Windows system partition**, not a data disk: `Windows/`,
  `Program Files/`, `pagefile.sys`, `hiberfil.sys` sit next to the 573 GB
  dataset. Treat write access as consequential.
- **The volume has real NTFS errors.** `dmesg` on 2026-08-13:
  `ntfs3(nvme1n1p2): Mark volume as dirty due to NTFS errors` /
  `It is recommended to use chkdsk.` This is not a stale hibernation flag; the
  kernel driver hit errors and flagged the volume itself. Only `chkdsk` from
  Windows fixes it — `ntfsfix` clears the dirty bit without repairing anything.
  The kernel `ntfs3` driver refuses rw while it stands; `ntfs-3g` (FUSE) mounts
  it anyway, which is how the drive is currently writable.
- **Driver choice does not affect training throughput.** Measured 2026-08-13,
  200 iterations at batch 6: kernel `ntfs3` 0.5251 s/it vs FUSE `ntfs-3g`
  0.5277 s/it, +0.5%, which is 0.3 h across a 54 h run. The workload is
  GPU-bound, not I/O-bound. Do not switch drivers for speed; the only reason to
  prefer `ntfs3` is that it refuses to write to a damaged volume.

### The venv

`~/.venvs/meta-cxr-stage1-311/bin/python` — torch 2.9.1+cu129, torchvision
0.24.1, transformers 4.53.2. It is the only environment on the host with torch,
and the RTX 5060 Ti is **sm_120**, which needs cu12.8+.

`~/.venvs/meta-cxr-rtx4060` (torch 2.5.1+cu124, kernels only to sm_90) was
**deleted on 2026-08-14** — it was named after earlier hardware and could not
run this GPU. Do not recreate it. It cost a day: it fails late, after the whole
model has loaded, with `CUDA error: no kernel image is available for execution
on the device`, and before that it fails more confusingly still — transformers
4.53 refuses `torch.load` under torch < 2.6 (CVE-2025-32434), so PubMedCLIP
raises a vulnerability error and the arch mismatch never surfaces. If either
symptom ever reappears, check the interpreter before anything else.

`~/myenv` has no torch and is unrelated.

### Running anything means SSH-ing there

This directory is a **development checkout on a machine with no GPU and no
dataset**. Nothing that actually runs the project — training, evaluation,
inference, smoke tests, GPU-dependent scripts — runs here. SSH to the host:

```bash
ssh phuong@minhphuong
cd ~/Documents/2026/KLTN/Code_github/META-CXR-full-smoke-git
git pull origin main        # ALWAYS pull first — this checkout drifts behind
```

That checkout has the same `origin` (`git@github.com:minhphuong150505/Meta-CXR.git`,
branch `main`), so the workflow is: commit and push from here, then pull and run
there. **Always pull before running** — the host has repeatedly been one or more
commits behind, and running stale code silently produces results attributed to
the wrong revision.

What may still be done locally: reading code, editing, `struct/` updates, and the
CPU test suite. Everything else goes over SSH.

**There is no cloud path.** GCP, rented L4, Kaggle and 2×3090 recipes were all
deleted on 2026-08-13 to cut cost — the user trains locally. Do not reintroduce
`cloud/`, `docs/cloud/`, GCS upload flows, or hardware-named configs. Anything
in git history describing them is dead.

`README.md` is authoritative and detailed (written in Vietnamese) — read it before
making claims about pipeline status. GPU evidence is limited: the tracked Table 5
Stage-1 **inference-only encoder ablation** completed 4/4 configurations on the
full test split, but this does not validate current Stage-1/Stage-2 training or
reproduce Stage-2 metrics. Do not generalize that evaluation result into a claim
that either training pipeline is GPU-validated.

## Commands

```bash
# CPU test suite. On a box without torchvision/transformers (verified 2026-08-14):
#   546 passed, 5 failed, 1 skipped, and 2 modules excluded before collection
#   — test_blip2_negative_sampling.py and test_encoder_ablation.py, both of which
#   import model.lavis and therefore torchvision. Collection errors abort the run,
#   so ignore them explicitly to see the real result:
CUDA_VISIBLE_DEVICES="" python -m pytest tests/ -q \
    --ignore=tests/test_blip2_negative_sampling.py --ignore=tests/test_encoder_ablation.py
# The 5 failures are test_native_independence (4: missing private env config)
# and test_stage1_eval_hook (1: missing torchvision). They are unchanged baseline.
CUDA_VISIBLE_DEVICES="" python -m pytest tests/test_explanation_metrics.py -q  # 7 passed
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

# Stage 1 — the only supported recipe. Single GPU; there is no DDP variant left.
# Launch it PLAIN, not through torch.distributed.run: with one GPU and
# run.distributed=false, torchrun only sets RANK/WORLD_SIZE, which sends
# init_distributed_mode down its distributed branch for no benefit. This plain
# form is what produced the earlier checkpoints (wandb metadata: program =
# "-m pretraining.train"). torchrun does work now that run.dist_url is present,
# but it buys nothing here.
CUDA_VISIBLE_DEVICES=0 python -m pretraining.train \
    --cfg-path pretraining/configs/mimic_cxr_full.yaml \
    --options run.batch_size_train=6 run.batch_size_eval=6 run.accum_grad_iters=11
# Those three overrides are what the earlier 10-epoch run used on the 5060 Ti.
# The YAML's own defaults (batch 8 / accum 8) also fit in 16 GB.
#
# ALL PRE-2026-08-14 CHECKPOINTS WERE DELETED on 2026-08-14, at the user's
# request, on the grounds that those runs went in the wrong direction. 15 files,
# 39 GB. Nothing on disk predates the current recipe. They were unloadable
# against it anyway: encoders.swin went false, so MHCAC saw 98 visual tokens
# instead of 147. The Table 5 numbers survive in results/ but cannot be
# reproduced or extended without retraining, because the checkpoint they were
# computed from is gone.
#
# A SECOND, INDEPENDENT BREAK landed 2026-08-14: each encoder now keeps its
# native token sequence (246 tokens, not 98) and every stream has its own
# positional encoding, so pos_enc went from one Parameter to a ModuleDict and
# cnn_downsampler left the state dict. Nothing from before that commit loads.
# Smoke: set run.truncate_train / truncate_val / truncate_test in the YAML.

# Stage 2 (single-GPU only — no DDP anywhere in Stage 2)
CUDA_VISIBLE_DEVICES=0 python training/run_medgemma_qlora.py \
    --train-limit 500 --val-limit 10 --test-limit 10 --no-upload --output-dir training/outputs/smoke

# Evaluation — calibrate on validation only, then score the test split
python scripts/calibrate_thresholds.py --predictions <val.npz> --objective f1 \
    --uncertain-policy ignore_uncertain --min-positive 20 --output <thresholds.json>
python scripts/evaluate_stage1.py --predictions <test.npz> --thresholds <thresholds.json> --output-dir <dir>
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_explanation.py \
    --checkpoint <checkpoint_best.pth> --cfg-path pretraining/configs/mimic_cxr_full.yaml \
    --split test --mask-cache-dir /mnt/drive1tb/datasets/explanation_masks \
    --output-dir /mnt/drive1tb/private-results/xai --export-figures 12
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
  → frozen encoders (BioViL-T 1408 + PubMedCLIP 768; SwinV2 and RadDINO off)
       BioViL   448 in  → 14x14 = 196 tokens          local, 32 px per cell
       PubMedCLIP 224 in → 1 CLS + 7x7 = 50 tokens    global + regional, 64 px
  → per-encoder ViewFusionModule on RAW pre-projection output  (mhcac/view_fusion.py)
  → FC projections → 1408, concatenated on the token axis (246 tokens)
  → MHCAC (mhcac/mhcac_12.py): 14 abnormalities × {Positive, Negative, Uncertain}
       student = image only (this is inference); teacher = image + report text,
       TRAIN ONLY, distilled into the student. Text attention is teacher-gated.
  → Q-Former, 32 query tokens, cross-attn every 2nd block; ITC (1024-sample
       negative queue) + ITM + LM
```

- Sampling is **one row per study**, not per image (`study_sampling: true`).
- **Every encoder keeps its native scale; nothing is pooled or dropped on the way
  into MHCAC.** This is the point of running two of them, and until 2026-08-14
  the code did the opposite: `cnn_downsampler` squeezed BioViL 14x14 → 7x7, and
  `_resize_patch_sequence` deleted PubMedCLIP's CLS (`50 == 49+1`), leaving two
  interchangeable coarse maps and no global token. Measured on real studies:
  CLIP's CLS is not recoverable from the patches that were kept
  (`cos(CLS, mean patch) = 0.21`), whereas BioViL's own global output *is* the
  patch mean (`cos = 1.0000`, `biovil_t/model.py:84`) — so the code was throwing
  away the irreplaceable one and keeping the redundant one.
  `Blip2Qformer._native_stream_layouts` now hands MHCAC a `StreamLayout` per
  encoder; `stream_layouts=None` restores the old path and is what swin/raddino
  recipes get. Cost of the extra 148 tokens: **0.2543 s/it vs 0.2529, +0.6%**.
  Pinned by `tests/test_stream_layouts.py`.
- **PubMedCLIP patch tokens are read through `post_layernorm` and then have their
  per-image mean subtracted.** HF returns the vision tower's `last_hidden_state`
  *without* the final LayerNorm (`modeling_clip.py:763-765` applies it to the
  pooled CLS only), and the raw residual stream has a fixed DC direction that is
  97% constant across the dataset: mean pairwise cosine between the 49 patches
  was **0.674** against BioViL's 0.0017, so attention over them came out nearly
  flat and the whole stream acted as a constant bias. LayerNorm alone only
  reaches 0.587; subtracting the mean reaches **-0.014**. Keep the split — token
  0 carries the global view, the 49 patches carry local deviation from it.
  Do **not** feed this encoder 448: measured worse (0.714) and it puts CLIP's
  interpolated position embeddings out of distribution while frozen.
- **`model.image_size` must equal `vis_processor.*.image_size` (448).** It is not
  self-evident: `init_vision_encoder` ignores it for biovil, so nothing read the
  value and the BLIP-2 default of 224 sat in the model config unnoticed. It now
  sets BioViL's token grid. Two guards, both fail loud — a `ValueError` naming
  the key when it is missing, and a token-count check in `MHCAC.forward`.
- **Positional encodings are per-encoder and init at `std=0.02`.** Tokens reach
  them L2-normalised to 1.0, so the old bare `torch.randn` gave the encoding norm
  27.8 and `cos(token+pos, pos) = 0.999` — at step 0 attention was driven purely
  by position. One shared encoding also meant BioViL token *k* and PubMedCLIP
  token *k* received the identical position vector.
- `ViewFusionBlock` zero-inits `W_O` and the last FFN Linear, so it is an exact
  identity at step 0 and a single-view checkpoint loads without regression.
  Studies with no auxiliary view are gated to zero, not dropped from the batch.
- `mhcac/loss.py` holds every loss; `ClassificationLoss` takes a `sample_mask` so
  unlabelled rows contribute nothing. `soft_target_kl_loss` detaches the teacher.
- Production config `mimic_cxr_full.yaml`: **10 epochs**, `selection_metric:
  macro_auprc` on **validation only**, bf16 AMP, `save_freq: 5`, `warmup_steps:
  300` counted in **optimizer updates, not microbatches**. Thresholds are
  calibrated post-hoc from `checkpoint_best` validation logits. The test split is
  held out of checkpoint selection entirely.
- **The recipe is classification-only as of 2026-08-13.** `lambda_itc/itm/lm`
  and `lambda_teacher_cls/distill` are all `0.0`; the objective now matches
  upstream META-CXR (`cls + 0.3*contrastive + 0.7*orthogonality + 0.3*sparsity`)
  plus the multi-view terms. `forward()` **skips** the Q-Former and text-encoder
  passes entirely when every weight reading them is zero. Measured on the host,
  200 iterations at batch 6: **0.5251 s/it against 0.8196 s/it** with the
  vision-language losses on — 1.56x, ~54 h instead of ~84 h for 10 epochs.
  **Consequence: the Q-Former gets no gradient**, so the checkpoint serves
  Stage-1 classification only and is NOT valid for Stage-2 `meta_cxr_qformer`
  modes. `medgemma_direct` is unaffected. Covered by
  `tests/test_loss_weight_gating.py`.
- **`run.eval_start_epoch: 5` — the first five epochs train without being
  scored.** Validation over the full split is expensive and the early epochs are
  never the ones selected, so skipping them buys wall-clock time. The knob counts
  epoch *indices*, matching the training log: `epoch: [5]` is the sixth.
  Because `checkpoint_best` is only written inside the evaluation branch, the
  same knob guarantees no unscored epoch can ever be selected. Default 0
  restores the historical behaviour.
- **Early-stop patience counts scored epochs only.** The window is clamped to
  open at `eval_start_epoch` (`cur_epoch - max(best_epoch, eval_start_epoch)`),
  because `best_epoch` initialises to 0: measuring from it alone would spend the
  whole budget on unscored epochs and kill the run on its first scored epoch,
  logging "early stopping", which reads like convergence. With the shipped
  values (10 epochs, eval from [5], patience 5) the earliest possible stop is
  [10] and the last epoch is [9], so **early stopping cannot fire** — inert, not
  broken. Set patience to 4 or less to make it live. Covered by
  `tests/test_eval_start_epoch.py`.
- Note the `data:` block must sit **inside** `model:` — `Config` merges only
  `run`/`model`/`datasets`.
- `mimic_cxr_2gpu.yaml` was deleted with the retired cloud/Kaggle recipes. Do
  not recreate or copy it from history; `mimic_cxr_full.yaml` is the only
  supported Stage-1 recipe.

#### Explanation-aware loss and XAI evaluation

Phase 1–3 added an optional Grad-CAM constraint, two-tier private mask cache and
XAI evaluator. **None of this explanation-aware path has run on GPU yet.** The
loss/math and cache pipeline have CPU tests; a real 200-study validation cache
was built on CPU, but neither a smoke/full train with the loss nor
`scripts/evaluate_explanation.py` has ever run against a checkpoint.

For each study with at least one Positive CheXpert label:

```text
s       = Σ_positive (logit_pos - logit_neg)²
α_c     = mean_ij ∂s/∂A_cij
H       = ReLU(Σ_c α_c A_c)
H_norm  = (H - min(H)) / (max(H) - min(H) + eps)
theta   = quantile(H_norm, 1 - top_k).detach()
H_plus  = H_norm * 1[H_norm >= theta]       # soft values, detached binary gate
L_exp   = 1 - Σ(H_plus * M) / (Σ H_plus + eps)
```

The soft values are essential **for the loss** so double backprop has a gradient;
the evaluation top-saliency metric deliberately uses the binary Eq. (5). CAMs
are supervised separately at BioViL 14×14 and PubMedCLIP/Swin 7×7, then averaged
as loss terms. Frozen encoders do not move; the loss reshapes the trainable
projection/MHCAC/head path that reads their features.

`model.loss.lambda_explanation` is the only enable/disable gate. There is no
separate `enabled` flag that can disagree with it:

```yaml
model:
  loss:
    lambda_explanation: 0.25   # 0.0 = completely off, including CAM capture
  explanation:
    top_k: 0.5
    warmup_start_epoch: 2
    warmup_epochs: 2
    streams: [biovil, pubmedclip, swin]
    mask_cache_dir: /mnt/drive1tb/datasets/explanation_masks
```

Production config remains **off** (`lambda_explanation: 0.0`, empty cache path)
until the GPU smoke is performed. The approved schedule at 0.25 is epoch [0]–[1]
0, [2] 0.125, [3] 0.1875, [4]+ 0.25. `RunnerBase.eval_epoch` is
`@torch.no_grad()`, so the training forward skips this loss during validation.
Do not remove that guard: Grad-CAM requires a live graph.

Mask data and verified traps (2026-08-14):

- CheXmask source:
  `/mnt/drive1tb/datasets/chexmask/MIMIC-CXR-JPG.csv`. The **real header is
  `dicom_id`**, not `Image ID` as the data dictionary claims. Use
  OriginalResolution, Dice mean >=0.7, union left/right lung, never heart.
- MS-CXR source:
  `/mnt/drive1tb/datasets/ms-cxr/MS_CXR_Local_Alignment_v1.1.0.csv`; schema is
  `dicom_id,category_name,label_text,path,x,y,w,h,image_width,image_height,split`.
  **Never use its `split` column.** At least 166 boxes marked `train` there belong
  to this project's test set. Project manifests are the sole split authority.
- Default private cache location is
  `/mnt/drive1tb/datasets/explanation_masks`; format is
  `masks_<split>.npy uint8 [N,112,112] {0,255}` plus
  `index_<split>.json` mapping identifier to row/source.
- Verified CPU smoke over 200 validation studies produced 193 valid masks: 189
  lung and 4 bbox. Lung coverage was 18.2–52.9% (median 32.6%); bbox union
  coverage 3.5–18.2%.
- PhysioNet restricted downloads require Basic auth **after** a 401 challenge.
  `curl -n` sends credentials preemptively and receives an uninformative 403;
  use `wget --user ... --ask-password`, not curl, for MS-CXR. CheXmask is open
  access but its cache is still a MIMIC-derived private artifact.

Build and evaluate only on the mounted private data host:

```bash
python preporcessing/build_explanation_masks.py --inspect
python preporcessing/build_explanation_masks.py \
  --split val --output-dir /mnt/drive1tb/datasets/explanation_masks

CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_explanation.py \
  --checkpoint <checkpoint_best.pth> \
  --cfg-path pretraining/configs/mimic_cxr_full.yaml --split val \
  --mask-cache-dir /mnt/drive1tb/datasets/explanation_masks \
  --ms-cxr-csv /mnt/drive1tb/datasets/ms-cxr/MS_CXR_Local_Alignment_v1.1.0.csv \
  --output-dir /mnt/drive1tb/private-results/xai --save-cams --export-figures 12
```

`evaluate_explanation.py` is intentionally separate from model-free
`evaluate_stage1.py`. It uses `model.eval()` with grad enabled and no optimizer,
reports each encoder stream with `lung` and `bbox` populations separate, and
marks lung annotation coverage unavailable. PNG/NPZ outputs are patient data;
the script refuses non-ignored repo destinations and never prints identifiers.

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

## The labels, the manifest, and how a checkpoint gets picked

Three things here are easy to get wrong and produce a run that looks healthy.

**A blank CheXpert cell is masked, not negative.** The export leaves a cell blank
when the labeler found no mention of the finding, which is not the radiologist
ruling it out. 79.4% of the label matrix is blank, so the old `.fillna(0)` made
roughly nine in ten "negatives" an absence of evidence. Blanks now carry
`ReportDataset.IGNORE_LABEL = -100` and are dropped per cell; `ClassificationLoss`
and the eval confusion matrix already kept only `labels >= 0`, so nothing
downstream needed changing. What it costs, measured on `full_allviews_v2`:
2.86 of 14 labels survive per study, 31% of studies keep exactly one, and the
imbalance **inverts** — positives are the majority for 12 of 14 findings
(Atelectasis: 44,718 positive against 1,502 negative). `default_class_weights` in
`blip2_qformer.py` were recomputed downward (0.18–2.01) and must be recomputed
again if the manifest or this policy changes. `No Finding` has zero negatives by
construction and is single-class under this policy; `include_meta_labels: false`
already keeps it out of macro metrics. Pinned by `tests/test_blank_label_masking.py`.

**The explanation-mask cache is built, and its coverage decides what you can
claim.** `/mnt/drive1tb/datasets/explanation_masks/` holds `masks_<split>.npy`
(uint8 `[N,112,112]`, values 0/255) plus `index_<split>.json` keyed by the anchor
`dicom_id`. Built 2026-08-14 from CheXmask + MS-CXR against `full_allviews_v2`;
the build takes ~25 minutes and streams the 13 GB CheXmask export, so do not
rebuild it casually. Verified: rows unique and in range, no empty or all-ones
mask, lung coverage p50 ≈ 33-35% on every split.

`build_explanation_masks.py` reimplements anchor selection instead of calling
`build_study_index`. The two agree exactly on real data — 0 cache keys that are
not dataset anchors, all 208,987 train keys found — but that is a checked fact,
not a guarantee. Recheck it if either side changes, because a divergence means
`_read_explanation_mask` silently returns `valid=False` and the term quietly
trains on less data than you think.

What the coverage allows:

| | train | val | test |
|---|---|---|---|
| studies with a mask | 93.8% | 93.5% | 90.5% |
| with >=1 positive finding | 60.6% | — | 77.6% |
| **both -> loss actually fires** | **56.7%** | — | 69.8% |
| **of those, real MS-CXR box** | **823** | **5** | **138** |

The cost is now measured, not estimated. 200 iterations at batch 6 on the
5060 Ti, same config and manifest, the term the only difference: **0.580 s/it
against 0.560, +3.6%**, peak VRAM 6,136 MiB against 6,112 (37% of 16 GB). A
10-epoch run is ~60 h with it and ~58 h without. The first estimate of 30-50%
was wrong because the Grad-CAM backward only traverses the MHCAC subgraph, not
the frozen encoders, and the term fires on 56.7% of studies. Evidence:
the only population that supports "the model looks at the pathology" is the box
one, and that is **138 test studies** — val's 5 cannot calibrate or select
anything. The lung-mask population is 30x larger but only supports the weaker
claim "the model stays inside the lungs"; `ExplanationSummary` keeps the two
apart on purpose.

**Use `processed/full_allviews_v2`, nothing else.** Two stale exports sit beside
it on the training host, and `meta-cxr-manifests-upgraded-20260806` was wired up
despite the name. Tell a stale export by any of: `extraction_method` is a single
constant (`legacy_preprocessed` appears nowhere in this repo), `target_valid` is
100% True (the length filter never ran), or `impression_valid` is missing (Stage 2
then dies in `assert_columns`). v2 reconciles exactly against the source —
377,110 rows = all of metadata, zero duplicate `dicom_id`, split identical to the
official MIMIC split, `findings_clean` empty iff `target_valid` is False, and
length bounds taken from train only.

**`selection_metric: loss`, and it has a known bias.** Validation loss is a
weighted sum dominated by the frequent labels, so a model that stops predicting a
rare finding entirely can outscore one that finds it sometimes. `macro_auprc` is
per-label and threshold-free and does not have that failure mode; it was replaced
because val is thin for two labels (Pleural Other 14 positives, Fracture 16).
Both are logged every scored epoch — check which epoch each would pick before
quoting either. `selection_mode` is deliberately absent from the YAML so
RunnerBase infers `min`; an explicit `max` left behind would keep the worst epoch.

## Data handling — non-negotiable

MIMIC-CXR is PhysioNet credentialed data under a DUA that forbids redistribution.
This remote is public.

- Never commit images, report text, processed split CSVs, feature caches,
  prediction JSONL, credentials, or model weights. `.gitignore` covers these
  broadly (`data/`, `Report/`, `*.npz`, `*.jsonl`, `checkpoints/**`, `*.pth`, …).
- **Never publish MIMIC-CXR or any derivative as a Kaggle Dataset, an open-data
  release, or under a licence such as CC0.** The PhysioNet DUA prohibits
  redistribution, and that covers cleaned reports, split CSVs, feature caches,
  predictions and checkpoints trained on them. This rule used to live in
  `configs/kaggle_datasets.yaml`; that file is gone, so it lives here now.
- **Executed notebooks are the easy leak** — their outputs embed `subject_id`,
  `study_id` and report text. `scripts/check_notebook_privacy.py` runs as a
  pre-commit hook; do not bypass it. `notebooks/` no longer exists, but the hook
  still guards any notebook added later.
- `configs/env_config.yaml` is git-ignored here (unlike the old checkout). Edit
  `configs/env_config.yaml.example` for anything shared.
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
  past integration work, not living specs. `docs/STAGE2_PIPELINE_MODES.md` is the
  one to actually follow.

## Keep CLAUDE.md and README.md current — this is part of the work

Whenever you fix something or add a feature, ask whether it changes anything
either of these two files claims, and update them **in the same commit** if so:

- **`CLAUDE.md`** (this file) — what an agent needs to not get it wrong: the
  training host and its quirks, commands that actually work, invariants, traps,
  which components are live vs dead. If you delete a path, remove the flag that
  used to reach it, change a default, or discover that a documented fact is
  false, fix it here. A stale line here misleads every future session.
- **`README.md`** — what a human needs: pipeline status, install, how to run
  each stage, what is and is not validated. Written in Vietnamese; keep it that
  way.

Rule of thumb: if a reader following these files would now do the wrong thing,
the change is not finished. This is not optional cleanup and not a separate
follow-up task — it ships with the code change, alongside the `struct/` update
described below.

Two habits that keep them honest:

- State evidence, not intent. "Verified over SSH on <date>" and "not yet run on
  GPU" are both useful; "should work" is not.
- When you remove something, say it is removed rather than deleting the line
  silently, if a future agent might otherwise try to reintroduce it.

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
