---
name: explanation-loss
description: The Stage-1 explanation-aware loss (weak CheXmask-lung + strong MS-CXR-bbox Grad-CAM terms) and the XAI evaluator. Load this when re-enabling lambda_explanation/lambda_explanation_strong, building or rebuilding the explanation mask cache, running scripts/evaluate_explanation.py, or interpreting saliency-precision numbers. Both lambdas are 0.0 in production as of 2026-08-17.
---

# Explanation-aware loss and XAI evaluation

This was section `#### Explanation-aware loss and XAI evaluation` of `CLAUDE.md`
until 2026-08-18. It moved here because the feature is switched off in
production, so the 206 lines no longer need to load in every session — but they
are the whole record of why it was turned off and what re-enabling it requires.

⚠ **Partly corrected 2026-08-31 — the SOURCES survived, and the cache has been
rebuilt.** This paragraph used to say the CheXmask and MS-CXR CSVs were gone.
They are not: both are on the data drive, which the 2026-08-17 reinstall did not
touch, and `CLAUDE.md`'s survival table was right where this file was wrong.
Verified on the host:

| | |
|---|---|
| `datasets/chexmask/MIMIC-CXR-JPG.csv` | present, 12,381 MB, header key is `dicom_id` |
| `datasets/ms-cxr/MS_CXR_Local_Alignment_v1.1.0.csv` | present, 311,892 bytes, 1,448 boxes |
| `datasets/explanation_masks/` (old cache) | present but **has no `masks_bbox_*`** — the 2026-08-14 build |

**The cache is rebuilt at `/home/phuong/explanation_masks_v2` (2026-08-31), and
it carries the bbox pair.** It had to go on `/home`: the data drive is now
mounted `ntfs3 ro`, so the old in-place location is not writable.

⚠ Two things were needed to make the documented build command work at all:
`configs/env_config.yaml` does not exist on a fresh host (create it from the
example — it is git-ignored), and
`preporcessing/build_explanation_masks.py` did not put the repo root on
`sys.path`, so running it by path died in `_project_manifest_paths` with
`ModuleNotFoundError: local_config` *after* `--inspect` had succeeded. Fixed.

Checkpoints the old A/B evaluator ran against are still gone; that part stands.


**OFF IN PRODUCTION as of 2026-08-17 — `lambda_explanation` and
`lambda_explanation_strong` are both `0.0` in `mimic_cxr_full.yaml`.** The user
retired the approach after the A/B below. The code, the mask cache and
`scripts/evaluate_explanation.py` are all kept and still work: the evaluator is
what produced the evidence, and it is the only way to re-test the idea after the
encoders are unfrozen. Do not delete them, and do not re-enable the lambdas
without reading the next four paragraphs.

**It did not help classification.** A controlled 5-epoch A/B on the host,
identical seed/manifest/recipe, the two lambdas the only difference; test split,
thresholds calibrated on val, `ignore_uncertain`:

| | ON (0.05/0.25) | OFF (0/0) |
|---|---:|---:|
| positive_macro_f1 | 0.8757 | **0.8767** |
| macro_auroc | 0.7850 | **0.7879** |
| macro_specificity | 0.3840 | **0.4127** |
| wall clock | 5:06:44 | **4:25:27** |

Every 95% CI overlapped; every difference favoured OFF; the term cost **+15.5%**
wall clock overall and +13.5/+17.0/+17.6% on epochs [2]/[3]/[4] where the lambdas
were live. (The +10.5% recorded elsewhere is the older *single pooled term*; the
split weak+strong version is dearer.)

**It also failed at its own objective, which is the stronger result.** `L_exp`
maximises saliency mass inside the mask — exactly what
`evaluate_explanation.py` measures. **The honest baseline for that number is the
mask area fraction, because a random CAM scores precisely that**; measured on the
test cache it is lung **0.3301**, bbox **0.2366**. Against that:

| stream | population | ON | OFF | random |
|---|---|---:|---:|---:|
| biovil | lung | 0.3692 | 0.3383 | 0.3301 |
| biovil | bbox | 0.2552 | 0.2533 | 0.2366 |
| pubmedclip | lung | 0.3810 | 0.4085 | 0.3301 |
| pubmedclip | bbox | 0.1769 | 0.2021 | 0.2366 |

One stream moved +0.031 the right way; the other moved ~0.026 the **wrong** way
on both populations and sits **below random** on bbox. CAMs are at chance in
**both** arms — including the arm that was never constrained — so the limit is
the representation, not the loss. **With the encoders frozen this term can only
re-weight channels of a fixed feature map; it cannot teach an encoder to look
elsewhere.** That is the mechanism to fix first.

⚠ **Never quote a saliency precision without the mask-area baseline beside it.**
0.25 reads like a result and is chance. `evaluate_explanation.py` does not yet
emit the baseline itself — compute it from the cache
(`(masks > 0).mean(axis=(1,2))`).

**Baselines for the 2026-08-31 rebuild, measured rather than assumed:**

| split | lung N | lung baseline | lung p50 | bbox N | bbox baseline | bbox p50 |
|---|---:|---:|---:|---:|---:|---:|
| train | 208,987 | 0.3466 | 0.3435 | 928 | 0.1555 | 0.1265 |
| val | 1,690 | 0.3520 | 0.3457 | 7 | 0.0730 | 0.0713 |
| test | 2,959 | **0.3314** | 0.3239 | 188 | **0.2366** | 0.2284 |

The test figures reproduce the previously recorded 0.3301 / 0.2366 — bbox to the
digit — which is the strongest available evidence that the rebuild is equivalent
to the cache the A/B was run against. Integrity checked on every split: uint8
`[N,112,112]`, values exactly `{0,255}`, rows unique and in range, **zero** empty
and **zero** all-ones masks, bbox keys of the form `<dicom_id>:<label_index>`.
`train` lung N is 208,987, matching the anchor count `CLAUDE.md` records, so the
builder's reimplemented anchor selection still agrees with `build_study_index`.

⚠ **The two terms are not separately logged**, so no past run can tell you what
the strong term did. `blip2_qformer.py:1297` blends them into one scalar as
`(lambda_weak/peak)*weak + (lambda_strong/peak)*strong`, i.e. `0.2*weak +
1.0*strong` at the old weights. Fix that before running any further ablation
here, or the experiment is unobservable. Note also the strong term fired on only
**869 of 222,758 train studies (0.39%)** and, under `warmup_start_epoch: 2,
warmup_epochs: 2` in a 5-epoch run, reached full weight for exactly one epoch —
so the A/B tests *this recipe*, not the idea in general.

Historical note: before that A/B, this section said "none of this path has run on
GPU yet". It has now — both arms, five epochs each, plus the evaluator against
both checkpoints.

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

**Split into two terms on 2026-08-16.** Before that a single pooled score summed
every positive finding, so the CAM answered "where is the evidence for *anything*
this study has?" — the right question for a lung mask, the wrong one for an expert
box drawn around one named finding.

| | weak | strong |
|---|---|---|
| target | CheXmask **lung** mask only | MS-CXR **box** for one finding |
| granularity | one pooled CAM per study | one CAM per (study, finding) |
| coverage | ~93% of studies | 823 train / 5 val / 138 test |
| supports the claim | "looks inside the lungs" | "looks at the pathology" |
| `lambda` | 0.05 | 0.25 |
| `top_k` | 0.2 | 0.5 |

They are returned separately by `ExplanationLoss.forward` and never summed inside
it: collapsing them lets the plentiful anatomical prior drown the scarce expert
signal while still looking like it learned localisation.

**The strong term costs one backward per *distinct finding* boxed in the batch —
typically one or two, not 14.** Studies without a box cost nothing.

⚠ **The weak term is gated to `explanation_mask_source == 0`.** The pooled cache
stores whichever annotation the builder preferred and
`choose_preferred_mask` prefers the MS-CXR bbox when a study has one, so without
that filter the weak term would run on a *bbox union* for exactly the 869 train /
164 test studies that also feed the strong term — supervising them twice and
making "weak = anatomical prior" false where it matters most. The strong term is
likewise skipped outright when `lambda_explanation_strong` is 0, rather than
computed and multiplied by zero.

Either lambda being > 0 enables the module; both 0 disables it entirely, including
CAM capture. There is still no separate `enabled` flag.

The shipped values are `0.0` / `0.0`. What the block looked like when it was on,
kept because re-enabling means restoring exactly this:

```yaml
model:
  loss:
    lambda_explanation: 0.05          # weak, CheXmask lung      (now 0.0)
    lambda_explanation_strong: 0.25   # strong, MS-CXR box       (now 0.0)
  explanation:
    top_k: 0.2            # weak
    strong_top_k: 0.5     # strong
    warmup_start_epoch: 2
    warmup_epochs: 2
    streams: [biovil, pubmedclip, swin]
    mask_cache_dir: /mnt/drive1tb/datasets/explanation_masks
```

Three traps in this path:

- **`ExplanationLoss.forward` returns three values** `(weak, strong, per_stream)`.
  A caller unpacking two breaks loudly, on purpose — silently dropping the strong
  term would be far worse.
- **Grad-CAM needs at least two channels to localise at all.** The channel weight
  is the gradient averaged over *tokens*, so a single-channel activation can only
  produce a flat CAM. This invalidated the first disease-specificity test; the
  fixture now uses two channels split left/right.
- **The per-finding boxes must ride the same sampled affine as the image.**
  `_apply_synced_image_mask_transforms` takes `extra_masks` and returns three
  values for exactly this reason. A separate draw would train the strong term on
  noise and still look healthy.

The cache gained a **second pair of files**, `masks_bbox_<split>.npy` and
`index_bbox_<split>.json` keyed `"<dicom_id>:<label_index>"`. A cache built before
2026-08-16 has no strong supervision rather than failing to load, so this needs a
rebuild before the strong term does anything. `label_index` is a position in
`CHEXPERT_LABELS`, which **must** stay identical to `chexpert_cols` in
`blip2_qformer.py` — reordering supervises the wrong finding silently. MS-CXR
categories with no CheXpert column are skipped and counted, never guessed.

Production config has this **off** — see the top of this section. When it was on
it ran against the cache at `/mnt/drive1tb/datasets/explanation_masks`; the
ablation used a rebuilt copy at `/home/phuong/explanation_masks_v2`, which is the
one carrying `masks_bbox_*`. The warmup schedule at 0.25 was epoch [0]–[1]
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
