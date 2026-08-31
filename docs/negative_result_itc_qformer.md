# Negative result: the Q-Former's ITC objective does not escape chance on this setup

**Status:** closed with a mechanism, 2026-08-31. `lambda_itc/itm/lm: 0.0` is the
supported setting and matches the reference implementation.

This is a negative result with an identified cause, not a record of a failure.
It belongs in the Limitations chapter, and it narrows the scope of what this
work claims rather than removing a contribution — see "What this does not
cost" below.

---

## 1. What was measured

Image-Text Contrastive (ITC) retrieval on the validation split, scored on the
pairs training actually uses (65.3% of studies; the rest carry no usable
FINDINGS). 256 valid pairs, so a chance mean rank is 127.5 and chance loss is
ln(256) = 5.5452. The gate requires `delta_nats >= +0.10`.

| # | date | configuration | temperature | rank i2t | rank t2i | `delta_nats` |
|---|---|---|---|---:|---:|---:|
| 1 | 2026-08-19 | untrained initialisation | 0.07 pinned | 130.68 | 130.30 | −0.0833 |
| 2 | 2026-08-19 | 525 updates, batch 8, 2 encoders | 0.00796 learned | 127.43 | 127.65 | −1.1168 |
| 3 | 2026-08-19 | 500 updates, temperature pinned | 0.07 pinned | 128.38 | 127.45 | **−0.0025** |
| 4 | 2026-08-31 | ~525 updates, batch 8, encoder FT off | 0.00944 learned | 127.49 | 127.76 | **−0.0118** |

Measurement 4 was run specifically to remove the two confounds available:
encoder fine-tuning was disabled and the batch held at 8, which is the exact
configuration that had produced the only encouraging training curve on record.
Its JSON carries `studies_scanned` 392 and `valid_fraction` 0.6531, so it is the
corrected measurement, not the earlier one that scored image-against-empty-string
pairs and was retracted.

**Four independent measurements, four times chance.**

## 2. The training curve is misleading, and this is reproducible

Both full attempts showed the same shape, so it should be treated as a property
of the setup rather than as evidence about retrieval:

| | iteration ~1,650 | iteration ~7,600 | epoch 1, iteration 23,750 |
|---|---:|---:|---:|
| `loss_itc` (2026-08-31 run) | 4.14 | 5.5625 | 5.5625, pinned |

Chance for the queue is ln(264) = 5.576. The loss drops ~1.4 nats below chance
early, then returns to chance and stays there. The 2026-08-18 attempt did the
same thing between iterations 1,000 and 6,000. **Do not read the training curve
as evidence of retrieval; only the held-out gate settles it.**

The learned temperature collapses in both attempts (0.00796, 0.00944). Pinning
it (measurement 3) removes the collapse and changes nothing else, so the
collapse is a symptom.

## 3. Three causes, all identifiable from the literature

### 3.1 The contrastive batch is ~210x too small

| | BLIP-2 stage 1 | this work |
|---|---|---|
| contrastive batch | **1,680–2,320** | **8** |
| pretraining data | 129M images | ~222k studies |
| hardware | 16 x A100 40 GB, < 6 days | 1 x RTX 5060 Ti 16 GB |

Softmax contrastive gives a positive pair the gradient "be more similar than
the hardest of the N−1 negatives". At N = 8 the hardest negative is usually
easy, so the gradient is weak and noisy — the failure mode SigLIP describes
when motivating a batch-independent loss. Chance is the predicted outcome.

The 8 is not a careless choice: it is what fits once the Q-Former's
vision-language block is enabled alongside two encoders on a 16 GB card
(measured: 14,763 MiB peak, and batch 8 with encoder fine-tuning also on OOMs).

### 3.2 The momentum queue was kept, against BLIP-2's design

BLIP-2 explicitly **removed** BLIP's momentum queue and used in-batch negatives,
because a frozen image encoder lets you fit more samples per GPU. This
implementation kept a 256-entry queue *and* has a batch of 8 — the worst of
both arrangements.

### 3.3 The queue holds stale keys

The queue is filled with detached features from the **live** encoder. MoCo and
ALBEF use a momentum encoder precisely so that keys stay comparable to current
queries; keys produced by an encoder that has since moved are not. The
reference implementation has no queue at all.

## 4. What this does not cost

The published reference implementation, `DasithEdirisinghe/META-CXR`, does not
train ITC either: the whole vision-language block of `blip2_qformer.py` is
commented out and Stage-1's loss is the MHCAC terms only. So `lambda_itc/itm/lm:
0.0` reproduces the published Stage 1 rather than diverging from it.

⚠ It does mean a Stage-1 checkpoint from this configuration is **not valid for
the Q-Former soft-token modes** of Stage 2, because the cross-attention that
would produce those soft tokens never sees a medical image. BLIP-2's own
ablation is explicit that this cannot be skipped: "Without representation
learning, the Q-Former fails to bridge the modality gap", with catastrophic
forgetting for decoder-style LLMs.

**The scope narrows, and the main experiment survives intact.** The comparison
this work reports is `native_anchor_only` against `native_anchor_guided`: both
use MedGemma's own image tower, and they differ in exactly one variable —
whether structured Positive/Negative/Uncertain cues from MHCAC classification
appear in the prompt. Those cues come from the classification head, which does
not depend on ITC. That is a **cleaner** ablation than one mediated by soft
tokens, because it isolates a single variable. The Stage-2 explainability layer
works on both arms unchanged: MedGemma's 256 image tokens on a 16x16 grid,
stage 1 of the attribution pipeline only.

## 5. What would be required to make it work

Not attempted here; recorded so it is not rediscovered from scratch.

1. **GradCache over the frozen-feature cache.** GradCache decouples the
   contrastive backward from the encoder along the batch dimension, giving
   near-constant memory at an arbitrary effective batch. This codebase is
   unusually well placed for it: the encoders are frozen and
   `run.feature_cache_dir` already exists, so the Q-Former could train alone on
   cached features with an effective batch in the thousands on the same 16 GB
   card. This addresses the measured cause directly.
2. **SigLIP's pairwise sigmoid loss in place of softmax ITC.** Removing the
   global normalisation makes each pair independent and degrades far less at
   small batch, and it is cheaper in memory. Its own sweet spot is still ~32k,
   so it is a multiplier on (1) rather than a substitute for it.
3. **A momentum encoder** if a queue is kept at all, or no queue, following
   BLIP-2.

Measured budget for a retry: an epoch at batch 8 is 3h26m (0.4445 s/it, 27,844
iterations), so ten epochs is ~34 h on this hardware. The gate at ~500
optimizer updates costs ~30 minutes and would have saved 33 of those hours.

## References

- Li et al., *BLIP-2: Bootstrapping Language-Image Pre-training with Frozen
  Image Encoders and Large Language Models*, ICML 2023. arXiv:2301.12597 —
  batch size 2320/1680, 129M images, in-batch negatives instead of the momentum
  queue, and the stage-1 ablation.
- Zhai et al., *Sigmoid Loss for Language Image Pre-Training*, ICCV 2023.
  arXiv:2303.15343 — why softmax contrastive degrades at small batch.
- Gao et al., *Scaling Deep Contrastive Learning Batch Size under Memory
  Limited Setup*, RepL4NLP 2021. arXiv:2101.06983 — GradCache.
