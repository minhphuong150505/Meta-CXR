# Learnable finding tokens — carrying Stage-1 mention into Stage 2

Experimental branch only. **Nothing here changes a production default, a
shipped checkpoint, or the meaning of any recorded number.** Every new code
path is behind an opt-in flag that defaults to off.

## Goal

Stage 1 produces two heads per finding: a mention gate `m = sigmoid(mention_logits)`
("will the report mention this at all") and a polarity distribution
`q = softmax(classification_logits)` over {negative, positive, uncertain},
**conditional on mention**. Stage 2 currently receives at most a *textual*
rendering of a thresholded combination of the two (`--cue-rule`), which four
independent measurements say does not help generation (`CLAUDE.md`, "BETTER CUES
HAVE NOT ESTABLISHED A GENERATION GAIN" and the n=100 fixed-stop confirmation).

Every one of those measurements passed the information through a **hard
threshold and a English sentence**. This branch tests a different channel: 13
learnable *finding tokens*, one per reportable finding, each carrying the
finding's identity plus the continuous mention/polarity numbers, projected into
MedGemma's embedding space and substituted at placeholder positions — the same
mechanism the 32 Q-Former soft tokens already use.

Done when: four matched arms have been generated and scored on one cohort with
paired bootstrap CIs, an inference-time ablation says whether the model uses the
mention channel at all, and a recommendation (adopt / keep experimental /
delete) is written here with the evidence behind it.

## What this is NOT

- Not a claim that it will work. The prior is unfavourable: the soft tokens
  already carry a linear-probe macro AUROC of 0.6847 (`scripts/probe_soft_tokens.py`),
  so the mention information may be redundant with what Stage 2 already gets.
- Not a production change. `--finding-tokens` defaults to `off`; omitting it
  leaves every existing code path byte-identical.
- Not a clinical result. Lexical NLG metrics are not clinical accuracy, and this
  repo has no validated clinical labeler (`training/evaluation/clinical.py`
  raises rather than scoring 0).

## 1. Design

### 1.1 What each token carries

13 tokens, one per entry of `stage2.prompts.ontology.MODELED_FINDINGS`
(`ABNORMALITIES_14` minus `No Finding`), in that fixed order. `No Finding` is
excluded because it has **zero negatives in every split** by construction
(`CLAUDE.md`, "`No Finding` is back in the classification head") — its `q` can
only be a constant.

Feature vector for finding *i*, with `m_i = sigmoid(mention_logits[i])` and
`q_i = softmax(classification_logits[i])`:

| variant (`--finding-tokens`) | numeric features | k |
|---|---|---|
| `q_only` (arm C) | `[q_neg, q_pos, q_unc]` | 3 |
| `full` (arm D) | `[m, m*q_pos, m*q_neg, m*q_unc]` | 4 |

`m*q_neg + m*q_pos + m*q_unc == m` exactly, so `m` is linearly redundant in
`full`. It is kept because the user's specification names it and because a
single linear projection then has a direct route to "was it mentioned" without
having to sum three inputs.

`q_only` is the control that isolates the *mention* information: it carries the
same identity embedding, the same projection, the same 13 token positions, and
the same polarity numbers — differing from `full` only in whether `m` multiplies
them. Arm C minus arm D is therefore the mention contribution, not the
"extra tokens" contribution.

**"Not mentioned" is not "negative".** Nothing in this design converts a low `m`
into a negative assertion. A low `m` simply scales all three polarity numbers
down toward zero, which is the honest encoding of "Stage 1 has no opinion here":
`q` is undefined when the finding was never mentioned, and multiplying by `m`
is what makes that undefinedness visible to the projection.

### 1.2 Identity, projection, normalization

```
token_i = W · LayerNorm( concat( E[i] , f_i ) ) + b        E: Embedding(13, 64)
                                                            W: Linear(64+k, hidden)
out_i   = gamma * token_i / rms(token_i)
```

- **Identity via a learnable embedding, not via position.** Ordering is fixed and
  pinned by a test, but the model is not asked to infer which finding a token
  means from where it sits: `E[i]` says so directly. This also means the
  projection `W` is *shared* across findings, so the numeric features have one
  consistent meaning instead of 13 independently-learned ones.
- **LayerNorm before the projection** because the two halves of the concatenation
  have wildly different scales: `E[i]` is free to grow during training while
  `f_i` is bounded in `[0, 1]`. Without it the numeric features would be a
  vanishing fraction of the input norm and the projection could learn to ignore
  them — which is the exact failure this experiment would then misreport as
  "mention information does not help".
- **RMS rescaling to the embedding table's own scale.** The substitution happens
  *inside* `get_input_embeddings()`, i.e. **after** Gemma's
  `Gemma3TextScaledWordEmbedding` multiplier has been applied to real tokens but
  not to substituted vectors. `gamma` is a single learnable scalar initialised to
  the measured RMS of the base embedding output, so the finding tokens start
  in-distribution rather than ~0. (This same asymmetry already applies to
  `img_proj`; it is **not** changed here — changing it would alter arms A and B
  and invalidate the comparison against every recorded run.)

### 1.3 Where the tokens go, and the attention mask

A new placeholder token `<finding_token>` is registered as an additional special
token and emitted 13 times by the prompt builder as
`PartKind.FINDING_TOKENS`, positioned **after** the Q-Former soft-token block and
**before** the task instruction.

**No attention-mask change is needed and none is made.** The finding tokens are
ordinary positions in `input_ids` with `attention_mask = 1`; MedGemma's decoder
is causal, so placing them before the instruction and the target means the
instruction and every generated token can attend to them, and they cannot attend
forward. Adding a custom mask here would be a silent divergence between the
train and generate paths for no benefit.

Substitution reuses `SoftTokenEmbeddingWrapper`'s mechanism by **composition**,
not by modification: `FindingTokenEmbeddingWrapper(SoftTokenEmbeddingWrapper(base, ...))`.
The inner wrapper substitutes at `<qformer_soft_token>`, the outer at
`<finding_token>`. `SoftTokenEmbeddingWrapper` is not edited at all, so arms A
and B execute exactly the code that produced the recorded results.

### 1.4 What stays untouched

Native pixel path, the 32 Q-Former soft tokens, `img_proj`, the cue-rule
machinery, `--cue-rule` defaults, stop-token handling, and the LoRA target
selection. No Q-Former token is reinterpreted as a finding: the two channels are
independent and the tokens are additional, not a re-labelling.

## 2. Leakage and train/inference parity

- The finding-token features come from `build_stage1_records`, which runs the
  **frozen Stage-1 model's predictions** on the image. The same function serves
  train, val, test and generation, so the input distribution is identical in all
  four. No ground-truth CheXpert label, no report-derived mention flag, and no
  target text ever enters the feature vector.
- **Stage 1 is frozen.** `build_stage1_records` is decorated `@torch.no_grad()`
  and the model is deleted afterwards; the finding-token encoder receives
  detached CPU tensors from the record cache. A test asserts gradient reaches
  `E`, `W` and `gamma` and that the stored features carry no `grad_fn`.
- **The Stage-1 checkpoint (`run_20260820_ft/checkpoint_best`) was selected on
  its own validation split, which is the same MIMIC val split Stage 2 uses.**
  This is pre-existing in arms A/B and every recorded arm C number; it is not
  introduced here. It is a real limitation and is stated in the report rather
  than papered over: the Stage-1 model has seen Stage-2's validation studies as
  *its own* validation, so Stage-2 validation is not fully held out with respect
  to Stage-1 model selection. The **test** split is clean for both stages.
- **Cache identity.** `stage1_cohort_fingerprint` gains
  `record_features = "with_class_logits"` **only when the finding-token branch is
  on**, following the exact pattern `cue_rule` already uses (omitted when off, so
  every existing cache still hits). Records built for a finding-token run
  additionally assert `class_logits` is present and has shape `[14, 3]`, and
  raise rather than fall back if it is missing. `class_logits` is now stored
  unconditionally (42 floats/study) so a future analysis needs no re-encode.

## 3. Arms

All four share: pipeline mode `meta_cxr_native_qformer_guided`, prompt config
`configs/experiment_native_qformer_guided.yaml`, section mode `findings_only`,
the same split/cohort and `--train-limit`, the same base model and LoRA config
(r=16/alpha=32), the same seed, the same optimizer/LR/epoch budget, the same
checkpoint-selection rule (validation cross-entropy), and the same greedy
decoding at 160 new tokens with the model's own stop IDs `[1, 106]`.

| arm | textual cues | finding tokens | isolates |
|---|---|---|---|
| **A** | `--cue-rule none` | off | baseline: native image + 32 soft tokens |
| **B** | `--cue-rule marginal_positive` | off | the current textual channel |
| **C** | `--cue-rule none` | `q_only` | the token channel without mention |
| **D** | `--cue-rule none` | `full` | the token channel with mention |

- **B is the honest comparator for "does this beat what we already have"**;
  A is the comparator for "does it add anything at all"; C vs D isolates `m`.
- **Every arm starts from the same initialisation.** No arm resumes another
  arm's adapter. `img_proj` and the LoRA weights are freshly initialised in all
  four from the same seed; the finding-token encoder exists only in C and D.
- **Parameter and token-count differences are reported, not hidden.** C and D add
  `13*64 + (64+k)*hidden + hidden + 1` parameters (~0.16 M against the 31.77 M
  already trainable, i.e. +0.5%) and 13 input tokens; B adds ~0 parameters and a
  variable number of *text* tokens. These are stated in the results table
  alongside every metric.
- Anti-repetition (`--no-repeat-ngram-size`, `--repetition-penalty`) stays **off**
  in all four, as in every recorded run, so the only moving part is the cue
  channel.

## 4. Test and run ladder

**Stage 0 — CPU tests** (no GPU, no data). New file
`tests/test_finding_tokens.py` plus additions to the prompt tests:

1. feature shapes: `[13, 3]` for `q_only`, `[13, 4]` for `full`; `No Finding` excluded.
2. label order equals `MODELED_FINDINGS` exactly, and index *i* of the feature
   matrix corresponds to `ABNORMALITIES_14[i+1]` — pinned against a hand-written list.
3. `m*q_pos + m*q_neg + m*q_unc == m` to float tolerance.
4. wrapper substitutes at exactly the 13 `<finding_token>` positions, leaves
   `<qformer_soft_token>` positions to the inner wrapper, and leaves all other
   positions bit-identical to the base embedding.
5. per-row correctness: row *b* of the batch receives row *b* of the features
   (the failure mode that `validate_soft_token_batch` exists for).
6. wrong position count / wrong batch size / wrong hidden width all raise.
7. gradient reaches `E`, `W`, `gamma`; features carry no `grad_fn`.
8. **default-path invariance**: with `--finding-tokens off` the rendered prompt,
   the part list, the cache fingerprint and the installed embedding module are
   identical to the pre-change code (compared against recorded expected values).
9. cache identity changes when and only when the branch is on; a cached record
   without `class_logits` raises for a finding-token run instead of falling back.
10. train/inference parity: the user-turn text and the substituted positions are
    byte-identical between `encode_train_example` and the generate path.

Plus the existing suite must not move from its baseline (host: 1,013 passed,
2 skipped as of `bdbf1af`).

**Stage 1 — GPU smoke.** 200 train studies, 10 val, 10 test, one epoch, arm D
only. Proves the record pass stores `class_logits`, the 13 placeholders survive
`apply_chat_template`, the substitution finds exactly 13 positions, and the run
writes `finding_tokens.pt` beside `img_proj.pt`. ~20 min. Read back: peak VRAM,
s/it, `status: complete`, and the count of substituted positions.

**Stage 2 — pilot, four arms, validation only.** Config is chosen here and
frozen before test is touched. Sized from the smoke measurement.

**Stage 3 — confirmation.** Generation on the frozen config, paired bootstrap
against the same cohort.

### Budget — requires approval before Stage 2

Arm C full-epoch training was 82h28m for 88,156 iterations at 3.37 s/it. Four
full arms is ~330 h and is not being proposed. The pilot proposal is
`--train-limit N` on the same recipe:

| N studies | iters (batch 2) | est. hours/arm | 4 arms |
|---:|---:|---:|---:|
| 10,000 | 5,000 | ~4.7 | ~19 h |
| 20,000 | 10,000 | ~9.4 | ~37 h |

plus a one-off Stage-1 record pass with `class_logits` (~73 min for a full train
split; proportionally less for a limited one) and ~1 h of generation per arm.
**No long run starts without the user approving one of these rows.** The exact
commands are in "Commands" below with `N` left as a variable.

## 5. Evaluation

Reported for every arm, on one cohort matched by `--restrict-to`:

- CIDEr, ROUGE-L, METEOR, BERTScore-F1 (BLEU-1..4 alongside).
- Degeneration and length: fraction of reports with a 5-gram repeated 3+ times,
  median words vs the reference median, truncation rate (fraction hitting the
  160-token cap), the fraction emitting `<end_of_turn>`, and generation failures.
- Paired per-study bootstrap, 2,000 resamples, seed 16, **D − A**, **D − B**,
  **D − C**, **C − A**, with CI95.
- Cost: wall time, peak VRAM, trainable parameter count, and mean input token
  count per arm.
- Finding-level accuracy **only if** a verified scorer exists at that point. It
  does not today — `lexicon_v2` is a hand-written lexicon fitted to val
  (`parse_coverage` 0.6477 in-sample) and `training/evaluation/clinical.py`
  raises rather than scoring. Any lexicon-derived count is labelled a lexical
  heuristic, never a clinical metric, and false-positive/omitted-finding rates
  are reported as **unavailable** if nothing validated exists.

### Ablation: does the model actually read the mention channel?

At inference only, on arm D's trained checkpoint, three interventions:

1. **zero** the numeric features (identity embedding kept);
2. **shuffle** the 13 feature rows across findings within a study (identity kept,
   so the token says "Edema" while carrying Pneumothorax's numbers);
3. **permute across studies** — study *i* gets study *j*'s features.

If all three leave the metrics unchanged, the branch is decorative whatever the
headline says. ⚠ **These are out-of-distribution interventions and are not a
substitute for arm C**, which is the properly-trained control. Report them as a
mechanism probe only.

## 6. Adoption criteria

Adopt only if **all** hold:

1. **D − B** clears zero on CIDEr *or* BERTScore-F1 with CI95 excluding zero, and
   no other reported metric moves significantly the wrong way;
2. **D − C** clears zero — i.e. the gain is attributable to the mention
   information, not merely to having 13 more learnable tokens;
3. repetition, length and truncation do not degrade;
4. the ablation shows the model is using the features;
5. cost is acceptable.

A falling training loss, a handful of good-looking reports, or one metric moving
slightly is **not** a result. If the evidence is short of the bar, the branch
stays flagged experimental or is proposed for deletion — and either way the
production default is not changed here. Adoption is proposed to the user
separately, with the numbers.

## Preconditions

- Branch off `main` at `c4d4357`; work on `feat/stage2-finding-tokens`.
- Host `phuong@100.116.167.90`, `~/.venvs/meta-cxr-stage1-311/bin/python`.
- `findmnt -no FSTYPE /mnt/drive1tb` must print `ntfs3`.
- GPU must be idle: `nvidia-smi`, `pgrep -af run_medgemma_qlora`, and the output
  directory must not already exist. **One GPU, one run** — refuse to queue.
- Stage-1 checkpoint root `~/run_20260820_ft`.

## Commands

Filled in as each stage is reached; nothing GPU-side runs before the CPU suite
is green and the host is confirmed idle.

## Abort if

- The GPU is busy or any output directory already exists — stop and report.
- The CPU suite moves from its baseline for any reason other than the new file.
- The smoke does not substitute exactly 13 positions per row.
- `class_logits` is missing from a record a finding-token run needs.
- Any artifact would carry patient data into Git.

## Execution report

_(appended by the executor)_
