# The hierarchical Stage-1 objective — `lambda_mention_conditioned_cls`

## Goal

Stage 1 currently trains the mention gate and the polarity classifier as **two
heads that never meet**: `lambda_cls: 1.0` fits `q` on the 20.5% of CheXpert
cells that are not blank, `lambda_gate: 0.5` fits `m` on all of them, and
nothing reconciles the two. The gate can answer "never mentioned" while the
classifier answers "Positive".

Every Stage-1 number this project reports is then scored under
`--label-framing study_presence --score marginal_presence`, i.e. on
`P(present) = sigmoid(m) · q_pos` — **a quantity nothing was ever trained on**.

`model.loss.lambda_mention_conditioned_cls` trains exactly that quantity as one
likelihood:

```
not mentioned      ->  -log(1 - m)
mentioned, class y ->  -log(m) - log(q[y])
```

Done when the run has finished, its thresholds are calibrated on validation, and
a paired per-study bootstrap on the same 3,269 test studies says whether
training the joint beats training the two factors separately.

## Why now, and why the old verdict does not apply

CLAUDE.md records a 2026-08-16 verdict that this "did not work". That judgment
was reached under `masked_polarity`, whose metric **masks blank cells and
therefore cannot see the joint at all**. It does not carry over. The only GPU
evidence is a 600-study / 3-epoch smoke, far too small to decide anything.

## The comparison, and the confound that had to be avoided

⚠ **The shipped `mimic_cxr_full.yaml` is the DEEP unfreeze (8 patterns,
53.12M), i.e. `run_20260821_deep`'s configuration — NOT `run_20260820_ft`'s
shallow 5-pattern one**, even though `run_20260820_ft` is the Stage-1 model this
project reports. Comparing a run launched from today's YAML against
`run_20260820_ft` would move the loss formulation **and** the unfreeze depth
together — the exact confound that already made kappa and the unfreeze
inseparable in `run_20260820_ft` itself.

**So the comparator is `run_20260821_deep`.** Identical config in every respect
except the three loss weights, which makes this a clean one-variable ablation.
Its artifacts are on the host and carry `mention_probabilities`, so a paired
bootstrap is possible:

| `run_20260821_deep`, test, `study_presence` + `marginal_presence` | |
|---|---:|
| `macro_auroc` | 0.7692 |
| `micro_auroc` | 0.8187 |
| `positive_macro_f1` | 0.3518 |
| `positive_macro_precision` | 0.3008 |
| `positive_macro_recall` | 0.4436 |
| `macro_specificity` | 0.8395 |

⚠ Read the result against **`run_20260821_deep`**, never against the 0.7643 /
0.3542 of `run_20260820_ft`.

## Two defects found and fixed before launching

1. **The new term was reported nowhere.** `loss_mention_conditioned` was not a
   field on `BlipOutput`, and enabling it forces `lambda_cls` and `lambda_gate`
   to 0.0 — so `loss_cls` and `loss_gate` print exactly `0.0000` and the total is
   otherwise auxiliary terms. A run whose `--options` silently failed to take
   would have looked **completely healthy for 14 hours while training no
   classification objective at all**. `base_task.train_step` collects any output
   key containing `loss`, so adding the field is enough.
2. **Two comments described a retracted design.** The comment above the loss call
   and its copy in `mimic_cxr_full.yaml` both claimed `classification_logits`
   becomes the log marginal in this mode. The code has never done that and must
   not: substituting the marginal aliases blank onto Negative, makes Positive
   unwinnable under the validation argmax, and pinned val F1 at exactly
   0.000000 for a whole smoke. Both corrected. There is no
   `conditional_classification_logits` key either.

Commit `b49ac68`.

## Preconditions (all verified 2026-09-11 before launch)

- Host `minhphuong` (100.116.167.90), `findmnt -no FSTYPE /mnt/drive1tb` -> `ntfs3`.
- GPU idle: 0 compute apps, 170 MiB, 0% util. `/home` 179 GB free.
- Revision `b49ac68` in the detached worktree `~/ft_review_20260910`.
- Output directory `~/run_20260911_mentioncond` did not exist.
- `mhcac.mention_conditioned_pos_weights` already ships in the YAML (alpha =
  n_not_mentioned / n_mentioned, capped at 10). Without them the term charges
  hiding a mentioned finding exactly as much as inventing one, and with 79.5% of
  cells blank that makes silence the majority answer.

## The tracked YAML is NOT edited

The three weights are passed as `--options`, and to the supervisor as
`EXTRA_OPTS`, so `mimic_cxr_full.yaml` keeps describing every other run made
from it.

## Commands

### Smoke (run, passed)

```bash
cd ~/ft_review_20260910 && CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled \
  ~/.venvs/meta-cxr-stage1-311/bin/python -m pretraining.train \
  --cfg-path pretraining/configs/mimic_cxr_full.yaml \
  --options run.output_dir=$HOME/mc_smoke run.truncate_train=2000 \
    run.truncate_val=200 run.max_epoch=1 run.eval_start_epoch=0 \
    model.loss.lambda_cls=0.0 model.loss.lambda_gate=0.0 \
    model.loss.lambda_mention_conditioned_cls=1.0
```

### Full run (launched)

```bash
cd ~/ft_review_20260910 && setsid nohup env CUDA_VISIBLE_DEVICES=0 \
  WANDB_MODE=disabled ~/.venvs/meta-cxr-stage1-311/bin/python -m pretraining.train \
  --cfg-path pretraining/configs/mimic_cxr_full.yaml \
  --options run.output_dir=$HOME/run_20260911_mentioncond \
    model.loss.lambda_cls=0.0 model.loss.lambda_gate=0.0 \
    model.loss.lambda_mention_conditioned_cls=1.0 \
  > $HOME/run_20260911_mentioncond.log 2>&1 < /dev/null &

# the MAIN pid is the one with ppid 1 -- pgrep returns DataLoader workers too
ps -eo pid,ppid,cmd | grep "[p]retraining.train" | awk '$2==1 {print $1}'

OUT=$HOME/run_20260911_mentioncond/mimic_cxr_full_blip2 \
LOG=$HOME/run_20260911_mentioncond.log BATCH=16 ACCUM=4 ADOPT_PID=<pid> \
EXTRA_OPTS="model.loss.lambda_cls=0.0 model.loss.lambda_gate=0.0 model.loss.lambda_mention_conditioned_cls=1.0" \
  setsid nohup bash ~/supervise.sh >> $HOME/run_20260911_mentioncond.supervise.log 2>&1 &
```

⚠ Launching directly and then ADOPTing is not optional: `setup_output_dir`
timestamps a nested directory when `$OUT` already exists, and the supervisor's
`mkdir -p "$OUT"` would create it. See CLAUDE.md, "`setup_output_dir` nests".

### Scoring, after it finishes

```bash
python scripts/calibrate_thresholds.py --predictions <val_best.npz> \
    --objective f1 --uncertain-policy ignore_uncertain \
    --label-framing study_presence --score marginal_presence \
    --selection plateau --plateau-fraction 0.95 --min-positive 5 \
    --output <thresholds.json>
python scripts/evaluate_stage1.py --predictions <test_best.npz> \
    --thresholds <thresholds.json> --uncertain-policy ignore_uncertain \
    --label-framing study_presence --score marginal_presence \
    --output-dir <private>
```

Then a paired per-study bootstrap against `run_20260821_deep`'s test npz on the
same 3,269 studies, each run using thresholds fitted on its **own** validation
split — that is what made the deep-vs-shallow comparison valid and it is the
same procedure here.

## Expected

- `loss_cls: 0.0000` and `loss_gate: 0.0000` on every training line.
- `loss_mention_conditioned` **non-zero and falling** — this is the one that
  matters; if it is absent or pinned, the options did not take.
- ~13,922 iterations per epoch, 10 epochs, ~0.35-1.0 s/it.
- Validation from epoch [5]; `checkpoint_best` on `selection_metric: loss`.
- ⚠ The total loss is **not comparable** to any previous run's: it is a
  different objective. Only the post-hoc test metrics compare.

## Abort if

- `loss_mention_conditioned` does not appear, or is exactly 0.0000.
- Validation `f1_positive_macro` is exactly 0.000000 for an epoch — that is the
  marginal-substitution signature and means the fix regressed.
- The supervisor reports an OOM fallback to batch 8: that halves every
  microbatch-local loss's sample count and is a **result to report**, not a
  recovery.
- `findmnt` reports `fuseblk` after any reboot.

## Smoke result — 2026-09-11, PASSED

`loss_cls: 0.0000`, `loss_gate: 0.0000`, `loss_mention_conditioned`
**1.9607 -> 1.8111 -> 1.7287**, total 2.6353 -> 2.4535, 0.3459 s/it.
`val_predictions_epoch_best.npz` carries `mention_probabilities` (range
0.1459-0.5252, mean 0.4418). Under the validation argmax Positive takes 81.5%
of valid cells and micro-Positive F1 is 0.8440 — **not 0.000000**, so the
retracted marginal substitution is genuinely absent. (That F1 is on 200
truncated val studies after one tiny epoch and is a plumbing check, not a
result.)

## Execution report — 2026-09-12, host `minhphuong`

- **revision:** `b49ac68`, detached worktree `~/ft_review_20260910`.
- **run:** `~/run_20260911_mentioncond`, launched 2026-09-11 10:59:50,
  `Training time 13:31:06`, supervisor `exited rc=0 / TRAINING COMPLETED`
  at 2026-09-12 00:33:18. **10/10 epochs, 0 restarts, 0 OOM, 0 fallbacks.**
- `max mem` **9,838 MiB** — the deep run's 9,839 to within a MiB, so the
  objective swap cost no memory. Epochs 1h16m-1h27m, 0.328-0.372 s/it.
- `checkpoint_best` = **epoch 9**, the last epoch, val loss **1.914257**, still
  falling (5-9: 1.932411, 1.919923, —, 1.918942, 1.914257). ⚠ The same shape as
  `run_20260820_ft`, and `run_20260821_ext` already falsified the inference that
  a falling val loss means more epochs would help. Do not read it as headroom.

### The monitoring fix earned its keep

Every training line showed `loss_cls: -0.0000` and `loss_gate: -0.0000`, as the
design requires, with `loss_mention_conditioned` **1.4560 (epoch 0) -> 1.3685
(epoch 1)** beside them. Without the `BlipOutput` field added in `b49ac68` this
log would have been indistinguishable from a 13.5-hour run training **no
classification objective at all**.

### Test result, 3,269 studies, `study_presence` + `marginal_presence`

Thresholds calibrated on each run's **own** validation split with the plateau
rule. Paired per-study bootstrap, 2,000 resamples, seed 16, against
`run_20260821_deep` — identical config but for the three loss weights:

| metric | deep | mention-conditioned | delta, 95% CI | |
|---|---:|---:|:---|---|
| `macro_auroc` | 0.7692 | 0.7693 | +0.0002 [-0.0045, +0.0046] | |
| `micro_auroc` | 0.8187 | **0.8434** | **+0.0247 [+0.0222, +0.0273]** | ✅ |
| `positive_macro_f1` | 0.3518 | 0.3617 | +0.0099 [-0.0013, +0.0203] | |
| `positive_macro_precision` | 0.3008 | 0.3082 | +0.0074 [-0.0049, +0.0183] | |
| `positive_macro_recall` | 0.4436 | **0.4781** | **+0.0345 [+0.0181, +0.0498]** | ✅ |
| `macro_specificity` | 0.8395 | 0.8294 | **-0.0101 [-0.0129, -0.0073]** | ❌ |

`macro_auprc` 0.3269 -> 0.3286.

### ⚠ The decomposition inverts how that table reads

`micro_auroc` +0.0247 with a very tight interval looks like the headline. It is
not, and this repo's own rule — "check macro against micro", "decompose the
score" — is what catches it. **Per-label AUROC, the same 3,269 studies:**

| | |
|---|---:|
| mean delta over 14 labels | **+0.0003** |
| labels where mention-conditioned wins | **8 of 14** — a coin flip |
| largest gain / largest loss | Atelectasis +0.0116 / Fracture -0.0118 |

**Not one finding got better.** The encoder unfreeze, by contrast, improved
AUROC on **14 of 14**. So the micro gain cannot be discrimination.

It is **cross-label score comparability**. `micro_auroc` pools every
label x study cell into one ranking and is therefore sensitive to whether the
scores mean the same thing across findings; per-label AUROC is invariant to any
per-label monotone rescaling and so cannot see it. Training `m·q` as one
likelihood makes `P(present)` commensurable across the 14 findings without
changing the ranking inside any of them.

The calibrated thresholds confirm the mechanism independently — scores that
agree across labels need less idiosyncratic cut points:

| | min | max | spread | stdev |
|---|---:|---:|---:|---:|
| deep | 0.101 | 0.648 | 0.548 | 0.1399 |
| mention-conditioned | 0.244 | 0.625 | **0.382** | **0.0922** |

34% tighter.

And the F1/recall/specificity pattern is the signature this file has named
before: **recall up and specificity down, both significant, with F1 and
precision not clearing zero** is an operating point moving, not a better model
— the mirror image of what `run_20260821_deep` and `run_20260821_ext` showed.

### Verdict

**The 2026-08-16 "did not work" verdict genuinely did not carry over** — it was
reached under `masked_polarity`, which masks blank cells and cannot see the
joint, and under the matching framing the objective produces a real, tightly
significant effect. **But the correct verdict is much narrower than "it
works": the hierarchical objective does not produce a better model, it
produces better-calibrated cross-label scores.**

That is worth having — it is the quantity this project actually scores, and one
global operating point now means something across all 14 findings — but it does
not move `macro_auroc`, which is the headline this project quotes, by anything
distinguishable from zero.

**Do not switch the reported Stage-1 model on this.** `run_20260820_ft` remains
it. This run is not even comparable to it (deep vs shallow unfreeze), and the
deep unfreeze itself never cleared zero on `macro_auroc` either.

### What would come next, if anything

The one clean follow-up is a **shallow**-unfreeze + mention-conditioned run, to
compare against `run_20260820_ft` directly rather than through the deep
configuration. ~14 h. It would answer whether the calibration gain survives on
the configuration this project actually reports. Nothing here requires it.

### Limitations

One seed, one run. `selection_metric: loss` under a different objective picks a
different epoch by construction, so "epoch 9 for both" is a coincidence, not a
controlled match. The micro/macro story is an inference from two measurements
plus the threshold spread; it is consistent and mechanistic but was not tested
by a dedicated calibration metric (ECE per label would settle it cheaply and was
not run).

### Artifacts on the host (not copied here)

`~/run_20260911_mentioncond/` (checkpoints, `result/*.npz`) ·
`~/run_20260911_mentioncond.log` · `~/run_20260911_mentioncond.supervise.log` ·
`~/eval_mentioncond/` (`thresholds.json`, `test/`, `paired.log`,
`paired_mentcond_vs_deep.json`) · `/tmp/perlabel_mc.py`. All outside the repo;
nothing from them enters a commit.
