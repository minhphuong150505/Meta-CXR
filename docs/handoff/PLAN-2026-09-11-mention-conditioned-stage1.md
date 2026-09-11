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

## Execution report

_(appended when the run finishes)_
