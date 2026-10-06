# PLAN 2026-10-06 — Does Stage 1 gain from more than one auxiliary view?

Planner: Claude (dev box). Executor: whoever runs on the training host.
Status: **code ready, nothing run on GPU.** The card is busy with the Stage-2
paper run (`run_medgemma_qlora.py --prompt-style paper`, started 2026-10-06
~10:30); do not start anything below while it is alive.

## Question

`model.data.max_aux_views` has always been 1: the sampler refused anything else.
The user wants to know whether letting the model see 2-3 auxiliary views per
study improves the paper's Stage-1 metrics.

## What the data allows — measured 2026-10-06 on `full_allviews_v2` (CPU)

Number of complementary views per study under the sampler's rule (anchor by
PA > AP > lateral; an auxiliary must be a different projection from the anchor;
a second lateral IS allowed):

| split | studies | 0 aux | 1 aux | 2 aux | 3+ aux |
|---|---:|---:|---:|---:|---:|
| train | 222,758 | 52.7% | 43.0% | **4.2%** (9,448) | 0.1% (193) |
| val | 1,808 | 54.6% | 40.4% | **4.9%** (89) | 0 |
| test | 3,269 | 61.9% | 32.8% | **5.3%** (174) | 0.1% (2) |

Raising the cap changes the input for **~5% of studies and nothing else.** For
the other 95% the model sees exactly what it sees today. Most second auxiliaries
are a repeated lateral (`PA + 2 LAT` is the common case, `PA + AP + LAT` rarer),
which adds little new anatomy. Consequence for the readout: a whole-split delta
will be diluted ~20x and is expected to be inside the bootstrap noise whatever
the model does. **Read the `views_3plus` subgroup** (n = 176 test / 89 val),
and expect wide CIs there too.

## Code (this commit)

- `model/lavis/data/mimic_cxr_utils.py`: cap raised to
  `MAX_SUPPORTED_AUX_VIEWS = 3`; default and shipped YAML stay at 1.
- `model/lavis/tasks/image_text_pretrain.py`: the eval hook now writes
  `num_views` (anchor + real auxiliaries the model actually saw) into the
  prediction `.npz`. Absent when the batch has no `aux_mask`.
- `training/evaluation/subgroup_analysis.py`: `views_3plus` subgroup.
- `scripts/compare_stage1_predictions.py`: paired bootstrap of two `.npz` on the
  same studies, overall and by `num_views` (`views_1/2/3plus/2plus`), plus the
  fraction of argmax cells that changed.
- The model side needed nothing: collater, `_encode_aux_streams`,
  `ViewFusionModule` and `MultiPositiveContrastiveLoss` were already written for
  `[B, N, ...]`.

⚠ The eval-hook change is untested on the CPU box (`test_stage1_eval_hook.py`
needs torchvision there). **Run the full suite on the host first**; baseline
there is 4 failures, all `test_native_independence`.

## Experiment A — inference only (cheap, run first)

Same checkpoint, three evaluate-only passes differing only in
`model.data.max_aux_views`. Checkpoint: `run_20261005_paper` phase 1c
`checkpoint_best.pth` (D-026 config, which is what the YAML ships).

⚠ The model was TRAINED with at most one auxiliary. With two, the fusion softmax
spreads over twice as many key tokens — out of distribution. A gain here is
evidence; **a null here does NOT rule out a gain after retraining** (that is
Experiment B).

```bash
cd ~/Meta-CXR && git pull
PY=~/.venvs/meta-cxr-stage1-311/bin/python
CUDA_VISIBLE_DEVICES="" $PY -m pytest tests/ -q          # expect 4 failed (native_independence)

# guards: nothing else on the card, findmnt, fresh output dirs
ps -eo pid,args --no-headers | grep -E "pretraining.train|run_medgemma_qlora|generate_stage2" | grep -v grep && exit 1
nvidia-smi --query-gpu=memory.used --format=csv,noheader     # < 1000 MiB
findmnt -no FSTYPE /mnt/drive1tb                              # ntfs3
OUT=~/aux_probe_20261006; test ! -e $OUT || exit 1; mkdir -p $OUT

ROOT=~/run_20261005_paper
CK=$ROOT/phase1c/mimic_cxr_full_blip2/checkpoint_best.pth
for N in 1 2 3; do
  CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled $PY -m pretraining.train \
    --cfg-path pretraining/configs/mimic_cxr_full.yaml \
    --options run.phase=phase1c run.phase_root=$ROOT \
      run.evaluate=True run.test_splits=[val,test] run.save_predictions=True \
      run.resume_ckpt_path=$CK run.output_dir=$OUT/aux$N \
      run.distributed=false run.world_size=1 \
      model.data.max_aux_views=$N \
    > $OUT/aux$N.log 2>&1 || { echo "aux$N failed"; break; }
  grep -c "Loading evaluation weights from" $OUT/aux$N.log    # must be 1, else worthless
  grep "with a complementary view" $OUT/aux$N.log
  # evaluate-only resolves its output dir IN PLACE next to resume_ckpt_path, so
  # every pass writes the same two filenames; move them out before the next one.
  mkdir -p $OUT/aux$N
  find $ROOT $OUT -name "*_predictions_epoch_provided.npz" -mmin -120 \
       -not -path "$OUT/aux*/*" -exec mv {} $OUT/aux$N/ \;
  ls $OUT/aux$N/      # must list val_ and test_predictions_epoch_provided.npz
done
```

If the `find` above misses them, locate with
`find ~ -name "*_predictions_epoch_provided.npz" -mmin -30` and move by hand —
never let pass N+1 overwrite pass N.

Then, CPU:

```bash
for S in val test; do for N in 2 3; do
  $PY scripts/compare_stage1_predictions.py \
    --baseline $OUT/aux1/${S}_predictions_epoch_provided.npz \
    --candidate $OUT/aux$N/${S}_predictions_epoch_provided.npz \
    --output $OUT/cmp_${S}_aux${N}_vs_aux1.json > $OUT/cmp_${S}_aux${N}.log
done; done
# sanity: aux1 must reproduce the training-time test file exactly
$PY scripts/compare_stage1_predictions.py --subgroups-from baseline \
  --baseline $OUT/aux1/test_predictions_epoch_provided.npz \
  --candidate $ROOT/phase1c/mimic_cxr_full_blip2/result/test_predictions_epoch_best.npz \
  --output $OUT/sanity_aux1_vs_train.json
```

Expected:
- sanity: `changed_argmax_fraction` 0 (or ~0 from cuDNN nondeterminism) and every
  delta ~0. If not, the evaluate path differs from training-time eval — stop.
- the `views_1` and `views_2` groups of `aux2/aux3 vs aux1` must show
  `changed_argmax_fraction == 0`: those studies get identical input. If they
  change, something other than the cap moved — stop and report.
- the only group that can move is `views_3plus`.

Cost: 3 passes over val+test, a few minutes each.

Abort: OOM, `Loading evaluation weights` count != 1, `size mismatch`, any
`views_1`/`views_2` change.

## Experiment B — retrain with the cap raised (only if the user asks after A)

Phase 1a is anchor-only and unaffected, so start from the existing
`checkpoint_phase1a.pth` and re-run 1b + 1c with `max_aux_views=3`, all else
identical to `run_20261005_paper`. Measured cost of that run: 1b 7h08m, 1c
4h40m, ~12 h total; the extra auxiliaries touch ~4% of train studies, so expect
roughly the same. Compare its test file to `run_20261005_paper`'s with
`compare_stage1_predictions.py` (here the 95% single/one-aux studies DO change,
because the weights differ, so read all groups).

```bash
NEW=~/run_YYYYMMDD_aux3; test ! -e $NEW || exit 1; mkdir -p $NEW
cp ~/run_20261005_paper/checkpoint_phase1a.pth $NEW/
ROOT=$NEW PHASES="phase1b phase1c" EXTRA_OPTS="model.data.max_aux_views=3" \
  setsid nohup bash scripts/run_stage1_phases.sh > $NEW.log 2>&1 < /dev/null &
```

## Execution report

(executor appends here)

### Launch, 2026-10-07 00:24

`~/multiaux_chain.sh` (pid 42003, `setsid`) runs A then B unattended. It waits for
the Stage-2 paper full run (pid 33294), checks guards, pulls, runs pytest, then
writes Experiment A to `~/aux_probe_20261007/` and Experiment B to
`~/run_20261007_aux3/` (log `~/run_20261007_aux3.log`). Progress and every abort
reason: `~/multiaux_chain.log`. B's test comparison against `run_20261005_paper`
lands in `~/aux_probe_20261007/B_compare.log`.
