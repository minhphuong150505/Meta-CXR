# PLAN 2026-10-24 — Ablation: does the multi-view package help Stage 1?

Planner: Claude (dev box). Executor: whoever runs on the training host.
Status: **DONE 2026-10-25** (user go-ahead 2026-10-24). Result: execution report at the end.

## Question

`docs/CHI_MUC_TINH_MOI.md` TM-05 records that the multi-view additions
(ViewFusion, StreamAdapter, MPC, view consistency) were never ablated against an
anchor-only model. Every multi-view experiment so far varied *how* multi-view
works (`max_aux_views` 3, EVOKE flags) and none beat the defaults. This plan
measures the package itself: `run_20261005_paper` (arm A, multi-view on, already
trained) against the same recipe with `multi_view: false` (arm B, new run).

## What `multi_view: false` removes — read before interpreting

From the code (`blip2_qformer.py:528-596`, `ReportDataset.py:609-612`):

| | arm A (`run_20261005_paper`) | arm B (this plan) |
|---|---|---|
| studies, anchor, order | one row per study, anchor PA > AP > lateral | **identical** (`study_sampling` is independent of `multi_view`) |
| auxiliary image loaded | ≤ 1 | none (`max_aux_views` forced to 0) |
| `view_fusion` | built, trained | not built |
| `stream_adapters` | built, trained, **on the anchor path too** | not built |
| `mpc_heads` / MPC loss | `lambda_mpc` 0.02 | not built; set to 0 |
| view-consistency loss | `lambda_view_consistency` 0.05 | set to 0 |

⚠ So B removes **more than fusion**: the StreamAdapters are residual layers on
the anchor's own tokens. A difference on single-view studies (`views_1`, whose
image input is identical in both arms) therefore measures adapters + MPC +
training noise, not the auxiliary view. The view's own contribution is read from
`views_2` against `views_1` (rule below). Separating the adapters from fusion
would need a third arm; not planned.

## Trap: phase blocks override `--options`

`apply_phase_to_config` merges the phase block AFTER the CLI options
(`pretraining/phases.py:256`), and phase1b and phase1c both set
`model.multi_view: true` and the two lambdas. **`--options model.multi_view=false`
is silently overwritten.** Override inside the phase blocks instead:

```
run.phases.phase1b.model.multi_view=false
run.phases.phase1b.model.loss.lambda_mpc=0.0
run.phases.phase1b.model.loss.lambda_view_consistency=0.0
run.phases.phase1c.model.multi_view=false
run.phases.phase1c.model.loss.lambda_mpc=0.0
run.phases.phase1c.model.loss.lambda_view_consistency=0.0
```

The `trainable` lists still name `view_fusion`, `stream_adapters`, `mpc_heads`;
`apply_trainable` matches by prefix and an absent module simply matches nothing.

## Matching arm A's recipe

`git diff 206fb0e HEAD -- pretraining/configs/mimic_cxr_full.yaml` (206fb0e is
the YAML `run_20261005_paper` ran with) differs in only three things that
touch 1b/1c: phase1b `epochs` 5 → 3, phase1b `test_splits` [] → [test]
(both 2026-10-11), and two new keys (`mpc_temperature: 0.07`,
`detach_aux: false`) whose values equal the old defaults. Restore the first
two, and run 1c (now opt-in):

```
run.phases.phase1b.epochs=5
run.phases.phase1b.run.test_splits=[]
PHASES="phase1b phase1c"
```

Seed stays 42. Phase 1a is anchor-only (`multi_view: false` already), so B
starts from A's own `checkpoint_phase1a.pth`, as `run_20261007_aux3` and
`run_20261009_evoke_d` did.

⚠ Code has moved since 2026-10-05 (GradCache reuse, `_qformer_tokens`, phase1q
work). Those were written to be bit-identical when off, but no full arm-A rerun
exists at HEAD. Gate G1 checks the evaluation path; the training path is a
stated limitation, as it was for both earlier ablations.

## Steps (host)

### G0 — guards and tests

```bash
cd ~/Meta-CXR && git pull --ff-only && git log -1 --oneline
PY=~/.venvs/meta-cxr-stage1-311/bin/python
CUDA_VISIBLE_DEVICES="" $PY -m pytest tests/ -q        # baseline: 4 failed, all test_native_independence
findmnt -no FSTYPE /mnt/drive1tb                        # ntfs3, else STOP
nvidia-smi --query-gpu=memory.used --format=csv,noheader   # < 1000 MiB
ps -eo pid,args --no-headers | awk '$0 ~ /pretraining\.train|run_medgemma_qlora|generate_stage2|precompute_features/ && $0 !~ /awk/'   # must print nothing
A=~/run_20261005_paper
ls -l $A/checkpoint_phase1a.pth $A/phase1c/mimic_cxr_full_blip2/checkpoint_best.pth \
      $A/phase1c/mimic_cxr_full_blip2/result/test_predictions_epoch_best.npz
df -h /home                                             # > 30 GB free
```

### G1 — evaluation path reproduces arm A (few minutes)

```bash
OUT=~/nomv_probe_20261024; test ! -e $OUT || exit 1; mkdir -p $OUT
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled $PY -m pretraining.train \
  --cfg-path pretraining/configs/mimic_cxr_full.yaml \
  --options run.phase=phase1c run.phase_root=$A run.evaluate=True \
    run.save_predictions=True run.output_dir=$OUT/A_rescore \
    run.resume_ckpt_path=$A/phase1c/mimic_cxr_full_blip2/checkpoint_best.pth \
    run.distributed=false run.world_size=1 > $OUT/A_rescore.log 2>&1
grep -c "Loading evaluation weights from" $OUT/A_rescore.log      # must be 1
# evaluate-only writes next to the checkpoint: move the fresh file out
find $A -name "test_predictions_epoch_provided.npz" -mmin -60 -exec mv {} $OUT/A_rescore.npz \;
$PY scripts/compare_stage1_predictions.py --subgroups-from baseline \
  --baseline $A/phase1c/mimic_cxr_full_blip2/result/test_predictions_epoch_best.npz \
  --candidate $OUT/A_rescore.npz --output $OUT/G1_sanity.json
```

Pass: `changed_argmax_fraction` 0 (or ~0) and every delta ~0. Otherwise STOP —
HEAD's evaluation differs from the one that scored arm A.

### S — smoke of arm B (~20 min)

```bash
SM=~/smoke_20261024_nomv; test ! -e $SM || exit 1; mkdir -p $SM
cp $A/checkpoint_phase1a.pth $SM/
NOMV="run.phases.phase1b.model.multi_view=false run.phases.phase1b.model.loss.lambda_mpc=0.0 \
run.phases.phase1b.model.loss.lambda_view_consistency=0.0 run.phases.phase1c.model.multi_view=false \
run.phases.phase1c.model.loss.lambda_mpc=0.0 run.phases.phase1c.model.loss.lambda_view_consistency=0.0 \
run.phases.phase1b.epochs=5 run.phases.phase1b.run.test_splits=[]"
ROOT=$SM PHASES="phase1b phase1c" \
  EXTRA_OPTS="$NOMV run.truncate_train=2000 run.truncate_val=400 run.truncate_test=400" \
  bash scripts/run_stage1_phases.sh > $SM.log 2>&1
```

Pass, all of:
- `rc=0`, both `checkpoint_phase1b.pth` and `checkpoint_phase1c.pth` in `$SM`;
- the dataset line reads `0 with a complementary view (multi_view=False, max_aux_views=0)`;
- no `multi-view fusion enabled` line in the log;
- the logged config shows `multi_view: false`, `lambda_mpc: 0.0`,
  `lambda_view_consistency: 0.0` in both phases, and 1b ran 5 epochs;
- `loss_mpc` absent or exactly 0; no NaN/inf, OOM, `size mismatch`.

Record `max mem` and s/it for 1b and 1c beside arm A's smoke (7,988 MiB,
0.33 / 0.61 s/it).

### F — full arm B (expect ≤ 12 h; A took 1b 7h08m + 1c 4h40m, B reads no aux images)

```bash
B=~/run_20261024_nomv; test ! -e $B || exit 1; mkdir -p $B
cp $A/checkpoint_phase1a.pth $B/
ROOT=$B PHASES="phase1b phase1c" EXTRA_OPTS="$NOMV" \
  setsid nohup bash scripts/run_stage1_phases.sh > $B.log 2>&1 < /dev/null &
```

Launch once. Monitor with `scripts/train_healthcheck.sh` (`EXPECT_RUNNING=1`);
never pin the epoch index in a progress grep.

### C — comparison (CPU, after F)

```bash
BT=$B/phase1c/mimic_cxr_full_blip2/result/test_predictions_epoch_best.npz
AT=$A/phase1c/mimic_cxr_full_blip2/result/test_predictions_epoch_best.npz
# baseline = A, candidate = B, so a positive delta means "B better";
# subgroups from A: B's file carries no num_views (multi-view off)
$PY scripts/compare_stage1_predictions.py --subgroups-from baseline \
  --baseline $AT --candidate $BT --output $OUT/B_vs_A_test.json > $OUT/B_vs_A_test.log
# supplementary, headline rule: B's own cutpoints, fitted on B's val
$PY scripts/calibrate_thresholds.py --rule cutpoints \
  --predictions $B/phase1c/mimic_cxr_full_blip2/result/val_predictions_epoch_best.npz \
  --checkpoint run_20261024_nomv --output $OUT/cutpoints_nomv.json
$PY scripts/evaluate_stage1.py --predictions $BT --cutpoints $OUT/cutpoints_nomv.json \
  --output-dir $OUT/eval_B_cutpoints
```

(If the val file has another name, `ls $B/phase1c/mimic_cxr_full_blip2/result/`.)
The cutpoint file is for this ablation only: keep it in `$OUT`, do not commit it
to `configs/stage1_cutpoints/`.

## Reading rule — fixed now, before any result exists

Primary: test, n = 3,269, argmax (paper protocol), A − B on `auroc_mean`,
`mean_weighted_f1_5`, `weighted_f1`, `macro_recall`, paired bootstrap 1,000 ×
seed 16. (The script reports B − A; flip the sign.) Noise bar: the only seed-noise sample on
record is the 2026-10-09 smoke (AUROC 0.013 at n = 400), so treat ~0.015 as the
bar, as the EVOKE plan did.

1. **"The multi-view package helps"** only if A > B on `auroc_mean` or
   `mean_weighted_f1_5` with a CI excluding zero and by more than ~0.015, and
   neither `weighted_f1` nor `macro_recall` is significantly worse in A.
2. **"The auxiliary view itself helps"** needs, in addition, the A − B gain on
   `views_2plus` (studies that had a lateral) to be larger than on `views_1`
   (input identical). A gain of equal size on `views_1` is attributed to
   adapters/MPC/noise, not to seeing a second view.
3. Otherwise: **"no measurable benefit at this budget"** — the thesis then
   argues for multi-view on design grounds only, and says so in Limitations.
   If B is significantly better, report that plainly; it does not by itself
   change the shipped YAML (that is the user's decision).

Report beside the result: one run per arm, seed variance not separated; B also
removes the anchor-path adapters; the cutpoint numbers are supplementary.

## Abort conditions

G0 guard fails; G1 not reproduced; smoke misses any pass item; OOM, NaN/inf,
`size mismatch`, missing phase checkpoint; another process appears on the GPU.
On abort: report command, exit status, first error with ~20 lines, `max mem`,
s/it, epoch/iter, and the log path — not the log.

## Data handling

`.npz`, cutpoint JSON, logs stay on the host. The execution report below carries
aggregate numbers only — no ids, paths to patient images, or report text.

## Execution report

Executor: Claude, planning session over SSH; one chain script (`~/nomv_chain.sh`,
launched once, `~/nomv_chain.log`), host checkout at `0cf2df0`.

### G0 -- passed, after one host hang

- The first `git pull` + pytest over SSH (2026-10-24 ~16:10) coincided with a
  **hard hang of the host**: offline ~1 h, rebooted 17:18:46, nothing in
  `journalctl -b -1` (no Oops, Xid, MCE). Cause unknown. Re-run detached
  (`setsid`): pytest exit 0 in 45 s. `/mnt/drive1tb` `ntfs3`, GPU idle, no other
  job. No further hang during the ~10 h that followed.

### G1 -- passed (17:36)

Evaluate-only re-score of arm A at HEAD (`Loading evaluation weights from` = 1)
vs A's training-time test `.npz`: every delta and `changed_argmax_fraction` 0
(|v| <= 1e-3 check). ⚠ A's training-time `.npz` predates the `num_views` export
(2026-10-06), so the compare runs `--subgroups-from candidate`, and step C uses
the re-score as A (identical by this gate). Evaluate-only writes
`test_predictions_epoch_provided.npz` next to A's checkpoint; the script moves it
to `~/nomv_probe_20261024/A_rescore.npz`.

### S -- passed (17:43)

2,000 / 400 / 400 studies, 1b 5 epochs + 1c 1 epoch: dataset line `0 with a
complementary view (multi_view=False, max_aux_views=0)`, no fusion built,
`loss_mpc` / `loss_view_consistency` 0.0000, no NaN/inf/OOM/size mismatch.
`max mem` 7,026 MiB (A's smoke 7,988). The two `"multi_view": true` lines in the
logged config are the phase1a/phase1q blocks, which do not run.

### F -- completed, rc=0 (17:43 -> 03:12)

| | arm B | arm A (record) |
|---|---|---|
| phase 1b, 5 epochs | **5h11m37s**, 0.24-0.31 s/it, `max mem` 4,816 MiB | 7h08m |
| phase 1c, 1 epoch | **4h12m09s**, 0.540 s/it, `max mem` 7,977 MiB | 4h40m |

`checkpoint_best` of 1b updated through epoch index 3; 0 Traceback / OOM in the log.

### C -- B vs A, test n = 3,269, argmax (paper protocol)

Script reports B - A; the table flips it to **A - B** (positive = multi-view
package better). Paired bootstrap 1,000 x seed 16. `*` = CI excludes zero.

| A - B | all (3,269) | views_1 (2,022) | views_2 (1,247) |
|---|---|---|---|
| `auroc_mean` | +0.0054 [-0.0005, +0.0112] | -0.0003 [-0.0067, +0.0059] | **+0.0107 [+0.0007, +0.0207]** * |
| `mean_weighted_f1_5` | **+0.0129 [+0.0090, +0.0174]** * | **+0.0126 [+0.0081, +0.0173]** * | **+0.0131 [+0.0062, +0.0203]** * |
| `weighted_f1` | **+0.0057 [+0.0037, +0.0078]** * | **+0.0050** * | **+0.0062** * |
| `macro_recall` | **+0.0060 [+0.0028, +0.0092]** * | **+0.0069** * | +0.0049 [-0.0009, +0.0106] |
| `weighted_precision` | **-0.0071 [-0.0094, -0.0047]** * | **-0.0087** * | **-0.0055** * |
| AUROC Pos / Neg / Unc | -0.0024 / +0.0009 / **+0.0187** * | -0.0051 * / -0.0030 / +0.0077 | -0.0025 / +0.0038 / **+0.0324** * |

Arm A argmax: weighted F1 0.7795, F1-5 0.6687, macro recall 0.4609, AUROC mean
0.7436; arm B 0.7738 / 0.6558 / 0.4548 / 0.7382. `changed_argmax_fraction` 0.053
(views_1 0.039, views_2 0.075). No test study has more than two views
(`max_aux_views: 1`), so `views_3plus` is empty and `views_2plus` = `views_2`.

Supplementary, headline rule (B's own val cutpoints, `~/nomv_probe_20261024/
cutpoints_nomv.json`, not committed): B weighted P/R/F1 **0.822 / 0.835 / 0.817**,
F1-5 **0.748**, macro recall 0.440, against A's recorded 0.827 / 0.834 / 0.820,
F1-5 0.749, macro recall 0.446 -- equal within ~0.005, no paired CI computed.

### Reading, by the pre-declared rule

1. **Rule 1 ("the package helps") is NOT met.** A - B on `mean_weighted_f1_5` is
   +0.0129 with a CI excluding zero, but below the ~0.015 noise bar; `auroc_mean`
   +0.0054 crosses zero. A is significantly better on weighted F1 (+0.0057) and
   macro recall (+0.0060) too, so nothing in A is worse -- the direction is
   consistent, the size is below the bar.
2. **Rule 2 ("the aux view itself helps") is NOT met on F1:** the F1-5 gain is the
   same on single-view studies (+0.0126), whose input is identical in both arms,
   as on two-view studies (+0.0131). By the rule that is adapters / MPC / noise,
   not the second view. **AUROC splits the other way:** zero on `views_1`
   (-0.0003), +0.0107 [+0.0007, +0.0207] on `views_2`, carried by the Uncertain
   class (+0.0324). Not pre-declared as decisive; report it as a hint.
3. **Verdict: no measurable benefit beyond the noise bar at this budget.** The
   package gives a small, consistent argmax gain (F1-5 +0.013, weighted F1
   +0.006, macro recall +0.006, all significant) that is NOT specific to
   multi-view studies, and disappears under the headline cutpoint rule (F1-5
   0.749 vs 0.748). Arm B trains 1b ~27% faster and uses less memory.

Limitations: one run per arm, seed variance not separated (the bar is the
2026-10-09 smoke's 0.013 AUROC at n = 400); B also removes the anchor-path
StreamAdapters, so `views_1` deltas mix adapters, MPC and noise; the training
path at HEAD was not re-checked against A's (G1 covers evaluation only).
Shipped YAML unchanged (multi-view stays on): that is the user's decision.
