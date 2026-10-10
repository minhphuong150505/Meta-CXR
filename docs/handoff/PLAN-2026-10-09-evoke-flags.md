# PLAN 2026-10-09 — EVOKE-alignment flags: GPU smoke

Commit `8187363` added two opt-in flags after comparing the multi-view path
against EVOKE (arXiv 2411.10224) and its released code:

- `model.view_fusion.detach_aux` (default `false`): detach auxiliary-view K/V in
  `ViewFusionModule`, as EVOKE's `multiview_fusion` does.
- `model.loss.mpc_temperature` (default `0.07`): MPC softmax temperature;
  EVOKE's `region_temp` is 0.5.

Defaults reproduce every recorded run. This plan only checks that all four
combinations run end to end. It says nothing about quality.

## Executor steps (host) — run directly over SSH by the planning session

```bash
cd ~/Meta-CXR && git pull --ff-only          # 8187363
CUDA_VISIBLE_DEVICES="" ~/.venvs/meta-cxr-stage1-311/bin/python -m pytest tests/ training/ -q
setsid nohup bash ~/evoke_flags_smoke.sh > ~/evoke_flags_smoke.log 2>&1 < /dev/null &
```

`~/evoke_flags_smoke.sh` runs serially, into `~/smoke_20261009_evoke/{A,B,C,D}`,
each `PHASES="phase1b phase1c"` via `scripts/run_stage1_phases.sh`, starting
from `run_20261005_paper/checkpoint_phase1a.pth`, with
`run.truncate_train=2000 run.truncate_val=400 run.truncate_test=400` plus:

| config | `detach_aux` | `mpc_temperature` |
|---|---|---|
| A | false | 0.07 (= defaults) |
| B | true | 0.07 |
| C | false | 0.5 |
| D | true | 0.5 |

Abort: test failures, NaN/inf, OOM, `size mismatch`, missing phase checkpoint,
flag absent from the logged config.

## Execution report

### 2026-10-09 14:25–15:01, commit 8187363

- Host pytest (`tests/` + `training/`): exit 0, no failures (~1,174 tests).
- First launch attempt refused itself: its `grep -E` guard matched its own
  command line (the trap in CLAUDE.md). Relaunched with a positional-argv
  guard; one launch only.
- All four configs `rc=0`, ~9 min each, `checkpoint_phase1b.pth` and
  `checkpoint_phase1c.pth` written, test predictions written (400 studies,
  135 with a complementary view). No NaN/inf/Traceback/OOM/size mismatch.
- Each run's logged config carries its own `detach_aux` / `mpc_temperature`.

| config | max mem MiB | 1b s/it | 1c s/it | `loss_mpc` first → last | val macro_recall best 1b / 1c |
|---|---:|---:|---:|---|---|
| A | 7,988 | 0.33 | 0.610 | 2.4758 → 0.4816 | 0.4183 / 0.4288 |
| B | 7,988 | 0.33 | 0.610 | 2.4755 → 0.4871 | 0.4247 / 0.4340 |
| C | 7,990 | 0.33 | 0.609 | 2.8327 → 1.3693 | 0.4259 / 0.4192 |
| D | 7,988 | 0.33 | 0.610 | 2.8327 → 1.3747 | 0.4151 / 0.4258 |

Read: both flags reach the model. At τ 0.5 the MPC loss lives on a different
scale, as it must; `detach_aux` costs neither memory nor time. The val numbers
are 400 studies after ~125 updates per epoch: noise, not a comparison. A real
comparison needs full 1b+1c runs (~12 h each, see the multi-aux plan) and is
the user's call.

### Full run D, chained 2026-10-09 15:45 (user's choice)

Config A is the default and its full run already exists: `run_20261005_paper`
(logged config: seed 42, `max_aux_views` 1, `lambda_mpc` 0.02,
`lambda_view_consistency` 0.05, `p_view_drop` 0.15, no `detach_aux` /
`mpc_temperature` key, i.e. false / 0.07). So only D needs a full run.

`~/evoke_d_full.sh` (setsid, log `~/evoke_d_full.log`) waits for the
noise-floor smoke, re-checks ntfs3 / idle GPU / fresh dir, pulls, copies
`run_20261005_paper/checkpoint_phase1a.pth` into `~/run_20261009_evoke_d`, runs
`PHASES="phase1b phase1c"` with `model.view_fusion.detach_aux=true
model.loss.mpc_temperature=0.5` (training log `~/run_20261009_evoke_d.log`),
then writes `compare_stage1_predictions.py` of its test file against
`run_20261005_paper`'s into `~/run_20261009_evoke_d/compare_vs_paper.log`
(1,000 resamples, seed 16). Expected ~12 h (paper run: 1b 7h08m, 1c 4h40m).

Read the result like experiment B of the multi-aux plan: one run per arm, so
a delta is only meaningful if it exceeds the seed noise measured below, and a
drop that is as large on `views_1` (input unchanged) is training noise.

### Smoke scoring and noise floor, 2026-10-09 16:00

Test, 400 studies, `compare_stage1_predictions.py` (1,000 resamples, seed 16).
`A_rep` = A re-run at seed 42; `A_s43` = A at seed 43; both commit cb62ce5
(code identical to 8187363).

- **`A_rep` vs A: 0 changed argmax cells, every delta exactly 0.** Training is
  bit-deterministic at a fixed seed, so a flag's effect is not GPU noise.
- **`A_s43` vs A (seed alone):** wF1 +0.0067 [+0.0003, +0.0129], F1-5
  -0.0058 [-0.0200, +0.0083], AUROC mean +0.0127 [+0.0016, +0.0239]; on
  `views_1` wF1 +0.0162, F1-5 +0.0204. A seed change alone produces
  "significant" deltas of this size: the bootstrap CIs cover test sampling,
  not training variance.

| vs A (all, n=400) | wF1 | F1-5 | macro recall | AUROC mean |
|---|---|---|---|---|
| seed 43 (noise) | +0.007 | -0.006 | +0.000 | +0.013 |
| B detach | **-0.032** | **-0.089** | -0.007 | +0.001 |
| C tau 0.5 | -0.004 | -0.029 | -0.007 | +0.007 |
| D both | -0.008 | -0.037 | -0.008 | +0.003 |

Read: no config moves AUROC beyond the seed noise. B's F1 drop is ~4x the
one-seed noise and is the only clear signal (detach at tau 0.07 looks
harmful at this scale); C and D sit at ~1.5-2x on F1-5 only. One noise
sample, ~125 updates per epoch: indicative, not a result. Full D launched
15:55:26 (commit dd076f3).

### Decision rule, fixed 2026-10-09 BEFORE the D result exists (user)

User decision: compare D against A (`run_20261005_paper`) only; B and C will
not get full runs. Pick the better one by this rule, on the test comparison
(`compare_vs_paper.log`, group `all`, n = 3,269):

1. **Adopt D** only if `auroc_mean` OR `mean_weighted_f1_5` improves with a CI
   that excludes zero AND by more than the smoke seed noise (AUROC 0.013,
   F1-5 0.006 -- one sample, so treat ~0.015 as the bar), and neither of
   `weighted_f1` / `macro_recall` gets significantly worse.
2. Otherwise **keep A** (the shipped defaults): simpler, matches every
   recorded run, and a tie is not worth a config change.
3. If D wins only on `views_2plus` (the ~1/3 of studies with a lateral) and
   loses or ties on `views_1`, report it, but do not adopt on that alone.

Adopting D means setting `detach_aux: true` and `mpc_temperature: 0.5` in the
shipped YAML and updating CLAUDE.md / README / struct in the same commit.

### Full D result, 2026-10-10 03:51 — D loses; keep A (defaults)

`run_20261009_evoke_d`: 1b 7h07m + 1c 4h39m, `rc=0`, no NaN/OOM/Traceback,
logged config `detach_aux: true`, `mpc_temperature: 0.5`; host not rebooted
(up since 2026-10-04). Test vs `run_20261005_paper`, each from its phase-1c
`checkpoint_best`, paired bootstrap 1,000 x seed 16:

| group | n | weighted F1 | F1-5 | macro recall | AUROC mean |
|---|---:|---|---|---|---|
| all | 3,269 | -0.0033 [-0.0052, -0.0015] | **-0.0160 [-0.0193, -0.0122]** | -0.0039 [-0.0067, -0.0009] | -0.0044 [-0.0102, +0.0018] |
| views_1 | 2,022 | -0.0087 [-0.0108, -0.0067] | -0.0197 [-0.0239, -0.0152] | -0.0040 [-0.0070, -0.0011] | -0.0053 [-0.0109, -0.0002] |
| views_2 | 1,247 | +0.0046 [+0.0016, +0.0080] | -0.0107 [-0.0171, -0.0044] | -0.0023 [-0.0100, +0.0045] | -0.0079 [-0.0151, +0.0053] |

Decision rule (fixed beforehand): D needed a gain in AUROC or F1-5 beyond
~0.015; both fell, and weighted F1 and macro recall fell significantly.
**Keep A.** The only gain, weighted F1 +0.005 on studies with a lateral, comes
with lower F1-5 and AUROC in the same group (rule 3: not adopted on that).
Phase-1b validation already pointed the same way (D's AUROC below A's at all
5 epochs). One run per arm: this rules out D as configured, not every
EVOKE-style variant. B and C were not run in full (user decision).
