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
