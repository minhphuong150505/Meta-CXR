# Stage 1 — full run on the reference-matched recipe

Planner: this session. Executor: the same session over SSH (the run is launched
detached and supervised; there is no multi-step agent work to delegate).
Host: `ssh phuong@100.116.167.90`.

## Goal

One complete Stage-1 run, 10 epochs, on the recipe that now matches the reference
implementation: `lambda_itc/itm/lm = 0.0`, teacher + distillation on, two
encoders, batch 16 x accum 4. Produce `checkpoint_best.pth` selected on
validation `loss`, and the first throughput measurement for this exact
configuration.

## What this run is and is not

- It **is** the classification pipeline: MHCAC student/teacher, mention gate,
  multi-view fusion, MPC, view consistency. Loss weights match
  `DasithEdirisinghe/META-CXR` (`cls 1.0`, `contrastive 0.3`, `orth 0.7`,
  `sparsity 0.3`), plus this fork's teacher distillation and blank-label masking.
- It is **not** a Stage-2 `meta_cxr_qformer` candidate. With the VL block at 0.0
  the Q-Former's image path never runs, so its cross-attention keeps its BLIP-2
  initialisation. The reference is in the same position. `medgemma_direct` is
  unaffected. This is a known, accepted cost — see `CLAUDE.md`.
- Explanation loss stays off (`lambda_explanation*: 0.0`); its mask cache did not
  survive the 2026-08-17 reinstall anyway.

## Preconditions — verified 2026-08-19 before launch

| | |
|---|---|
| host repo | `7701fed`, branch `refactor/disease-specific-explanation`, clean |
| `/mnt/drive1tb` | mounted `ntfs3 (ro)` — the configs' path, not the udisks one |
| `/home` | 262 GB free, ext4 — `output_dir` must live here |
| GPU | idle, 166 MiB, no `pretraining.train` or supervisor running |
| venv | `~/.venvs/meta-cxr-stage1-311/bin/python` |

⚠ `scripts/supervise_stage1.sh` passes `run.batch_size_train` /
`run.accum_grad_iters` as overrides, so **its defaults beat the YAML**. They were
still 8 x 8 from the Q-Former era and were corrected to 16 x 4 in the same commit
as this plan. The launch below also sets them explicitly. If you ever change the
YAML's batch, change the script too.

## Commands

```bash
# 1. sync the host and deploy the supervisor (it is versioned with the recipe)
cd ~/Meta-CXR && git pull --ff-only && cp scripts/supervise_stage1.sh ~/supervise.sh

# 2. launch, detached, ONE invocation
OUT=$HOME/run_20260819_refmatch \
LOG=$HOME/run_20260819_refmatch.log \
WID=stage1-refmatch-20260819 \
BATCH=16 ACCUM=4 \
    setsid nohup bash ~/supervise.sh \
    >>$HOME/run_20260819_refmatch.supervise.log 2>&1 &
```

⚠ **One card, one run, one output path.** Check `nvidia-smi` and
`pgrep -af pretraining.train` first. On 2026-08-19 a launch was issued twice 50
seconds apart and the second process OOMed against the first, then overwrote the
first's report.

## Expected

- **Startup ~7 minutes** before the first `Train: data epoch` line: weights load
  and the encoders build. `blip2_pretrained.pth` is already cached. stdout is
  block-buffered (~8 KB bursts), so the log looks frozen while `[INFO]` lines
  keep arriving on stderr — the supervisor knows this and gives startup its own
  45-minute budget.
- **Epoch length 13,922 iterations** (222,752 studies / 16).
- **`s/it` is unmeasured for this configuration** and is the first number to
  read back. Bounds from measurements that do exist on this card: 0.2347 s/it
  (two encoders, no teacher, VL off) and 1.0364 s/it (three encoders, teacher on,
  VL off); the pinned-temperature probe ran 0.4354 s/it at batch 8 with the VL
  block **on**. Read the real number at ~200 iterations and record it.
- **`max mem` around 9.4 GB or below.** 16 x 4 peaked at 9,433 MiB with three
  encoders and the VL block on; this configuration is strictly smaller.
- `loss_itc`, `loss_itm`, `loss_lm` must print **exactly 0.0000** — that is the
  gating working and the Q-Former being skipped. Anything else means the YAML did
  not take.
- `loss_mpc` should fall within the first epoch (it went 3.35 → 2.66 previously);
  `loss_distill` in the 1e-2 range, not 1e-8, which was the degenerate value that
  got the teacher switched off once before.
- Validation starts at epoch **[5]** (`eval_start_epoch`), so epochs [0]–[4]
  train unscored and `checkpoint_best.pth` cannot exist before then. This is
  expected, not a failure. `checkpoint_last.pth` is rewritten every 1,000
  iterations.
- Total: 10 epochs plus five validation passes.

## Abort if

- `loss_itc/itm/lm` are non-zero → the config did not take. Kill and fix.
- OOM. It is a capacity finding: record batch/accum and peak VRAM. The supervisor
  will halve the batch and double accumulation, which keeps the effective batch
  at 64 but halves the candidates every microbatch-local loss sees (MPC in
  particular). **Treat a fallback as a result to report, not a recovery.**
- The machine hard-hangs. It did at 22:25 on 08-18 and 01:20 on 08-19, both with
  nothing at all in `journalctl -b -1`. The supervisor dies with the box —
  relaunch by hand, from `checkpoint_last`, and say so.
- `checkpoint_best.pth` still absent after epoch [5] finishes scoring.

## After the run

1. Calibrate thresholds on **validation only**, then score the test split once:

   ```bash
   python scripts/calibrate_thresholds.py --predictions <val.npz> --objective f1 \
       --uncertain-policy ignore_uncertain --min-positive 20 --output <thresholds.json>
   python scripts/evaluate_stage1.py --predictions <test.npz> --thresholds <thresholds.json> \
       --uncertain-policy ignore_uncertain --output-dir <dir>
   ```

   `--uncertain-policy` is not optional: it defaults to `three_class` in both
   scripts while training uses `ignore_uncertain`, and omitting it silently scores
   a different label binarisation (measured on one npz: macro AUROC 0.6537 vs
   0.7850).

2. Quote `macro_auroc` and per-label AUROC **with prevalence beside it**, not
   `positive_macro_f1`. On this label distribution six labels reach recall 1.0000
   with precision equal to prevalence, and `Fracture` reaches F1 0.983 on an AUROC
   of 0.442.

## Execution report

_(appended after the run)_

## Execution report — 2026-08-19

**The run did not survive. The cause is below the application: two kernel Oopses
in the NVMe block driver.**

- commit run: `dbd2dd5`
- launched 11:28:57, aborted by the supervisor 11:52:58 after three attempts

| attempt | outcome |
|---|---|
| 1 | **Segmentation fault, rc=139**, ~60 s in, before any training line |
| 2 | first training line 11:35:27, then **GPU ≤5% for 6 min while alive** → killed |
| 3 | same — GPU idle 6 min → killed; supervisor aborted on 3 failures with no checkpoint write |

`journalctl -k`:

```
11:29:05  traps: python[25114] general protection fault ... in python3.11
11:38:49  Oops: general protection fault ... non-canonical address 0xefff8a0f99170338 [#1]
          RIP: nvme_setup_rw+0x9c/0x2c0 [nvme_core]     Comm: pt_data_worker
11:47:24  Oops: general protection fault ... non-canonical address 0xdead000000000122 [#2]
          RIP: __rmqueue_pcplist+0x54/0x2e0             Comm: pt_data_worker
```

The first Oops is in the **NVMe driver**, faulting inside a DataLoader worker —
i.e. while reading the dataset. `0xdead000000000122` in the second is Linux's
`LIST_POISON2`: the page allocator's free list was already corrupt by then, which
is fallout from the first, not an independent bug. The kernel is
`Tainted: G D W O` — `D` = DIE, it has already died once.

**This is not something the recipe can fix, and it is not an OOM.** RAM was 23 GB
available of 30, `/dev/shm` 131 MB used of 16 GB, GPU peak well under the card.
No `oom-kill` anywhere in the log.

**It very likely also explains the two "unexplained" hard hangs** — 2026-08-18
22:25 and 2026-08-19 01:20, both of which left *nothing* in `journalctl -b -1`. A
kernel Oops in the block layer can take the machine down before anything reaches
the disk, which is exactly the signature that was recorded as "abrupt power loss
or hard hang".

### State left behind

- PID 25788 still holds **8,832 MiB of VRAM** and does not respond to `kill -9`
  — it is stuck in uninterruptible kernel state. **Only a reboot clears it.**
- A `[pt_data_worker] <defunct>` zombie is reparented to init.
- No checkpoint was written, so nothing is resumable. `~/run_20260819_refmatch/`
  can be deleted.

### What has to happen before Stage 1 is attempted again

1. **Reboot.** Anything measured on a kernel that has Oopsed twice is untrustworthy.
2. **SMART on both drives** (needs the user's sudo — there is no passwordless sudo):
   `sudo nvme smart-log /dev/nvme1` and `/dev/nvme0` — read `critical_warning`,
   `media_errors`, `percentage_used`. The dataset disk is a **BIWIN NV7200 1TB**;
   the system disk is a **Patriot P400L 500GB**.
3. **`chkdsk` from Windows on the NTFS volume.** It was already recorded as dirty
   with real NTFS errors, and `ntfs3` building malformed requests on a damaged
   volume is a coherent path to a fault inside `nvme_setup_rw`. `ntfsfix` clears
   the dirty bit without repairing anything and must not be used as a substitute.
4. Only one kernel is installed (`7.0.0-29-generic`), so there is no older kernel
   to A/B against. If SMART is clean and chkdsk finds nothing, installing an
   older kernel series is the next test.

Until then, do not start a 20-hour run on this storage: the failure is silent
from the application's point of view, and the previous two occurrences took the
whole machine with them.
