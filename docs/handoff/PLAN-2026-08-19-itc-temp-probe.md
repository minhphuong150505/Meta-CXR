# ITC temperature probe — is the contrastive head dead, or just its temperature?

Owner of the plan: Claude. Executor: Codex. Host: the training desktop
(`ssh phuong@100.116.167.90`). Everything below runs there, nothing runs in the
dev checkout.

## Goal

Decide, for ~40 minutes of GPU time instead of ~33 hours, whether the ITC
collapse observed on `run_20260818_qformer` is caused by the **learned
temperature** or by the **representation**. One number decides it: the mean rank
of the true pair on val, measured at a pinned temperature, before and after 500
optimizer updates.

## Background — what is already known (do not re-measure)

`run_20260818_qformer` (batch 8 / accum 8, swin off, `lambda_itc/itm/lm = 0.1`)
reached iter 13,000 / 27,844 of epoch 0 before the machine hard-hung at ~01:20 on
2026-08-19. `checkpoint_last.pth` at iter 13,000 is intact and resumable.

- Train `loss_itc` sat 0.9–1.3 nats **below** chance (ln 264 = 5.576) from iter
  1,000 to 5,000, then returned to **exactly** chance from iter 6,000 and stayed
  there for 7,000 more iterations.
- The gate on the iter-4200 snapshot (`~/ckpt_at_500_updates.pth`, ≈525
  optimizer updates) failed on val: `delta_nats` **−24.52** against a required
  ≥ +0.10, true-pair rank **116.64 / 126.81** against chance 127.5 — barely
  better than the untrained model's 120.21 / 129.56.
- The learned temperature fell **0.024888 → 0.00796** over those 525 updates,
  heading for its 1e-3 clamp. `delta_nats` scales with 1/temperature, which is
  most of why the loss number exploded to 30.07.
- `loss_itm` (0.575 vs chance 0.6365) and `loss_lm` (3.11 → 2.6) are learning
  normally. The failure is specific to ITC.

**Hypothesis under test:** the temperature collapses faster than the
representation can separate pairs, and the sharpened logits then dominate the
gradient. If so, pinning the temperature should let the ranks move. If the ranks
still do not move, the temperature was a symptom and the ITC weights should go
to 0.0.

## Preconditions

1. Repo at `~/Meta-CXR`, branch `refactor/disease-specific-explanation`.
   **`git pull` first** — this plan needs the commit that adds
   `model.loss.itc_temp` / `itc_temp_learnable` and `--options` on
   `scripts/check_itc_gate.py`. Record the sha you ran.
2. Python: `~/.venvs/meta-cxr-stage1-311/bin/python`. Nothing else.
3. **The dataset must be visible at the path the config names.** As of
   2026-08-19 08:5x it was mounted read-**write** by udisks at
   `/run/media/phuong/A4E6C088E6C05BE4`, while both `configs/env_config.yaml`
   and `pretraining/configs/mimic_cxr_full.yaml:576` (`mask_cache_dir`, a
   hardcoded absolute path) name `/mnt/drive1tb`. Preferred fix, ask the user
   for sudo:

   ```bash
   sudo umount /run/media/phuong/A4E6C088E6C05BE4
   sudo mount -t ntfs3 -o ro UUID=A4E6C088E6C05BE4 /mnt/drive1tb
   df -T /mnt/drive1tb        # must print ntfs3
   ```

   If the user is unavailable, `~/itc_gate_500.sh` shows the fallback: patch both
   files with `sed`, restore them from a `trap ... EXIT`, and verify
   `git status --short` is clean afterwards. **Never leave the tracked YAML
   patched.**
4. GPU free — the training run is dead, nothing else should be using the card.

## Commands

### Step 1 — matched baseline, untrained, temperature pinned (~7 min)

The existing `~/itc_gate_baseline.json` was measured at temperature 0.024888 and
is **not** comparable to a pinned-temperature run. Measure a fresh one.

```bash
cd ~/Meta-CXR
~/.venvs/meta-cxr-stage1-311/bin/python -u scripts/check_itc_gate.py \
  --cfg-path pretraining/configs/mimic_cxr_full.yaml \
  --split val --pairs 256 --batch-size 8 --device cuda:0 \
  --options model.loss.itc_temp_learnable=False model.loss.itc_temp=0.07 \
  --output ~/itc_gate_fixedtemp_baseline.json
```

Expect `"temp_learnable": false` and `"temperature": 0.07` in the JSON. If it
says anything else the override did not land — stop and report.

### Step 2 — probe run, 500 optimizer updates, temperature pinned (~30 min)

```bash
cd ~/Meta-CXR
CUDA_VISIBLE_DEVICES=0 ~/.venvs/meta-cxr-stage1-311/bin/python -u -m pretraining.train \
  --cfg-path pretraining/configs/mimic_cxr_full.yaml \
  --options run.output_dir=/home/phuong/probe_itc_temp \
            run.truncate_train=32000 \
            run.max_epoch=1 \
            run.eval_start_epoch=99 \
            model.loss.itc_temp_learnable=False \
            model.loss.itc_temp=0.07 \
  > ~/probe_itc_temp.log 2>&1
```

Why these numbers: 32,000 studies ÷ batch 8 = 4,000 microbatches = **500
optimizer updates** at accum 8, the point the gate is designed to be read at.
`eval_start_epoch=99` skips validation, which this probe does not need.
`warmup_steps` stays at 800 on purpose — the checkpoint being compared against
was also mid-warmup, and changing it would break the comparison.

Do **not** add batch overrides. Peak VRAM should sit near 14.5 GB / 15.9 GB, the
same as the run that already survived 13,000 iterations.

### Step 3 — gate the probe checkpoint (~7 min)

```bash
cd ~/Meta-CXR
~/.venvs/meta-cxr-stage1-311/bin/python -u scripts/check_itc_gate.py \
  --cfg-path pretraining/configs/mimic_cxr_full.yaml \
  --checkpoint /home/phuong/probe_itc_temp/mimic_cxr_full_blip2/checkpoint_last.pth \
  --split val --pairs 256 --batch-size 8 --device cuda:0 \
  --options model.loss.itc_temp_learnable=False model.loss.itc_temp=0.07 \
  --output ~/itc_gate_fixedtemp_500.json
```

## Expected — the numbers to read back

Report exactly these, nothing more:

1. From both JSONs: `temperature`, `temp_learnable`, `loss_itc`, `delta_nats`,
   `mean_rank_of_true_pair_i2t`, `mean_rank_of_true_pair_t2i`, `chance_rank`.
2. From `~/probe_itc_temp.log`, the `loss_itc` column at iters 0, 1000, 2000,
   3000, 4000 (both the window value and the running mean in parentheses), plus
   `loss_itm`, `loss_lm`, `time:` and `max mem` on the last line.

```bash
grep -a "Train: data epoch" ~/probe_itc_temp.log | awk 'NR%125==1' \
  | grep -oE "\[ *[0-9]+/[0-9]+\].*loss_lm: [0-9.]+ \([0-9.]+\)"
tail -1 ~/probe_itc_temp.log
```

**The decision rule, fixed in advance so it cannot be rationalised afterwards.**
The primary signal is the true-pair rank, because it does not depend on
temperature at all. Chance is 127.5.

| Outcome | Reading | What it means |
|---|---|---|
| **PASS** | i2t rank improves by **≥ 15 places** vs the Step-1 baseline, AND `delta_nats` ≥ +0.10 | The temperature was the problem. Pin it, resume the full run. |
| **FAIL** | rank moves < 15 places (the previous attempt moved 3.6) | The representation is the problem. `lambda_itc` → 0.0; the temperature was a symptom. |

15 places out of 256 is deliberately modest — 500 updates is early, and the bar
only has to separate "moving" from "the 3.6 places we already measured".

## Abort if

- Step 1 reports `temp_learnable: true` or a temperature ≠ 0.07 → the override
  did not reach the model. Stop, report, do not run Step 2.
- OOM. This is a **capacity finding**, not something to retry with a smaller
  batch: batch 4 starves ITM's hard negatives, which is the failure this whole
  configuration exists to avoid. Report the peak VRAM and stop.
- The machine hard-hangs again (it did at 22:25 on 08-18 and 01:20 on 08-19,
  both with nothing at all in `journalctl -b -1`). Report the last iter reached
  and stop; do not relaunch blind.
- Step 2 has not printed a `Train: data epoch` line within 15 minutes of launch.
  Startup legitimately takes ~7 min (weights + encoders) and stdout is
  block-buffered, so check GPU utilisation with `nvidia-smi` before calling it a
  stall — 0% for 6 minutes is the real signal.

## Do NOT

- Do not touch `lambda_itc/itm/lm`, the batch size, the encoders, or
  `warmup_steps`. The probe is a single-variable experiment.
- Do not resume `run_20260818_qformer`. That decision waits on this result.
- Do not delete `~/ckpt_at_500_updates.pth` or
  `~/run_20260818_qformer/` — they are the comparison point.
- Do not commit a patched `mimic_cxr_full.yaml` or `env_config.yaml`.
- Do not paste the training log. Summarize per the format in `../../AGENTS.md`.

## Known-stale test, ignore it

`tests/test_loss_weight_gating.py` has 2 failures asserting `lambda_itc == 0.0`;
the shipped YAML has carried 0.1 since the Q-Former was re-enabled on
2026-08-18. Pre-existing, unrelated to this probe — the CPU suite is 7 failed,
not the 5 that `CLAUDE.md` records. Do not "fix" it as part of this work.

---

## Execution report — 2026-08-19, minhphuong

**The launch command was issued twice, 50 seconds apart, so two Codex sessions
ran against one GPU.** Session A (09:50:18) executed the plan in full. Session B
(09:51:08) started while A held the card, OOMed in Step 1 and correctly aborted;
its report is kept below because it is accurate about itself. Both sessions were
given the same `-o` output path, so the file that survived was B's — which is why
the first thing this run looked like was a total abort. It was not.

- **commit run:** `a4897ddd9f992f68f6ce6c5dad4ded554b121246`
- **host:** minhphuong (RTX 5060 Ti 16 GB), `/mnt/drive1tb` mounted ntfs3 read-only

### Step 1 — baseline, untrained, temperature pinned at 0.07

`itc_gate_fixedtemp_baseline.json`, 09:52:17. `temp_learnable: false`,
`temperature: 0.07`.

| | |
|---|---|
| `loss_itc` | 5.6296 (chance ln 256 = 5.5452) |
| `delta_nats` | **−0.0844** |
| rank i2t / t2i | **113.34 / 124.97** (chance 127.5) |

### Step 2 — probe run, 500 optimizer updates

`~/probe_itc_temp.log`. 4,000 microbatches, **0.4354 s/it, 29:01 total**, no OOM,
no fallback, exit clean. `truncate_train=32000`, `max_epoch=1`,
`eval_start_epoch=99`, `itc_temp_learnable=False`, `itc_temp=0.07` — no other
override, batch and accum left at the shipped 8/8.

Train-side `loss_itc` (window value, running mean in parentheses), chance = ln 264 = 5.576:

| iter | `loss_itc` | `loss_itm` | `loss_lm` |
|---|---|---|---|
| 0 | 2.0503 (2.0503) | 2.0657 | 7.2815 |
| 350 | 5.3984 (5.4761) | 0.6247 (0.7862) | 5.4670 (6.3881) |
| 1650 | 5.0449 (5.3787) | 0.5684 (0.6469) | 3.3546 (4.5027) |
| 2950 | 5.7469 (5.2504) | 0.5291 (0.5934) | 2.9697 (3.9104) |

`loss_itc` oscillates on both sides of chance instead of pinning to it, which is
different from `run_20260818_qformer` — and turns out not to matter, see below.

### Step 3 — gate on the probe checkpoint, temperature pinned

Run twice. Session A's (`itc_gate_fixedtemp_500.json`, 10:22:33) landed 41
seconds after training exited, close enough to the final `checkpoint_last.pth`
write (10:21:51) to be worth distrusting, so it was **re-run cleanly on the idle
GPU** (`itc_gate_verify_500.json`). The two agree, so the original stands.

| | session A | verification re-run |
|---|---|---|
| `loss_itc` | 8.5654 | **8.5571** |
| `delta_nats` | −3.0203 | **−3.0119** |
| rank i2t / t2i | 116.02 / 110.91 | **117.44 / 112.93** |

### Verdict: **FAIL**

| | baseline | after 500 updates | rule |
|---|---|---|---|
| rank i2t | 113.34 | **117.44** — *worse by 4.1* | needs ≥ 15 better |
| rank t2i | 124.97 | **112.93** — better by 12.0 | — |
| `delta_nats` | −0.0844 | **−3.0119** | needs ≥ +0.10 |

At an identical, pinned temperature, 500 updates of training the contrastive
objective made it **2.93 nats worse on held-out data than random initialisation**.
i2t retrieval moved backwards. t2i improved 12 places, but the run-to-run spread
between two untrained builds is ~7 places (113.34 here against 120.21 measured on
the earlier build of the same untrained model), so 12 is not a safe signal on its
own — and it is contradicted by both other numbers.

**The temperature was a symptom, not the cause.** Pinning it removes the exploding
loss but does not make the representation separate pairs. `lambda_itc` should go
to 0.0.

### What I did NOT do

- Did not resume `run_20260818_qformer`; did not touch `~/ckpt_at_500_updates.pth`.
- Did not change `lambda_*`, batch, accum, encoders or `warmup_steps`.
- Did not remount anything or patch any config file — the mount was already
  correct, so the plan's sed/trap fallback was never used. `git status` clean.
- Did not evaluate ITM or LM. They are not what this probe was about, and the
  ITM number in particular is not readable while ITC is at chance, because ITM
  mines its hard negatives from the ITC similarity matrix.

### Session B's report, kept verbatim (duplicate launch, aborted)

- commit run: `a4897ddd9f992f68f6ce6c5dad4ded554b121246`
- command / status: Step-1 gate → exit 1
- result: ABORT in Step 1 on `torch.OutOfMemoryError` while moving PubMedCLIP to
  CUDA. The failed allocation was 2.00 MiB with 13.50 MiB free of 15.48 GiB.
  `nvidia-smi` then reported 15,485 MiB used / 363 MiB free, 79% utilisation;
  PID 20439 held 15,308 MiB. **That PID was a concurrently launched process
  running the plan's exact Step-2 command, not a process launched by this
  execution.** No gate metrics or PASS/FAIL decision were produced by it.
- what it did NOT do: skipped Steps 2 and 3 per the plan's OOM abort rule; did
  not retry, change batch/accumulation, edit either config, remount, alter any
  loss/encoder/warmup setting, or interfere with the concurrent process. It
  explicitly refused to claim the pre-existing baseline JSON as its own output.

That refusal is the right behaviour and is why the two runs could be told apart
afterwards. The lesson is about the launcher, not the agent: **two `codex exec`
invocations sharing one `-o` path and one GPU produce one true report and one
misleading file, and the misleading one wins the filename.**
