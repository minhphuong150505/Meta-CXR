# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this checkout is

`Meta-CXR-source/` is the **current** source repo (remote `git@github.com:minhphuong150505/Meta-CXR.git`,
branch `main`). It is a sibling of the older `../META-CXR/` checkout. The parent
`../CLAUDE.md` describes that older layout — where it disagrees with this file
(no `stage2/`, `safety/`, `runtime/`, `scripts/`, `training/evaluation/`,
`medgemma_inference/`; Stage-2 test counts), **this file wins for work done
inside this directory.**

## Who does what — Claude plans, Codex executes

Set by the user on 2026-08-19. It is a division of *roles*, not of judgement:
both agents are expected to push back when the other is wrong.

| | Claude Code (this session) | Codex |
|---|---|---|
| Owns | the plan, the source edits, `CLAUDE.md` / `README.md` / `struct/` | running things |
| Does | read code, design the change, write/apply the diff, decide what "done" means | `git pull`, venv/pytest, smoke runs, Stage-1/Stage-2 launches, `supervise_stage1.sh`, log triage on the host |
| Does not | drive long GPU runs or babysit SSH sessions | redesign the recipe, change losses/configs, or rewrite docs on its own initiative |

The handoff artifact is a file, not a chat message, so a later session can pick
it up cold: Claude writes `docs/handoff/PLAN-<YYYY-MM-DD>-<topic>.md` (exact
commands, expected output, abort conditions), Codex appends an
`## Execution report` section to the *same* file. See
`docs/handoff/README.md`.

**Error logs come back summarized, never pasted whole.** A Stage-1 log is
hundreds of MB of MetricLogger lines and pasting it burns the budget that is
supposed to pay for the fix. What Codex sends back:

1. the exact command and its exit status;
2. the first error, with ~20 lines of context and the final traceback frame —
   not the whole traceback chain;
3. the numbers that decide anything: `s/it`, `max mem`, the loss terms by name,
   `epoch`/`iter` at failure, `nvidia-smi` VRAM if it is an OOM;
4. where the raw log still lives on the host (`~/<run>.log`), so Claude can ask
   for one specific `grep` instead of the file.

If the summary is not enough, Claude asks for a named `grep`/`sed -n` range.
That is the only way raw log text should travel.

**When Claude Opus is out of usage or otherwise unavailable, escalate to a
Claude Sonnet 5 agent rather than stalling** — Sonnet executes an already-written
plan and triages logs at a fraction of the cost. Keep Opus for what actually
needs it: architecture, loss/recipe decisions, and reading an unexplained result.

Two things this arrangement does **not** relax:

- Whoever makes the behavioral change, the `CLAUDE.md` / `README.md` / `struct/`
  update ships in the same commit (see the two sections at the end of this file).
- Codex is bound by "Data handling — non-negotiable" exactly as Claude is. No
  report text, `subject_id`/`study_id`/`dicom_id`, split CSVs, `.npz`, `.jsonl`
  or checkpoints in a commit, in a handoff file, or in a summary sent back.

## The training host — there is only one, and it is local

All training runs on the user's own desktop, reachable over Tailscale. Its OS
hostname is `minhphuong`, which is what `hostname` prints.

⚠ **The Tailscale name has flipped back — `phuong-b760m-pro-rs-d4-wifi` no
longer resolves** (`Temporary failure in name resolution`, checked 2026-08-18
18:49). `tailscale status` lists the node as **`minhphuong`, 100.116.167.90**.
Do not hardcode either name: run `tailscale status` and use what it prints, or
SSH the IP directly, which is what actually worked:

```bash
ssh phuong@100.116.167.90        # or: ssh phuong@minhphuong
```

⚠ **THE MACHINE WAS WIPED AND REINSTALLED ON 2026-08-17.** Root filesystem
created 20:54 that day, Ubuntu 26.04 LTS, on a **different physical disk** from
the previous install. Verified over SSH 2026-08-18.

**The split is by disk, and it is the whole story: `/home` was destroyed, the
1 TB data drive was untouched.** The installer went to the other disk. So the
question for any artifact is simply *which drive was it on*.

| Lost — was on `/home` (ext4) | Survived — on `/mnt/drive1tb` (NTFS) |
|---|---|
| `abl_on/`, `abl_off/` — the explanation-loss A/B checkpoints and `.npz` predictions | `run_b16fast_20260814/` — `checkpoint_best`, `checkpoint_4`, `checkpoint_9`, `checkpoint_last` |
| `explanation_masks_v2/` — the rebuilt cache carrying `masks_bbox_*` | `run_gate3_20260815/` — `checkpoint_4`, `checkpoint_last` (**no** `checkpoint_best`) |
| The repo checkout | `datasets/explanation_masks/` — 2.6 GB, train/val/test, but **no `masks_bbox_*`** |
| The venv (**rebuilt 2026-08-18**, see below) | `mimic-cxr-jpg-full/` — images, metadata CSVs, and `processed/full_allviews_v2` |
| | `datasets/chexmask/`, `datasets/ms-cxr/` — the annotation sources |
| | `meta-cxr-ablation/`, `private-results/` — Table 5 evals, `xai_*`, `trainpred` |
| | `torch-cache/hub/checkpoints/blip2_pretrained.pth` + `resnet50` |

9 `.pth` files, 11.5 GB, survive on the data drive. `/home` is 984 MB used of
325 GB — a fresh home with only a `setup-dev-environment.sh` in it.

**What this costs, precisely:** the explanation-loss A/B (D-017) cannot be
re-scored or extended, because both arms' checkpoints and predictions were on
`/home`. Its **numbers** survive in git — this file, `README.md`, D-017 — but
nothing on disk backs them any more. The **strong** explanation term also lost
its only supervision: the surviving mask cache is the 2026-08-14 build, which
predates `masks_bbox_*`, so it must be rebuilt before that term does anything.
Everything else — Stage-1 checkpoints, the manifests, the dataset, the Table 5
evaluation — is intact and usable.

⚠ **Correction, recorded so the mistake is not repeated:** an earlier pass
searched only `/home/phuong` for `*.pth`, found none, and reported that every
checkpoint was gone. That was wrong. **Search the data drive too** — older runs
wrote there, and only runs after the 2026-08-15 mount change wrote to `/home`.

### Hardware and disks (verified over SSH, 2026-08-18)

| | |
|---|---|
| GPU | 1× NVIDIA **RTX 5060 Ti, 16 GB** — unchanged by the reinstall |
| OS | Ubuntu 26.04 LTS, root fs created 2026-08-17 20:54 |
| System disk | 465.8 GB — `p1` 1 GB EFI, `p2` **139.7 GB ext4 `/`** (UUID `c2eb4245-…`), `p3` **325 GB ext4 `/home`** (UUID `ba112a5a-…`) |
| Dataset disk | 931.5 GB — `p2` **930.7 GB NTFS, UUID `A4E6C088E6C05BE4`**, `p3` 781 MB NTFS |
| `/mnt/drive1tb` | NOT persistent — absent from `/etc/fstab`, so **every reboot loses it**. `ntfs3`, `ro`. 931 GB, 612 GB used, 319 GB free |
| Checkpoints | old ones survive on the data drive (read-only); **new** runs need `run.output_dir=/home/phuong/<run>` (ext4, 324 GB free) |

⚠⚠ **NEVER MOUNT THIS DISK BY `/dev/nvmeXn1p2` — THE NAMES SWAP BETWEEN
REBOOTS.** Observed both ways on the same machine: on 2026-08-18 morning the
dataset was `/dev/nvme0n1p2` and root was `nvme1n1p2`; after the reboot that
afternoon they had traded, root became **`nvme0n1p2`** and the dataset
**`nvme1n1p2`**. Either older command in git history therefore has a 50% chance
of naming the *root filesystem*. The UUID is stable and is the only safe
identifier:

```bash
# needs the user — no passwordless sudo on this host
sudo mkdir -p /mnt/drive1tb
sudo mount -t ntfs3 -o ro UUID=A4E6C088E6C05BE4 /mnt/drive1tb
df -T /mnt/drive1tb            # must say ntfs3, not fuseblk
```

Confirm the target before mounting with
`lsblk -o NAME,SIZE,FSTYPE,UUID,MOUNTPOINT` — the dataset partition is the
930.7 G `ntfs` one with no mountpoint.

**MIMIC-CXR is confirmed intact** (2026-08-18, after the user mounted it):
`mimic-cxr-jpg-full/files`, the metadata CSVs, and
`mimic-cxr-jpg-full/processed/full_allviews_v2` — the manifest export this
project requires — are all there. The mount is `ntfs3` read-only, which is the
recommended configuration, so nothing needs changing.

### NTFS history — kept, but unverified on this install

Everything below was measured on the *previous* install. The partition has not
been mounted once since the reinstall, so none of it is currently confirmed —
including whether the volume is still dirty. Treat it as a failure signature to
recognise, not as current state.

- **It is a Windows system partition**, not a data disk: `Windows/`,
  `Program Files/`, `pagefile.sys`, `hiberfil.sys` sat next to the 573 GB
  dataset. Treat write access as consequential.
- **The volume had real NTFS errors.** `dmesg` on 2026-08-13:
  `ntfs3(...): Mark volume as dirty due to NTFS errors` /
  `It is recommended to use chkdsk.` Not a stale hibernation flag; the kernel
  driver hit errors and flagged the volume. Only `chkdsk` from Windows repairs
  it — `ntfsfix` clears the dirty bit without fixing anything. The kernel
  `ntfs3` driver refuses rw while it stands; `ntfs-3g` (FUSE) mounted it anyway.
- **`ntfs-3g` stalled a real run — mount read-only with `ntfs3` instead.** On
  2026-08-15, five epochs in, `ntfs-3g` degraded until training effectively
  stopped: `time:` went from 0.23 to 1.0–1.4 s/it with **10–14 minute stretches
  of no log output**, GPU at 0% and 25–40 W, ~60 iter/min against ~260 healthy.
  The signature is unambiguous — check `ps -eo pid,stat,wchan:22` **before**
  suspecting the model, batch size, RAM or swap: `ntfs-3g` in state `D`, 10–11
  of 12 `pt_data_worker` in WCHAN `request_wait_answer` (the FUSE kernel wait),
  main process in `futex_do_wait` behind them. Twelve workers funnel through one
  single-threaded userspace daemon, so when it blocks, everything blocks. Full
  swap is a red herring: `vmstat` showed `si/so = 0`, full but static.
  An earlier measurement said the driver did not matter (2026-08-13, 200
  iterations at batch 6: `ntfs3` 0.5251 s/it vs `ntfs-3g` 0.5277, +0.5%) — that
  was taken while the volume was behaving and **does not generalise**.
- **Read-only `ntfs3` recovered most of the loss, not all.** Training only
  *reads* the dataset, so read-only costs nothing and removes the FUSE daemon.
  Over 1,550 iterations: median **0.231 s/it**, matching the 0.2347 healthy
  baseline, stalls gone — but a slow tail survived (6 of 32 sampled iterations
  over 0.5 s, p90 0.786, max 4.25) and epoch ETA settled at ~69 min against 55.
  Roughly 3× better than the crippled FUSE state, ~25% short of baseline. That
  residue is the volume's own damage, which only `chkdsk` addresses. `-o force`
  mounts rw but overrides the kernel's damage check on the disk holding the
  dataset, and only buys keeping checkpoints on the same drive.

### The venv — rebuilt 2026-08-18, verified on GPU

`~/.venvs/meta-cxr-stage1-311/bin/python` — **Python 3.11.16, torch 2.9.1+cu129,
torchvision 0.24.1+cu129, transformers 4.53.2, numpy 1.26.4**. Rebuilt after the
reinstall wiped the original, and confirmed on the card: `torch.cuda` available,
`get_device_capability() == (12, 0)`, `sm_120` present in `get_arch_list()`, and
a real 2048×2048 matmul executed on device. All 24 project imports resolve.

**The system Python is 3.14.4 and cannot be used.** Ubuntu 26.04 ships only
3.14, and the pinned dependency set is built for 3.10/3.11. The 3.11 interpreter
comes from `uv`, installed into `$HOME` with no sudo:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.11
uv venv --seed ~/.venvs/meta-cxr-stage1-311 --python 3.11   # --seed or there is no pip
PY=~/.venvs/meta-cxr-stage1-311/bin/python
$PY -m pip install torch==2.9.1 torchvision==0.24.1 \
    --index-url https://download.pytorch.org/whl/cu129
grep -vE '^(torch|torchvision)==' requirements-stage1.txt > /tmp/req.txt
$PY -m pip install -r /tmp/req.txt
```

⚠ **Do NOT `pip install -r requirements-stage1.txt` unfiltered.** Its header says
"CUDA 12.4 wheel channel" and it pins `torch==2.5.1` / `torchvision==0.20.1` —
kernels stop at **sm_90**, and this GPU is **sm_120**. That combination fails
late, after the whole model has loaded, with `CUDA error: no kernel image is
available for execution on the device`; and before that it fails more
confusingly still, because transformers 4.53 refuses `torch.load` under torch
< 2.6 (CVE-2025-32434), so PubMedCLIP raises a vulnerability error and the arch
mismatch never surfaces. That trap has now cost a full day once. Install torch
from the cu129 channel first, then everything else with those two lines removed.

⚠ **`iterative-stratification==0.1.9` ships a top-level `tests` package** into
site-packages, and a *regular* package beats the repo's namespace-package
`tests/` directory regardless of `sys.path` order. The symptom is the whole
suite aborting at collection:

```
ERROR tests/test_threshold_calibration.py
E   ModuleNotFoundError: No module named 'tests.test_classification_metrics'
```

Confirm with `python -c "import tests; print(tests.__path__)"` — if it points
into site-packages, that is it. The stray directory holds only that package's
own test file and nothing imports it, so move it aside:

```bash
SP=~/.venvs/meta-cxr-stage1-311/lib/python3.11/site-packages
mv "$SP/tests" "$SP/.tests-from-iterative-stratification.bak"
```

`iterstrat` itself still imports fine afterwards. **This comes back on every
reinstall of that package** — re-check it whenever the venv is rebuilt.

### Running anything means SSH-ing there

This directory is a **development checkout on a machine with no GPU and no
dataset**. Nothing that actually runs the project — training, evaluation,
inference, smoke tests, GPU-dependent scripts — runs here.

⚠ **`/mnt/drive1tb` is not in `/etc/fstab`, so EVERY reboot loses it**, and this
host reboots more than you would expect — it hung outright on 2026-08-18 22:25
with **no Xid, no NVRM error, no MCE and no thermal event** in `journalctl -b -1`,
and nothing logged at all at the moment of death (the signature of an abrupt
power loss or hard hang, not a GPU fault). Mounting needs the user: there is no
passwordless sudo.

⚠⚠ **The `nvme0n1` / `nvme1n1` names swapped AGAIN across that reboot**, which is
now the third observed swap and should settle the question permanently. Within a
single evening: at 22:00 the dataset was `nvme0n1p2`; at 23:40 `nvme0n1p2` was
the **root filesystem** and the dataset had become `nvme1n1p2`. Only
`UUID=A4E6C088E6C05BE4` is stable. A device-named mount command has roughly even
odds of naming `/`.

⚠ **The host is behind Tailscale SSH, which needs interactive browser approval.**
This is why a plain `ssh` appears to hang and why `-o BatchMode=yes` does not
fail fast: the connection completes the key exchange, then prints

```
# Tailscale SSH requires an additional check.
# To authenticate, visit: https://login.tailscale.com/a/<token>
```

and waits. An agent cannot click that link — start the SSH in the background,
surface the URL to the user, and wait for them to approve. Tailscale SSH also
presents **its own host key**, not the machine's `sshd` key, so the key differs
from any `known_hosts` entry predating it. That difference is expected and is
**not** evidence of a reinstall (the reinstall is a separate, real event — see
the top of this section).

The checkout is now at **`~/Meta-CXR`** (re-cloned 2026-08-18; the old
`~/Documents/2026/KLTN/Code_github/META-CXR-full-smoke-git` path is gone). The
host has **no SSH key for GitHub**, so it was cloned over HTTPS:

```bash
ssh phuong@100.116.167.90        # see the Tailscale-name note above
git clone https://github.com/minhphuong150505/Meta-CXR.git ~/Meta-CXR
cd ~/Meta-CXR
```

Same `origin` as this checkout (branch `main`), so the workflow is unchanged:
commit and push from here, then pull and run there. **Always pull before
running** — the host has repeatedly been one or more commits behind, and running
stale code silently produces results attributed to the wrong revision.

What may still be done locally: reading code, editing, `struct/` updates, and the
CPU test suite. Everything else goes over SSH.

**There is no cloud path.** GCP, rented L4, Kaggle and 2×3090 recipes were all
deleted on 2026-08-13 to cut cost — the user trains locally. Do not reintroduce
`cloud/`, `docs/cloud/`, GCS upload flows, or hardware-named configs. Anything
in git history describing them is dead.

`README.md` is authoritative and detailed (written in Vietnamese) — read it before
making claims about pipeline status. GPU evidence is limited: the tracked Table 5
Stage-1 **inference-only encoder ablation** completed 4/4 configurations on the
full test split, but this does not validate current Stage-1/Stage-2 training or
reproduce Stage-2 metrics. Do not generalize that evaluation result into a claim
that either training pipeline is GPU-validated.

## Commands

```bash
# CPU test suite. On a box without torchvision/transformers (verified 2026-08-14):
#   7 failed (re-measured 2026-08-19), and 2 modules excluded before collection
#   — test_blip2_negative_sampling.py and test_encoder_ablation.py, both of which
#   import model.lavis and therefore torchvision. Collection errors abort the run,
#   so ignore them explicitly to see the real result:
CUDA_VISIBLE_DEVICES="" python -m pytest tests/ -q \
    --ignore=tests/test_blip2_negative_sampling.py --ignore=tests/test_encoder_ablation.py
# The 7 failures are baseline, none of them a real defect:
#   test_native_independence   4 -- missing private env config
#   test_stage1_eval_hook      1 -- missing torchvision
#   test_loss_weight_gating    2 -- STALE: they assert lambda_itc == 0.0, which
#       stopped being true when the Q-Former was re-enabled on 2026-08-18. The
#       test encodes an intent that changed; leave it until the itc probe
#       settles whether lambda_itc goes back to 0.0, then fix it in that
#       direction rather than editing it twice.
CUDA_VISIBLE_DEVICES="" python -m pytest tests/test_explanation_metrics.py -q  # 7 passed
python -m pytest tests/test_stage2_prompts.py -q          # one file
python -m pytest tests/test_stage2_prompts.py -q -k negative_policy   # one test

# Syntax check for the stdlib-only packages
CUDA_VISIBLE_DEVICES="" python -m compileall -q \
    stage2 training scripts runtime safety tests medgemma_inference

# Lint (config in pyproject.toml; ruff lint only, no formatter pass)
ruff check .

# Pre-commit — the notebook privacy guard is mandatory, see "Data handling"
pip install pre-commit && pre-commit install

# Preflight before any GPU run (checks CUDA, RAM, disk, shm, paths, HF auth)
python scripts/vm_preflight.py --stage 1

# Stage 1 — the only supported recipe. Single GPU; there is no DDP variant left.
# Launch it PLAIN, not through torch.distributed.run: with one GPU and
# run.distributed=false, torchrun only sets RANK/WORLD_SIZE, which sends
# init_distributed_mode down its distributed branch for no benefit. This plain
# form is what produced the earlier checkpoints (wandb metadata: program =
# "-m pretraining.train"). torchrun does work now that run.dist_url is present,
# but it buys nothing here.
CUDA_VISIBLE_DEVICES=0 python -m pretraining.train \
    --cfg-path pretraining/configs/mimic_cxr_full.yaml \
    --options run.output_dir=/home/phuong/<run>
# output_dir MUST be on /home (324 GB free, ext4). /mnt/drive1tb does not exist
# on the reinstalled host, and when it is mounted again it should be read-only
# ntfs3, where a write fails. See "The training host".
# Launch it with NO batch overrides. The YAML ships batch 16 / accum 4
# (effective 64), measured 2026-08-14 as the best of {6, 16, 24, 32}. The old
# `batch_size_train=6 ... accum_grad_iters=11` command line is superseded.
#
# 16 is chosen because lambda_mpc carries ~25% of the loss and
# MultiPositiveContrastiveLoss draws negatives from the live microbatch only:
# at batch 6 that was ~3.3 usable studies, which is not a contrastive problem.
# Memory is 91.5% static (56.2 MiB per sample on a 3,615 MiB fixed cost), so
# even batch 32 uses 39% of the 16 GB card.
#
# A 10-epoch run is ~9.0 h measured end to end (epochs average 55 min with the
# explanation loss on, 49 min without; that term costs +10.5%, and the cost is
# switched rather than scaled -- lambda 0.125 and 0.25 time the same).
# Batch scaling has NOT been re-measured since the
# dataloader was fixed, so there is no current evidence about the best batch
# size for speed -- see "the dataloader was the bottleneck" below.
#
# ALL PRE-2026-08-14 CHECKPOINTS WERE DELETED on 2026-08-14, at the user's
# request, on the grounds that those runs went in the wrong direction. 15 files,
# 39 GB. Nothing on disk predates the current recipe. They were unloadable
# against it anyway: encoders.swin went false, so MHCAC saw 98 visual tokens
# instead of 147. The Table 5 numbers survive in results/ but cannot be
# reproduced or extended without retraining, because the checkpoint they were
# computed from is gone.
#
# A SECOND, INDEPENDENT BREAK landed 2026-08-14: each encoder now keeps its
# native token sequence (246 tokens, not 98) and every stream has its own
# positional encoding, so pos_enc went from one Parameter to a ModuleDict and
# cnn_downsampler left the state dict. Nothing from before that commit loads.
# Smoke: set run.truncate_train / truncate_val / truncate_test in the YAML.

# Stage 2 (single-GPU only — no DDP anywhere in Stage 2)
CUDA_VISIBLE_DEVICES=0 python training/run_medgemma_qlora.py \
    --train-limit 500 --val-limit 10 --test-limit 10 --no-upload --output-dir training/outputs/smoke

# Evaluation — calibrate on validation only, then score the test split
python scripts/calibrate_thresholds.py --predictions <val.npz> --objective f1 \
    --uncertain-policy ignore_uncertain --min-positive 20 --output <thresholds.json>
# --uncertain-policy is NOT optional here: it defaults to three_class in BOTH
# scripts, while training runs ignore_uncertain. Omitting it on the eval line
# after calibrating with ignore_uncertain silently scores the thresholds under a
# different label binarisation -- measured 2026-08-16 on the same npz:
# macro_auroc 0.6537 vs 0.7850, positive_macro_f1 0.7734 vs 0.8757. It does not
# warn; the report just prints "Uncertain policy: three_class" in its metadata.
python scripts/evaluate_stage1.py --predictions <test.npz> --thresholds <thresholds.json> \
    --uncertain-policy ignore_uncertain --output-dir <dir>
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_explanation.py \
    --checkpoint <checkpoint_best.pth> --cfg-path pretraining/configs/mimic_cxr_full.yaml \
    --split test --mask-cache-dir <mask-cache> --output-dir <private-results>/xai \
    --export-figures 12
# The cache these paths used to name was on /mnt/drive1tb and did not survive the
# 2026-08-17 reinstall — it has to be rebuilt. Keep outputs off the repo tree:
# PNG/NPZ from this script are patient data.
python scripts/evaluate_stage2.py --predictions <reports.jsonl> \
    --metrics bleu,rouge,meteor,cider,bertscore --skip-clinical-metrics --output-dir <dir>

# Manifest invariants (split leakage, required columns, section targets)
python -m training.dataio.validate_manifest --section-mode findings_and_impression
```

Environments: the cloud setup recommends **two separate venvs** to isolate the
Stage-1 workflow from the heavier Stage-2 QLoRA extras. The current lock files are
additive, not conflicting: `requirements-stage2.txt` includes
`requirements-stage1.txt` and adds accelerate/bitsandbytes/PEFT. `pyproject.toml`
deliberately declares zero runtime dependencies; it exists for tooling config
only, and the repo is not a src-layout package (modules import by path from the
repo root).

First run of anything: `cp configs/env_config.yaml.example configs/env_config.yaml`
and fill in paths — `local_config.py` raises `FileNotFoundError` otherwise.

## Architecture

Stage 1 (representation + classification) and Stage 2 (report generation) are
**deliberately decoupled**, and preserving that decoupling is the single most
load-bearing design constraint in the repo.

### Stage 1 — `pretraining/train.py`

Entrypoint registers LAVIS components via star imports, then hands off to the
vendored fork in `model/lavis/`. Data flow:

```
study (not image) → anchor + ≤1 auxiliary view
  → frozen encoders (BioViL-T 1408 + PubMedCLIP 768; SwinV2 and RadDINO off)
       BioViL   448 in  → 14x14 = 196 tokens          local, 32 px per cell
       PubMedCLIP 224 in → 1 CLS + 7x7 = 50 tokens    global + regional, 64 px
  → per-encoder ViewFusionModule on RAW pre-projection output  (mhcac/view_fusion.py)
  → FC projections → 1408, concatenated on the token axis (246 tokens)
  → MHCAC (mhcac/mhcac_12.py): 14 abnormalities × {Positive, Negative, Uncertain}
       student = image only (this is inference); teacher = image + report text,
       TRAIN ONLY, distilled into the student. Text attention is teacher-gated.
  → Q-Former, 32 query tokens, cross-attn every 2nd block; ITC (1024-sample
       negative queue) + ITM + LM
```

- Sampling is **one row per study**, not per image (`study_sampling: true`).
- **Every encoder keeps its native scale; nothing is pooled or dropped on the way
  into MHCAC.** This is the point of running two of them, and until 2026-08-14
  the code did the opposite: `cnn_downsampler` squeezed BioViL 14x14 → 7x7, and
  `_resize_patch_sequence` deleted PubMedCLIP's CLS (`50 == 49+1`), leaving two
  interchangeable coarse maps and no global token. Measured on real studies:
  CLIP's CLS is not recoverable from the patches that were kept
  (`cos(CLS, mean patch) = 0.21`), whereas BioViL's own global output *is* the
  patch mean (`cos = 1.0000`, `biovil_t/model.py:84`) — so the code was throwing
  away the irreplaceable one and keeping the redundant one.
  `Blip2Qformer._native_stream_layouts` now hands MHCAC a `StreamLayout` per
  encoder; `stream_layouts=None` restores the old path and is what swin/raddino
  recipes get. Cost of the extra 148 tokens: **0.2543 s/it vs 0.2529, +0.6%**.
  Pinned by `tests/test_stream_layouts.py`.
- **PubMedCLIP patch tokens are read through `post_layernorm` and then have their
  per-image mean subtracted.** HF returns the vision tower's `last_hidden_state`
  *without* the final LayerNorm (`modeling_clip.py:763-765` applies it to the
  pooled CLS only), and the raw residual stream has a fixed DC direction that is
  97% constant across the dataset: mean pairwise cosine between the 49 patches
  was **0.674** against BioViL's 0.0017, so attention over them came out nearly
  flat and the whole stream acted as a constant bias. LayerNorm alone only
  reaches 0.587; subtracting the mean reaches **-0.014**. Keep the split — token
  0 carries the global view, the 49 patches carry local deviation from it.
  Do **not** feed this encoder 448: measured worse (0.714) and it puts CLIP's
  interpolated position embeddings out of distribution while frozen.
- **`model.image_size` must equal `vis_processor.*.image_size` (448).** It is not
  self-evident: `init_vision_encoder` ignores it for biovil, so nothing read the
  value and the BLIP-2 default of 224 sat in the model config unnoticed. It now
  sets BioViL's token grid. Two guards, both fail loud — a `ValueError` naming
  the key when it is missing, and a token-count check in `MHCAC.forward`.
- **Positional encodings are per-encoder and init at `std=0.02`.** Tokens reach
  them L2-normalised to 1.0, so the old bare `torch.randn` gave the encoding norm
  27.8 and `cos(token+pos, pos) = 0.999` — at step 0 attention was driven purely
  by position. One shared encoding also meant BioViL token *k* and PubMedCLIP
  token *k* received the identical position vector.
- `ViewFusionBlock` zero-inits `W_O` and the last FFN Linear, so it is an exact
  identity at step 0 and a single-view checkpoint loads without regression.
  Studies with no auxiliary view are gated to zero, not dropped from the batch.
- **`view_consistency_loss` is soft and conditional as of 2026-08-16.** It used to
  be an unconditional symmetric KL, justified as "adding views must not change
  *which* abnormalities are predicted". That premise is wrong here: a lateral view
  exists to show what the frontal cannot, and 55.3% of train studies have one, so
  the old term charged the model for using the second view at all. Two knobs, under
  `model.view_consistency`: `margin` (hinge — divergence below it is free) and
  `confidence_gate` (waive the penalty where fusing made the cell *more* confident,
  i.e. sharpened rather than smeared). The gate is **detached**, or the model could
  minimise the term by manipulating the gate instead of the prediction. Both
  default to off, so the previous run stays exactly reproducible for ablation; prod
  sets `margin: 0.05, confidence_gate: true`, **not yet run on GPU**.
- `mhcac/loss.py` holds every loss; `ClassificationLoss` takes a `sample_mask` so
  unlabelled rows contribute nothing. `soft_target_kl_loss` detaches the teacher.
- **`lambda_mpc` was a constant for an entire run, and is now live.**
  `MultiPositiveContrastiveLoss` read tensors stashed *before* the only trainable
  module — raw frozen-encoder output, with the auxiliary branch under
  `torch.no_grad()` — so it had no parameter upstream and no gradient. Measured
  across four epochs: **3.9929 / 3.9949 / 3.9941 / 3.9943**, carrying 0.1 of the
  loss weight and ~22% of the reported total while teaching nothing. A per-encoder
  residual `StreamAdapter` (`vision_encoders/stream_adapter.py`) now sits between
  the frozen encoder and the stash point, applied to anchor and every auxiliary
  view, zero-init so it is identity at step 0; a SimCLR-style head gives the
  objective its own 256-d space. Smoke, 4 epochs: **4.0875 → 3.7786 → 3.3919 →
  3.0873**. `lambda_mpc` is now **0.02** with `mpc_warmup_steps` ramping it in.
  Pool per stream (PubMedCLIP CLS, BioViL patch mean), never over the 246
  concatenated tokens — that would weight BioViL 196/246 by token count alone.
  ⚠ The adapter adds **inference-path** parameters: earlier checkpoints cannot be
  resumed into it. Attaching the loss *after* fusion would be self-defeating, since
  the fused anchor already contains the auxiliary view.
- Production config `mimic_cxr_full.yaml`: **10 epochs**, `selection_metric:
  macro_auprc` on **validation only**, bf16 AMP, `save_freq: 5`, `warmup_steps:
  300` counted in **optimizer updates, not microbatches**. Thresholds are
  calibrated post-hoc from `checkpoint_best` validation logits. The test split is
  held out of checkpoint selection entirely.
- **The recipe trains the FULL pipeline again as of 2026-08-18 — the Q-Former is
  back on.** `lambda_itc/itm/lm` are `0.1` each and `lambda_teacher_cls/distill`
  are `0.5`, so `forward()` no longer skips the Q-Former and text-encoder passes.
  The checkpoint this produces is therefore a candidate for Stage-2
  `meta_cxr_qformer` modes, which the classification-only checkpoints were not.
  The skip logic itself is unchanged and still covered by
  `tests/test_loss_weight_gating.py`; it is simply not triggered now.

  ⚠ **This is not a settled decision, and the reason it was reverted twice is
  not addressed.** The vision-language block was switched off on 2026-08-13 for
  cost and again on 2026-08-16 because it sat at *exactly* chance: `loss_itc`
  6.9375 = ln(1024) at queue 1024, `loss_itm` 0.6420 against the 1:2 prior
  entropy of 0.6365 — the optimum for constant logits, i.e. a collapsed head, not
  a slow one. Nothing since then targets that collapse. **Run
  `scripts/check_itc_gate.py` at init and again after ~500 optimizer updates and
  require the true-pair rank gap to grow by >= 0.10 nats.** If it does not, these
  weights are feeding noise into the shared projector, the stream adapters and
  the view fusion — the exact parameters MHCAC reads — and belong back at 0.0.

  ⚠ **It forced batch 16 -> 8, and it cost Swin.** Measured on the 5060 Ti,
  2026-08-18, all with `lambda_itc/itm/lm: 0.1`:

  | encoders | batch | peak VRAM | s/it | outcome |
  |---|---|---|---|---|
  | 3 (swin on) | 8 | 11,238 then OOM | — | OOM in the **backward** pass, iteration 1 |
  | 3 (swin on) | 4 | 11,083 | 0.40 | fits; ~62 h/10ep; ITM negatives fall to 3 |
  | 2 (swin off) | 8 | **14,460 / 15,850 (93.4%)** | 0.42 | **shipped** — 700 iters + val, no OOM |

  Note *where* batch 8 with three encoders died: iteration 0 completed at 11,238
  MiB and the **retained gradient buffers** (301–319M trainable params) pushed
  iteration 1 over. An OOM on iteration 1 rather than 0 is the signature of that,
  not of a single oversized activation.

  So the real choice was Swin, or a Q-Former with enough negatives to learn
  anything — ITM draws hard negatives from the live microbatch only, and batch 4
  leaves it 3 candidates. **Swin went off** (`model.encoders.swin: false`), which
  also restores `_native_stream_layouts`: at two streams PubMedCLIP keeps its CLS
  token and each encoder keeps its native scale, the configuration this repo had
  already measured to be the better one.

  ⚠ **93.4% is tight but it is stable, not variable.** Peak was *identical*
  (14,460 MiB) at 150 and at 700 iterations, so it is not composition-dependent
  despite the aux filter making the number of encoded auxiliary rows vary 0–8 per
  batch. `nvidia-smi` shows ~15,119 MiB in the live run. Leave the batch alone;
  if it ever OOMs, supervise.sh falls back to 4/16 — which silently reintroduces
  the ITM starvation this configuration exists to avoid, so **treat an OOM
  fallback as a result to act on, not a recovery**.

  Loss trajectory over the first 700 iterations at batch 8 / swin off, which is
  the reason this configuration was preferred to batch 4:

  | | at 150 it | at 700 it | chance |
  |---|---|---|---|
  | `loss_itc` | 6.75 | **5.83 avg, 5.28 latest** | 5.58 = ln(264) |
  | `loss_itm` | 1.19 | **0.724 avg, 0.621 latest** | 0.6365 (1:2 prior entropy) |
  | `loss_lm` | 7.05 | **5.78 avg, 4.55 latest** | — |

  At batch 4 the same numbers went the *other* way (`loss_itc` rising 6.22 ->
  6.48, i.e. above chance). This is ~87 optimizer updates and settles nothing on
  its own — `scripts/check_itc_gate.py` at ~500 updates is still the gate — but
  it is the first time these objectives have moved in the right direction here.

  ⚠⚠ **THE GATE HAS RUN, TWICE, AND ITC IS AT CHANCE (2026-08-19).** Measured on
  val, on the pairs training actually scores (65.3% of studies — the rest have no
  usable FINDINGS), 256 valid pairs, chance rank 127.5:

  | arm | temperature | rank i2t | rank t2i | `delta_nats` |
  |---|---|---|---|---|
  | untrained | 0.07 pinned | 130.68 | 130.30 | -0.0833 |
  | 525 updates, temperature learned | 0.00796 | **127.43** | **127.65** | -1.1168 |
  | 500 updates, temperature pinned | 0.07 | **128.38** | **127.45** | **-0.0025** |

  Every arm lands on chance. The gate needs `delta_nats >= +0.10`; the pinned run
  returns **-0.0025**. Pinning the temperature removes the exploding loss and
  changes nothing else, so **the collapsing temperature was a symptom.**
  `~/gate_v2_*.json` on the host hold these three.

  ⚠ **Earlier numbers for this gate — `delta_nats` -24.52, and a claim that
  training made ITC "2.93 nats worse than random initialisation" — were a
  MEASUREMENT BUG and are retracted.** `check_itc_gate.py` scored all 256 loaded
  pairs while ignoring `generation_mask`, so ~29% of them were an image against an
  EMPTY string: unanswerable as queries, and as candidates a block of ~75
  identical text vectors inside every other row's softmax. That is what put
  `loss_itc` at 8.56 against a chance of 5.55. Fixed 2026-08-19 — the script now
  loads `pairs * oversample` studies, keeps the first `pairs` valid ones, and
  records `studies_scanned` / `valid_fraction` in its JSON. **Distrust any gate
  JSON without those two keys.** Training was never affected:
  `_image_text_contrastive` has always taken `generation_mask`, dropping invalid
  rows from the loss and `-inf`-masking them out of the candidate set.

  Train-side, the story was the same all along and less dramatic than the broken
  gate suggested: `loss_itc` ran 0.9-1.3 nats below chance (ln 264 = 5.576) from
  iter 1,000 to 5,000 of `run_20260818_qformer`, then returned to exactly chance
  from iter 6,000 and stayed there for 7,000 iterations. The pinned-temperature
  probe oscillated either side of chance (5.40 / 5.04 / 5.75 at iters 350 / 1650 /
  2950) rather than pinning to it — a difference that does not survive contact
  with the val measurement. Do not read the training curve as evidence about
  retrieval. Note the queue is 256, not 1024, so it fills in ~32 iterations; the
  early low values are not a queue-filling artefact.

  **THE REFERENCE IMPLEMENTATION DOES NOT TRAIN ITC AT ALL.** In
  `DasithEdirisinghe/META-CXR` — the repo this fork descends from, and the one
  behind the published paper — the entire vision-language block of
  `blip2_qformer.py` is commented out. `loss_itc` and `loss_itm` appear on no
  uncommented line, and Stage-1's returned loss is
  `cls_loss + 0.3*contrastive + 0.7*orth + 0.3*sparsity` — the MHCAC terms only.
  Their `blip2_pretrain_stage1.yaml` is consistent with that: `batch_size_train: 2`
  (ITC over 2 candidates has a chance of ln 2 = 0.69 and teaches nothing) and
  `load_pretrained: False`. **So the published Stage 1 is classification-only, and
  nothing in this lineage has ever demonstrated a working ITC on this architecture
  or this data.** Re-enabling it was never "restoring" anything.

  One concrete design flaw remains, if anyone wants to try again: the 256-entry
  negative queue is filled with detached features from the **live** encoder. MoCo
  and ALBEF use a momentum encoder for exactly this reason — stale keys produced
  by an encoder that has since moved are not comparable to current queries. The
  reference has no queue at all. This is a hypothesis, not a measured cause.

  **So `lambda_itc: 0.0` is the supported setting.** With ITC at chance the ITM
  number is not readable either: ITM mines its hard negatives from the ITC
  similarity matrix, so it has been solving an easier problem than the config
  intends. Full record, including the two concurrent Codex launches that made the
  first report look like a total abort, in
  `docs/handoff/PLAN-2026-08-19-itc-temp-probe.md`.
- **`model.loss.itc_temp` / `model.loss.itc_temp_learnable` pin the ITC
  temperature (added 2026-08-19).** Default `0.07` / `true`, which reproduces the
  historical behaviour: BLIP-2 learns it, and `blip2_pretrained.pth` supplies a
  trained ~0.0249. Setting `itc_temp_learnable: false` does two things, and it
  needs both — `self.temp` stops receiving gradient **and** the loss reads the
  non-persistent `itc_temp_fixed` buffer instead, because `load_state_dict` would
  otherwise restore a pretrained or resumed checkpoint's temperature over the
  config value. `scripts/check_itc_gate.py` mirrors the same rule and records
  `temp_learnable` in its JSON: **`delta_nats` scales with 1/temperature, so two
  measurements are only comparable at the same temperature**; the rank fields are
  scale-free and are the honest cross-regime signal. The script also takes
  `--options KEY=VALUE`, so a variant can be measured without editing the tracked
  YAML.
  Measured on GPU 2026-08-19: a 500-update probe at pinned 0.07 (batch 8 / accum
  8, `truncate_train=32000`, `max_epoch=1`, `eval_start_epoch=99`) took **29:01 at
  0.4354 s/it** with no OOM. The knobs work; what they buy is nothing, per the
  table above.
- **Padded auxiliary views no longer reach the encoders (2026-08-18).** The
  collater pads ragged studies with `torch.zeros_like(anchor)` and 44.7% of train
  studies have no auxiliary view, so `_encode_aux_streams` was spending roughly
  half of every auxiliary forward — across all three frozen encoders — on
  all-zero images whose output `ViewFusionModule` then gated to exactly zero.
  `real_aux_rows`/`scatter_aux_rows` in `mhcac/view_fusion.py` now select the
  real rows and scatter the results back, padded slots left at literal zero.
  Output shape and every caller are unchanged; the invariant that padded slots
  are never *read* is what makes it safe, and it is held by `ViewFusionModule`
  (masked out of the softmax, residual gated off) and by
  `MultiPositiveContrastiveLoss` (`cand_valid`). Pinned by four tests in
  `tests/test_view_fusion.py`, including one that puts random junk in the dense
  path's padding and requires the fused output to match. **Speedup not yet
  measured on GPU.** It does change how much dropout RNG is drawn, so a new run
  will not be bit-identical to an old one at the same seed.
- **`run.eval_start_epoch: 5` — the first five epochs train without being
  scored.** Validation over the full split is expensive and the early epochs are
  never the ones selected, so skipping them buys wall-clock time. The knob counts
  epoch *indices*, matching the training log: `epoch: [5]` is the sixth.
  Because `checkpoint_best` is only written inside the evaluation branch, the
  same knob guarantees no unscored epoch can ever be selected. Default 0
  restores the historical behaviour.
- **Early-stop patience counts scored epochs only.** The window is clamped to
  open at `eval_start_epoch` (`cur_epoch - max(best_epoch, eval_start_epoch)`),
  because `best_epoch` initialises to 0: measuring from it alone would spend the
  whole budget on unscored epochs and kill the run on its first scored epoch,
  logging "early stopping", which reads like convergence. With the shipped
  values (10 epochs, eval from [5], patience 5) the earliest possible stop is
  [10] and the last epoch is [9], so **early stopping cannot fire** — inert, not
  broken. Set patience to 4 or less to make it live. Covered by
  `tests/test_eval_start_epoch.py`.
- **The dataloader was the bottleneck, and `data: 0.0000` hid it.** Until
  2026-08-14 training ran at exactly the speed of the data pipeline: the loader
  alone measured **0.6520 s/batch** against the run's **0.6524 s/it**, while the
  model alone did **0.2099**. The GPU was idle 68% of the time (~37% utilisation
  in wandb). Trust neither `data:` nor GPU utilisation on their own here:
  MetricLogger's `data:` times how long `next()` blocks, but the Python loop
  runs ahead of CUDA asynchronously, so the batch has always arrived by then and
  the real wait lands in `time:`. The way to see it is to time the loader with
  no GPU work at all, and the model on a pre-loaded batch, and compare both to
  the run.
  The cause was `ReportDataset.remap_to_uint8`: a float64 min-max stretch run on
  the full 7 MP image before any resize, moving ~336 MB of DRAM per image. It
  was 45.7% of an 83.6 ms study -- more than JPEG decode -- and being
  bandwidth-bound it made twelve workers starve each other, delivering 24.5
  studies/s against the 143/s their single-threaded cost predicted. On 8-bit
  input the function is exactly a 256-entry lookup table, and on this dataset
  the table is the identity (every image already spans [0,255]), so it was
  spending 24 ms per image to return its own input. The uint8 fast path is
  bit-identical, pinned by `tests/test_remap_to_uint8.py`. Result: loader
  0.6520 -> 0.1722 s/batch (92.9 studies/s), full loop **0.6346 -> 0.2347 s/it,
  2.70x**, `next(it)` from 26.5% of wall to 0.1%, and the step is now 94.1% of
  wall. A 10-epoch run went from ~25 h to **~9.1 h**.
  Next in line if more is wanted: JPEG decode, now 66% of the item. `Image.draft()`
  decodes at 1/4 scale in the DCT domain and would cut it several-fold, but that
  changes the resampling chain and therefore the pixels -- it is a preprocessing
  change, not a free one, and has not been done.
- Note the `data:` block must sit **inside** `model:` — `Config` merges only
  `run`/`model`/`datasets`.
- `mimic_cxr_2gpu.yaml` was deleted with the retired cloud/Kaggle recipes. Do
  not recreate or copy it from history; `mimic_cxr_full.yaml` is the only
  supported Stage-1 recipe.

#### Throughput with three encoders and the teacher on — measured 2026-08-18

The **0.2347 s/it** baseline quoted throughout this file was measured with
**two** encoders and the privileged-text teacher **off**. It does not describe
the configuration that ships today. With `encoders.swin: true` and
`lambda_teacher_cls`/`lambda_distill` at 0.5, on the 5060 Ti at batch 16 /
accum 4, 800 iterations into a real run:

| | |
|---|---|
| step time | **1.0364 s/it**, range 1.0358–1.0370 over 12 consecutive samples |
| epoch | 13,922 iters → **4.0 h** |
| 10-epoch run | **~40 h**, plus five validation passes from epoch [5] |
| `max mem` | **9,433 MiB** torch, 12,061 MiB by nvidia-smi — 74% of the card, no OOM |

That is **4.4x the two-encoder/no-teacher baseline**, turning a ~9 h run into
~40 h. The split between swin and the teacher branch has **not** been measured
separately, so do not attribute it. What can be said: it is not I/O. The
variance across samples is ±0.0006 s and `max mem` is constant, with none of the
slow tail that the ntfs3 volume damage produces (p90 0.786, max 4.25 s/it) — so
this is compute, not the disk.

Sanity signals from the same 800 iterations, all healthy:
`loss_itc/itm/lm` exactly 0.0000 (the gating works, the Q-Former is skipped),
`loss_mpc` falling 3.35 → 2.66 within the epoch (the `StreamAdapter` fix holds),
`loss_distill` **0.0084–0.0115** against the degenerate **1.4e-08** that got the
teacher switched off before, and `lr` 2.6e-5 at optimizer update 200 of an
800-update warmup, which is exactly the linear ramp.

#### Running Stage 1 unattended — `scripts/supervise_stage1.sh`

**In the repo as of 2026-08-18**, so it is versioned with the recipe it launches.
The host's `~/supervise.sh` is a *deployment* of it — after changing either,
`scp`/`cp` it across, or the two silently diverge:

```bash
cd ~/Meta-CXR && git pull && cp scripts/supervise_stage1.sh ~/supervise.sh
```

Launch it detached, or it dies with the SSH session:

```bash
OUT=$HOME/<run> LOG=$HOME/<run>.log WID=<wandb-id> \
    setsid nohup bash ~/supervise.sh >>$HOME/<run>.supervise.log 2>&1 &
```

Three traps it encodes, each found the hard way on 2026-08-18:

- **Startup is not a stall.** The watchdog counts `Train: data epoch` lines, and
  there are none while the run downloads `blip2_pretrained.pth` (1.9 GB) and
  builds the encoders. A single timeout applied from launch kills a healthy run
  ~12 minutes in. Startup needs its own, much larger budget until the first
  training line appears.
- **stdout is block-buffered.** The run writes to a file, not a tty, so
  MetricLogger `print()` lines arrive in ~8 KB bursts — measured at one flush per
  ~75 s, 0 → 17 lines at once. `[INFO]` lines come through immediately because
  `logging` writes to stderr, which is why the log looks alive while the line
  count sits at zero. The line count is therefore a **coarse** progress signal
  and a run that merely slows stretches the gap proportionally.
- **GPU utilisation is the signal that actually catches the known failures.**
  Both the DataLoader IPC deadlock and the ntfs-3g stall sat at 0% while the
  process stayed alive. It is unbuffered and immediate. The script trips on
  6 minutes of <=5% utilisation independently of the log.

It also falls back on OOM (halve batch, double accum, effective batch
unchanged), resumes from `checkpoint_last`, and aborts after three consecutive
failures that produce no new checkpoint. `ADOPT_PID=<pid>` attaches it to an
already-running train process instead of starting a new one — which is how the
watchdog itself can be fixed without losing a run in progress.

Three fixes landed 2026-08-18 when it was made to supervise *continuously*:

- **Progress is a checkpoint WRITE, not a checkpoint FILE.** It counted files
  with `ls | wc -l`, which was right only while checkpoints were written once per
  epoch under distinct names. Since `run.save_every_iters` landed,
  `checkpoint_last.pth` is rewritten **in place** every 1000 iterations, so the
  file count sits at 1 for a whole ~4 h epoch: three restarts inside one epoch —
  precisely the window mid-epoch checkpointing exists to protect — would have
  tripped the "nothing is being learned" abort on a run that was progressing
  fine. It now tracks the newest `*.pth` mtime, which moves on every write. That
  same signal also resets the stall clock, since a checkpoint write is forward
  progress even while the log sits in a buffer.
- **`MAX_ATTEMPTS` defaults to 0, meaning unlimited.** The old default of 8 could
  retire a healthy multi-day run purely on restart count. The give-up rule is
  now progress-based only, which is the honest test.
- **An OOM no longer burns the no-progress budget.** It is a capacity finding,
  and the next attempt is a genuinely different configuration.
- `train_lines()` returned **two** lines (`"0\n0"`) when the log had no match,
  because `grep -c` prints `0` *and* exits 1, so the `|| echo 0` fired anyway.
  Every `[ "$n" -gt … ]` against that raised "integer expression expected" and
  evaluated false — which happened to look like "no progress" during startup and
  hid the error in stderr.

`blip2_pretrained.pth` re-downloads on a fresh `/home` even though a copy
survives at `<data drive>/torch-cache/hub/checkpoints/`. Symlink
`~/.cache/torch/hub/checkpoints` at it to skip 1.9 GB.

#### Explanation-aware loss and XAI evaluation

**OFF IN PRODUCTION as of 2026-08-17** — `lambda_explanation` and
`lambda_explanation_strong` are both `0.0` in `mimic_cxr_full.yaml`. The user
retired the approach after a controlled 5-epoch A/B: it did not help
classification, and its own saliency metric came out at chance in **both** arms
because the encoders are frozen. The code, the mask cache and
`scripts/evaluate_explanation.py` are all kept and still work.

**The full record — the A/B tables, the mask-area baseline that makes the
saliency numbers readable, the weak/strong split, the cache format, the
PhysioNet download traps and the two measurement bugs to fix first — is in the
`explanation-loss` skill (`.claude/skills/explanation-loss/SKILL.md`).** Load it
before re-enabling either lambda, touching the mask cache, or quoting any
saliency precision. ⚠ Never quote a saliency precision without the mask-area
baseline beside it: 0.25 reads like a result and is chance.

### Stage 2 — `training/run_medgemma_qlora.py`

`google/medgemma-1.5-4b-it`, 4-bit NF4 QLoRA, single process / single GPU,
checkpoint selected by validation cross-entropy.

`training/pipeline_modes.py` is stdlib-only and names each architecture explicitly
(the old `--image-mode {native,qformer,both}` was renamed because it described an
implementation detail and made the hybrid easy to mislabel as "native MedGemma"):

| Mode | Stage 1? | Visual path |
|---|---|---|
| `medgemma_direct` (default) | no | MedGemma's own image tower + projector |
| `meta_cxr_qformer` | yes | Q-Former soft tokens |
| `meta_cxr_qformer_with_mhcac_prompt` | yes | soft tokens + structured P/N/U text |
| `text_only_language_prior_ablation` | no | none — the only mode with `requires_multimodal=False` |
| `both_for_ablation` | — | runs direct then qformer sequentially |

Two further modes are inference-only against external checkpoints:
`pretrained_medgemma_findings_first` (routes through `medgemma_inference.run_pretrained_findings`,
not the fine-tuning CLI) and `pretrained_medgemma_impression_phase2` (declared but
disabled by a runtime guard).

**The independence invariant, enforced by `tests/test_native_independence.py`:**
every LAVIS/Stage-1 import lives in `training/stage1/lavis_loader.py` and nowhere
else. A Stage-2 entrypoint must never import it at module scope — only inside the
branch that has already decided it needs Stage 1. `training/dataio/manifest.py`
reads the split CSVs with pandas alone for the same reason.

Q-Former modes are **`findings_only`**; the native route also supports
`impression_only` and `findings_and_impression` (the default). The run errors out
rather than silently substituting one section for the other.

Soft-token conditioning **substitutes** projected Q-Former vectors at
`<qformer_soft_token>` positions — it does not sum into them
(`training/medgemma/soft_tokens.py`). Getting the per-row indexing wrong is
silent: loss still falls, but every study is described using another study's
image. Hence per-row shape validation and fail-closed behavior.

### Prompt v2 — `stage2/prompts/`

`PromptBuilder` is the single prompt entry point for train **and** inference, and
touches no model, tokenizer or torch, so parity is byte-for-byte testable. It
emits `PromptPart`s plus a prompt version, config hash and template hash recorded
in artifact metadata. Five visual modes in `schemas.py`: `native_anchor_only`,
`native_anchor_guided`, `native_multiview`, `qformer_visual_only`, `qformer_guided`.
Only guided modes see structured Stage-1 predictions, and they are phrased as
fallible auxiliary cues, never ground truth — `qformer_visual_only` receives no
labels at all, which is what keeps the ablation uncontaminated.

Opt-in via `--prompt-config configs/stage2_prompt_v2.yaml`; without the flag the
legacy prompt is used. The prompt prefix is masked out of training labels, and
the soft token is added to `bad_words_ids` at generation.
`configs/prompt_ablation/P1..P9.yaml` drive `scripts/run_prompt_ablation.py`.

### Evaluation — `training/evaluation/`, driven by `scripts/evaluate_stage*.py`

There is **no top-level `evaluation/` directory**; older docs that cite one are
stale. The core (classification metrics, AUROC/AUPRC, threshold calibration,
bootstrap, BLEU/ROUGE, error analysis) needs only numpy, so it runs wherever the
tests do. Plots need the `eval-plots` extra; METEOR/CIDEr/BERTScore need
`eval-generation`.

Clinical metrics (CheXbert, RadGraph, RadCliQ, RadFact) are **deliberately not
installable extras** — they're research code behind separate licences, not
reproducible pins. `training/evaluation/clinical.py` raises
`MissingOptionalDependency` naming the package, or `NotImplementedError` if the
package is present but the adapter was never validated against published
reference scores. **A missing clinical metric is reported as unavailable, never
as a score of 0**, and lexical metrics must not be presented as clinical accuracy.

### `safety/` and `runtime/` (both stdlib-only)

`safety/pipeline.py` orchestrates draft report → parsed claims → verification →
final report or abstention; it holds no verification logic itself so a real
phrase-grounding model can be swapped in via the same protocol. Its output record
carries no `subject_id`/`study_id`/`dicom_id`/path/reference text, so it is safe
to persist. `parse_coverage` is surfaced on purpose: a pipeline that parsed 2 of
12 sentences has not checked the report however clean its numbers look.

`runtime/budget.py` bills wall-clock time against an hourly rate (a stalled GPU
costs the same as a busy one) and carries `prior_elapsed_seconds` so resumes
cannot reset the ceiling. It only ever stops a run — it never downgrades the
model or enables extra sections. `runtime/device.py` resolves device/dtype from
config or the machine; nothing hardcodes `cuda:0`.

## The labels, the manifest, and how a checkpoint gets picked

Three things here are easy to get wrong and produce a run that looks healthy.

**The model can now say "nothing to report" — `model.loss.lambda_gate`.** MHCAC
carries a second head, one binary output per abnormality, answering *will the
report mention this finding at all?* It is the only consumer of the **79.5%** of
the CheXpert matrix that is blank; every other loss masks those cells out, which
is right for a Positive/Negative/Uncertain question but left the model forced to
pick one of three classes for all fourteen findings on every image. Measured
consequence before the head existed: **10.8 of 14 findings called Positive per
test study**, and **every single one of 3,269 studies labelled `No Finding =
Positive` while 8.8 other findings were flagged on it** — 100% self-contradictory
output. `No Finding` stays in the gate on purpose even though it is excluded from
the classifier: *was it mentioned* is *did the radiologist call this study
normal*, 74,305 against 148,453, which is well posed.
Gate weights are `(n_not_mentioned / n_mentioned) x kappa_gate`, kappa 2, capped
at 10 — the cap is load-bearing, five labels have raw ratios of 17–157 and
uncapped they simply relocate "always answer the majority class" into the gate.
`mention_mask` drops studies with no CheXpert record: their blank pattern is
unknown, not empty. Pinned by `tests/test_mention_gate.py`.

**A hierarchical alternative to both of the above exists and is off by default:
`model.loss.lambda_mention_conditioned_cls`.** The gate and the classifier are two
heads that never met — the gate could answer "never mentioned" while the classifier
answered "Positive", and nothing reconciled them, because the gate's prediction fed
its own BCE and nothing else. It was added to stop over-calling and **it did not
work**: on the run that finished 2026-08-16 (test split, calibrated thresholds,
`ignore_uncertain`) macro specificity was **0.2637**, recall 0.9021 against
precision 0.6835, with specificity ~0 on Support Devices, Fracture, Lung Opacity
(0.017) and Atelectasis (0.100). Setting this weight > 0 makes them one likelihood
— `-log(1-m)` when unmentioned, `-log(m) - log(q[y])` when mentioned — and
exports the four-state joint alongside — `P(blank) = 1-m`, `P(Neg) = m·q_neg`,
`P(Pos) = m·q_pos`, `P(Unc) = m·q_unc` — as `mention_marginal_log_probs`.

**`classification_logits` stays `q`, the polarity distribution conditional on the
finding being mentioned.** Substituting a three-state marginal there was a category
error and cost a GPU smoke to find: the CheXpert P/N/U metric masks blank cells, so
it scores polarity *given* mention, which is `q`. Worse, argmax over that marginal
can only choose Positive when `m > 1/(1 + q_pos - q_neg)`, which is ≥ 0.5 even for a
perfect conditional classifier and 0.55–0.8 in practice — so with 79.5% of cells
blank, a correctly calibrated sparse gate makes Positive **mathematically
unwinnable**. Validation F1 sat at exactly 0.000000 for every epoch while the old
mode reached 0.62 on identical data. Report-time emission is meant to be two-stage:
open the gate on a per-label threshold fitted on validation, then read the class off
`q`. Do not reintroduce an argmax over the marginal, and note that validation takes a
literal three-way argmax (`image_text_pretrain.py:114`), not a calibrated threshold.

The hierarchy reads `mention_mask` from the batch. It briefly read
`has_chexpert_label`, which the dataset does not emit — the absent key fell through
to `default=True` and trained every unmatched study as fourteen "not mentioned"
cells. Its **conditional class term** carries no inverse-frequency or kappa
weights on purpose: the operating point belongs in the thresholds this project
already calibrates on validation. Its **mention term** does carry
`mhcac.mention_conditioned_pos_weights` (`alpha = n_not_mentioned / n_mentioned`,
capped at 10, no kappa). Unweighted, it charged hiding a finding the radiologist
*did* write about exactly as much as mentioning one they did not — measured
**1.00x**, against 4–10x in the gate BCE it replaces — and with 79.5% of cells
blank that symmetry makes silence the majority answer. ⚠ The weight means `m` is
no longer a calibrated mention probability: its odds are inflated by alpha, so a
raw 0.5 threshold means a true probability of `1/(1+alpha)`. Recover with
`logit(p) = logit(m) - log(alpha)`, or fit the gate threshold on validation. Enabling it **requires** `lambda_cls: 0.0` and
`lambda_gate: 0.0`; the constructor raises otherwise. **Smoke-tested on GPU only** (600 studies, 3 epochs): F1 0.3640 → 0.4611 → 0.5325, against 0.2805 → 0.5164 → 0.6249 for the old mode on identical data. That smoke shows the mode trains; it is far too small to say which mode is better.

**Classification weights are full inverse frequency now, not the square root.**
sqrt deliberately under-corrects and the residual was not small: 12 of 14 labels
still pushed toward Positive, up to **5.36x** for Atelectasis, and measured
specificity tracked it almost monotonically — Atelectasis 0.000, Support Devices
0.000, Lung Opacity 0.017, against 0.675 for Edema whose ratio was already 1.02.
`(n_neg/n_pos)` brings every label to 1.00, then a per-label `kappa` (4 for
Pneumothorax, 3 for Pneumonia/Edema/Consolidation, 2 or 1 for the rest) applies
the clinical preference.

⚠ **`w_pos < 1` on 9 of 14 labels is not a light penalty, and raising it would
make things worse.** Blank masking inverted the imbalance, so positives are the
majority and `(n_neg/n_pos)` is below 1 for most labels; every shipped `w_pos`
sits at exactly `balance x kappa`, i.e. a missed positive is already charged 1–4x
a false alarm *once imbalance is accounted for*. Only the ratio to `w_neg` means
anything — the absolute value does not. Given macro specificity of 0.2637 with
four labels near zero, the lever that would actually raise specificity is
**reducing kappa toward 1** and then fitting thresholds on validation against an
explicit specificity target, not raising `w_pos`.

**That kappa table is a proposal and needs a clinician's
sign-off before any published claim**, and kappa does not create skill: AUROC is
fixed at 0.7776 and kappa only picks an operating point on that curve.

**`No Finding` is back in the classification head as of 2026-08-15**
(`model.mhcac.excluded_labels: []`), because the mention gate changed what it is
worth. Only 33.4% of studies have it written at all, so *was it mentioned* is a
real binary — 74,305 against 148,453 — and the 49,760 train studies carrying it
as their only label are no longer stranded: they train the gate across all
fourteen cells whatever this list says. What that does **not** recover is a
learnable Positive/Negative split. Among the mentioned cells, all 74,305 are
Positive and none are Negative, in every split, so the classification head can
only learn a constant for it and it re-enters the per-pathology table at a free
F1 of 1.0000; `run.include_meta_labels: false` keeps it out of the macro. State
this limitation rather than paper over it: some of the 148,453 not-mentioned
studies genuinely are normal and simply were not described that way, so the
gate's target here is noisy in one direction — missing positives, never spurious
ones. `excluded_labels` remains the switch, and it applies to the classification
head only; the gate always covers all fourteen.

Historical note, since the reasoning matters more than the setting: it was excluded on 2026-08-14 (`model.mhcac.excluded_labels`).
The labeler sets it to 1.0 when the report describes no abnormality and leaves it
blank otherwise -- it never emits 0.0 -- so across the whole dataset it is
74,305 positives and **zero** negatives on train, 582/0 on val, 568/0 on test.
Under blank masking every surviving cell is a positive, so the loss can only
teach a constant. The cost is real and was not obvious: **22.3% of train studies
(49,760) carry it as their only label** and are left with no usable cell at all,
dropping out of the classification loss and of the explanation loss, which needs
a positive. 50,484 of 227,827 labelled studies end up empty. They still feed
`lambda_mpc` and view consistency. Two validity flags exist for this reason --
`_has_chexpert_label_raw` (pre-exclusion, guards the join) and
`_has_usable_label` (post-exclusion, becomes the sample mask); collapsing them
makes the join-integrity check fire on every intentionally excluded row.
Pinned by `tests/test_excluded_labels.py`. The untaken alternative is to derive
No Finding negatives from the presence of any other positive.

Two more labels are degenerate on the **test** split specifically and should not
be quoted: `Pleural Other` has 63 positives and **0 negatives** there, and
`Fracture` has 89 and **3**. `positive_macro_f1` already excludes the meta
labels `No Finding` and `Support Devices` but still includes `Pleural Other` at
a free F1 of 1.0000 -- 0.8745 becomes **0.8631** without it. More broadly,
`positive_macro_f1` is not a usable headline on this label distribution: for six
labels the model scores recall exactly 1.0000 with precision equal to the
prevalence, i.e. it predicts positive for everything, and `Fracture` reaches
F1 0.983 on an AUROC of **0.442**. Quote `macro_auroc` (0.7776) and per-label
AUROC with prevalence beside it.

**A blank CheXpert cell is masked, not negative.** The export leaves a cell blank
when the labeler found no mention of the finding, which is not the radiologist
ruling it out. 79.4% of the label matrix is blank, so the old `.fillna(0)` made
roughly nine in ten "negatives" an absence of evidence. Blanks now carry
`ReportDataset.IGNORE_LABEL = -100` and are dropped per cell; `ClassificationLoss`
and the eval confusion matrix already kept only `labels >= 0`, so nothing
downstream needed changing. What it costs, measured on `full_allviews_v2`:
2.86 of 14 labels survive per study, 31% of studies keep exactly one, and the
imbalance **inverts** — positives are the majority for 12 of 14 findings
(Atelectasis: 44,718 positive against 1,502 negative). `default_class_weights` in
`blip2_qformer.py` were recomputed downward (0.18–2.01) and must be recomputed
again if the manifest or this policy changes. `No Finding` has zero negatives by
construction and is single-class under this policy; `include_meta_labels: false`
already keeps it out of macro metrics. Pinned by `tests/test_blank_label_masking.py`.

**The explanation-mask cache was built, and its coverage decides what you can
claim — but it did NOT survive the 2026-08-17 reinstall.** It lived on
`/mnt/drive1tb` and is gone; rebuilding takes ~25 minutes and streams the 13 GB
CheXmask export, and it needs the dataset disk mounted first. The format: a
cache directory holds `masks_<split>.npy` (uint8 `[N,112,112]`, values 0/255)
plus `index_<split>.json` keyed by the anchor `dicom_id`. Built 2026-08-14 from
CheXmask + MS-CXR against `full_allviews_v2`, and verified then: rows unique and
in range, no empty or all-ones mask, lung coverage p50 ≈ 33-35% on every split.
The coverage table below describes that build, so re-verify after any rebuild.

`build_explanation_masks.py` reimplements anchor selection instead of calling
`build_study_index`. The two agree exactly on real data — 0 cache keys that are
not dataset anchors, all 208,987 train keys found — but that is a checked fact,
not a guarantee. Recheck it if either side changes, because a divergence means
`_read_explanation_mask` silently returns `valid=False` and the term quietly
trains on less data than you think.

What the coverage allows:

| | train | val | test |
|---|---|---|---|
| studies with a mask | 93.8% | 93.5% | 90.5% |
| with >=1 positive finding | 60.6% | — | 77.6% |
| **both -> loss actually fires** | **56.7%** | — | 69.8% |
| **of those, real MS-CXR box** | **823** | **5** | **138** |

The cost is now measured, not estimated. 200 iterations at batch 6 on the
5060 Ti, same config and manifest, the term the only difference: **0.580 s/it
against 0.560, +3.6%**, peak VRAM 6,136 MiB against 6,112 (37% of 16 GB). A
10-epoch run is ~60 h with it and ~58 h without. The first estimate of 30-50%
was wrong because the Grad-CAM backward only traverses the MHCAC subgraph, not
the frozen encoders, and the term fires on 56.7% of studies. Evidence:
the only population that supports "the model looks at the pathology" is the box
one, and that is **138 test studies** — val's 5 cannot calibrate or select
anything. The lung-mask population is 30x larger but only supports the weaker
claim "the model stays inside the lungs"; `ExplanationSummary` keeps the two
apart on purpose.

**Use `processed/full_allviews_v2`, nothing else.** Two stale exports sit beside
it on the training host, and `meta-cxr-manifests-upgraded-20260806` was wired up
despite the name. Tell a stale export by any of: `extraction_method` is a single
constant (`legacy_preprocessed` appears nowhere in this repo), `target_valid` is
100% True (the length filter never ran), or `impression_valid` is missing (Stage 2
then dies in `assert_columns`). v2 reconciles exactly against the source —
377,110 rows = all of metadata, zero duplicate `dicom_id`, split identical to the
official MIMIC split, `findings_clean` empty iff `target_valid` is False, and
length bounds taken from train only.

**`selection_metric: loss`, and it has a known bias.** Validation loss is a
weighted sum dominated by the frequent labels, so a model that stops predicting a
rare finding entirely can outscore one that finds it sometimes. `macro_auprc` is
per-label and threshold-free and does not have that failure mode; it was replaced
because val is thin for two labels (Pleural Other 14 positives, Fracture 16).
Both are logged every scored epoch — check which epoch each would pick before
quoting either. `selection_mode` is deliberately absent from the YAML so
RunnerBase infers `min`; an explicit `max` left behind would keep the worst epoch.

**Changing `selection_metric` on a resume silently disables `checkpoint_best`
unless you also neutralise `best_agg_metric` in the checkpoint.** `train()`
initialises it to `+inf` under mode `min` and `-inf` under `max`
(`runner_base.py:694`), then `_save_checkpoint` persists whatever it holds — so
a `loss` run that has not yet scored an epoch writes **`best_agg_metric: inf`**
into `checkpoint_last.pth`. Resume that checkpoint under an F1/AUPRC metric and
`runner_base.py:719` restores the `inf`, after which `_metric_improved` tests
`value > inf + min_delta` and is **False forever**: training looks completely
healthy and `checkpoint_best.pth` is never written. Verified on the live
checkpoint 2026-08-15 (`epoch=4, best_agg_metric=inf, best_epoch=0`).
The fix is to set the key to `None` before resuming — `runner_base.py:719`
guards with `if resumed_metric is not None`, so `None` leaves the freshly
initialised `-inf` in place:

```python
ck = torch.load(p, map_location="cpu", weights_only=False)
assert ck["epoch"] == <expected>          # refuse to patch the wrong checkpoint
ck["best_agg_metric"] = None; ck["best_epoch"] = None
torch.save(ck, p)
```

Patch a *copy* on ext4 and resume from that, not the original on the NTFS mount.
Confirm it took by grepping the log for
`Resume checkpoint from ... (best_agg_metric=None, best_epoch=None)`.
`selection_mode` needs no change: it is inferred `max` for any metric whose name
does not contain the substring `loss` (`runner_base.py:501`).

## Data handling — non-negotiable

MIMIC-CXR is PhysioNet credentialed data under a DUA that forbids redistribution.
This remote is public.

- Never commit images, report text, processed split CSVs, feature caches,
  prediction JSONL, credentials, or model weights. `.gitignore` covers these
  broadly (`data/`, `Report/`, `*.npz`, `*.jsonl`, `checkpoints/**`, `*.pth`, …).
- **Never publish MIMIC-CXR or any derivative as a Kaggle Dataset, an open-data
  release, or under a licence such as CC0.** The PhysioNet DUA prohibits
  redistribution, and that covers cleaned reports, split CSVs, feature caches,
  predictions and checkpoints trained on them. This rule used to live in
  `configs/kaggle_datasets.yaml`; that file is gone, so it lives here now.
- **Executed notebooks are the easy leak** — their outputs embed `subject_id`,
  `study_id` and report text. `scripts/check_notebook_privacy.py` runs as a
  pre-commit hook; do not bypass it. `notebooks/` no longer exists, but the hook
  still guards any notebook added later.
- `configs/env_config.yaml` is git-ignored here (unlike the old checkout). Edit
  `configs/env_config.yaml.example` for anything shared.
- `image_path` in the processed CSVs is **relative** (`files/p1X/pXXXXXXXX/sYYYYYYY/<dicom>.jpg`)
  and is joined onto `mimic_cxr_jpg_root`, which must point at a directory
  directly containing `files/`. Do not rewrite these to absolute paths.

## Conventions

- `tests/conftest.py` registers `model` and `model.lavis` as **path-only**
  packages so submodules resolve without executing `model/lavis/__init__.py`,
  which would drag in the whole GPU stack. Without it the suite cannot be
  collected on a CPU box. It also stubs `timm.models.hub` when timm is absent.
  When a CPU test needs a GPU-only import, stub it here — do not pip-install into
  the CPU venv.
- `model/lavis/` is a modified fork of Salesforce LAVIS and is excluded from ruff:
  reformatting it would make every future upstream diff unreadable. Same for
  `mhcac/mhcac_8..11.py` (legacy variants; only `mhcac_12.py` is wired).
- `preporcessing/` is misspelled in the tree. Leave it.
- Modules under `training/` carry a dual import shim
  (`try: from stage2_utils ... except ImportError: from training.stage2_utils ...`)
  so they work both as scripts and via `python -m`. Match it in new files there.
- `inference.py` is the legacy Vicuna-7B + LoRA Gradio path and has **not** been
  migrated to MedGemma. Two functionally identical copies of BioViL-T exist
  (`biovil_t/`, `vision_encoders/biovil_t/`).
- The many `docs/*audit*.md` / `*_baseline.md` files are point-in-time records of
  past integration work, not living specs. `docs/STAGE2_PIPELINE_MODES.md` is the
  one to actually follow.

## Keep CLAUDE.md and README.md current — this is part of the work

Whenever you fix something or add a feature, ask whether it changes anything
either of these two files claims, and update them **in the same commit** if so:

- **`CLAUDE.md`** (this file) — what an agent needs to not get it wrong: the
  training host and its quirks, commands that actually work, invariants, traps,
  which components are live vs dead. If you delete a path, remove the flag that
  used to reach it, change a default, or discover that a documented fact is
  false, fix it here. A stale line here misleads every future session.
- **`README.md`** — what a human needs: pipeline status, install, how to run
  each stage, what is and is not validated. Written in Vietnamese; keep it that
  way.

Rule of thumb: if a reader following these files would now do the wrong thing,
the change is not finished. This is not optional cleanup and not a separate
follow-up task — it ships with the code change, alongside the `struct/` update
described below.

Two habits that keep them honest:

- State evidence, not intent. "Verified over SSH on <date>" and "not yet run on
  GPU" are both useful; "should work" is not.
- When you remove something, say it is removed rather than deleting the line
  silently, if a future agent might otherwise try to reintroduce it.

## Source Documentation Synchronization

`struct/` is the persistent source-code knowledge base for this repository and is
tracked in Git. Behavioral source changes and their affected `struct/` pages must
be committed together.

**Before modifying source code:**

1. Read `struct/HOME.md`.
2. Read the documentation of the components you are about to touch —
   `struct/project/<dir>/_index.md`, then `<file>.py.doc.md`, then
   `<file>.py.methods/<fn>.md`.
3. Read `struct/project/_meta/DECISIONS.md` when architectural decisions are
   relevant. It records which components are active, legacy, conditional, or
   still unclassified, and why. Do not re-derive those conclusions.
4. Check `struct/project/_meta/ACTIVE_COMPONENTS.md` and `LEGACY_AND_OPTIONAL.md`
   before assuming a file is dead. "No static import" does not mean "unused" —
   this repo uses CLI entrypoints, registries, YAML config, shell scripts and
   config-gated branches.

**After modifying source code:**

1. Update the corresponding directory documentation (`_index.md`).
2. Update the file documentation (`<file>.doc.md`).
3. Update affected method documentation (`<file>.methods/`).
4. Update caller/callee relationships, in both directions.
5. Update config documentation if behavior changed.
6. Update `ARCHITECTURE.md` / `DATA_FLOW.md` / `CALL_GRAPH.md` if necessary.
7. Add documentation for new files and functions.
8. Remove or re-label documentation for deleted or renamed components.
9. Update the source tree in `struct/HOME.md` when repository structure changes.
10. Verify all relative Markdown links still resolve.

Only update what actually changed. Do not rewrite all of `struct/` on every edit.

A code change that changes behavior is not complete until the relevant `struct/`
documentation is synchronized.

**Source code remains the final source of truth.** If `struct/` conflicts with the
code, inspect the code and update `struct/` — never the other way round. Known
doc-vs-code conflicts are already recorded in
`struct/project/_meta/LEGACY_AND_OPTIONAL.md` under "Potential issues".

**Never write patient data into `struct/`** — no `subject_id`, `study_id`,
`dicom_id`, real image paths, or report text, in any documentation page.
