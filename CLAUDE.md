# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this checkout is

`Meta-CXR-source/` is the **current** source repo (remote `git@github.com:minhphuong150505/Meta-CXR.git`,
branch `main`). It is a sibling of the older `../META-CXR/` checkout. The parent
`../CLAUDE.md` describes that older layout — where it disagrees with this file
(no `stage2/`, `safety/`, `runtime/`, `scripts/`, `training/evaluation/`,
`medgemma_inference/`; Stage-2 test counts), **this file wins for work done
inside this directory.**

## Who does what — one Claude plans, another Claude executes

Set by the user on 2026-08-19. **Codex is gone** — the subscription lapsed on
2026-08-19 and the executor is now Claude Code, which is installed and
authenticated on the training host (`~/.local/bin/claude`, 2.1.233). The split of
*roles* is unchanged; only the executor's identity is.

| | Planner — this checkout, no GPU | Executor — on the training host |
|---|---|---|
| Owns | the plan, the source edits, `CLAUDE.md` / `README.md` / `struct/` | running things |
| Does | read code, design the change, write/apply the diff, decide what "done" means | `git pull`, venv/pytest, smoke runs, Stage-1/Stage-2 launches, `supervise_stage1.sh`, log triage |
| Does not | drive long GPU runs or babysit SSH sessions | redesign the recipe, change losses/configs, or rewrite docs on its own initiative |

Two ways to run the executor, both fine:

- **Directly over SSH from the planning session.** Simplest for a few commands;
  launch anything long with `setsid nohup ... &` so it survives the connection.
- **Headless Claude Code on the host**, which is what to use for a multi-step
  plan. Verified working 2026-08-19:

  ```bash
  ssh phuong@100.116.167.90 'bash -lc "cd ~/Meta-CXR && \
      claude -p \"<the instruction>\" --allowedTools \"Bash Read Write Edit Grep Glob\" \
      --model opus"' < /dev/null
  ```

  Prefer `--allowedTools` over `--dangerously-skip-permissions`: it is enough for
  everything a plan needs, and the safety classifier will refuse to launch the
  dangerous form on your behalf. Put the instruction in a file on the host and
  pipe it in when it is long. **Launch exactly once** — see the warning below.

⚠ **Never launch two executors against this GPU.** On 2026-08-19 the launch
command was issued twice 50 seconds apart: the first ran the plan correctly, the
second OOMed against it and, sharing the same output path, left *its* abort
report behind as the file everyone read. One card, one run, one output path.

The handoff artifact is a file, not a chat message, so a later session can pick
it up cold: the planner writes `docs/handoff/PLAN-<YYYY-MM-DD>-<topic>.md` (exact
commands, expected output, abort conditions), the executor appends an
`## Execution report` section to the *same* file. See `docs/handoff/README.md`.

**Error logs come back summarized, never pasted whole.** A Stage-1 log is
hundreds of MB of MetricLogger lines and pasting it burns the budget that is
supposed to pay for the fix. What comes back:

1. the exact command and its exit status;
2. the first error, with ~20 lines of context and the final traceback frame —
   not the whole traceback chain;
3. the numbers that decide anything: `s/it`, `max mem`, the loss terms by name,
   `epoch`/`iter` at failure, `nvidia-smi` VRAM if it is an OOM;
4. where the raw log still lives on the host (`~/<run>.log`), so the planner can
   ask for one specific `grep` instead of the file.

If the summary is not enough, ask for a named `grep`/`sed -n` range. That is the
only way raw log text should travel.

**When Opus is out of usage or otherwise unavailable, escalate to a Sonnet 5
agent rather than stalling** — Sonnet executes an already-written plan and
triages logs at a fraction of the cost. Keep Opus for what actually needs it:
architecture, loss/recipe decisions, and reading an unexplained result.

⚠ **An executing agent's report is evidence, not truth — check the artifacts.**
Both agents that ran on 2026-08-19 reported honestly, and the record was still
misleading, because two of them wrote to one path. Verify the numbers against the
files on disk (mtimes included) before acting on them, and re-run a measurement
that landed suspiciously close to another process finishing.

Two things this arrangement does **not** relax:

- Whoever makes the behavioral change, the `CLAUDE.md` / `README.md` / `struct/`
  update ships in the same commit (see the two sections at the end of this file).
- The executor is bound by "Data handling — non-negotiable" exactly as the
  planner is. No report text, `subject_id`/`study_id`/`dicom_id`, split CSVs,
  `.npz`, `.jsonl` or checkpoints in a commit, in a handoff file, or in a summary
  sent back.

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

✅ **RESOLVED 2026-08-19 BY DISABLING XMP — the diagnosis below was correct and
the fix was free.** With XMP off the first full Stage-1 run since the reinstall
completed: `run_20260819_xmpoff`, 10/10 epochs, `rc=0`, 18:15:29 -> 06:50:54
(**12h35m**), **0.3505 s/it**, `max mem` 8,526 MiB, and **zero kernel faults** in
`journalctl -b -k`. Cost of running at JEDEC instead of XMP, measured on a 500-
iteration smoke: **+1.3%** (0.2988 -> 0.3028 s/it). Best epoch 6; `checkpoint_best`
selected on `val_loss` and `f1_positive_macro` agree on it.

**Do not re-enable XMP on this host.** Keep the four-DIMM analysis below: it is
what identified the cause, and it is the thing to re-read if faults return.

⚠⚠⚠ **HISTORICAL — THE MACHINE HAD A MEMORY-CORRUPTION PROBLEM (2026-08-19).** Five kernel faults have now been collected across
two boots, in five **unrelated** core subsystems:

```
filemap_get_read_batch          page cache          Comm: mount.ntfs-3g
__rmqueue_pcplist               page allocator      (LIST_POISON2)
__list_del_entry_valid_or_report  the kernel's own list-corruption detector
do_exit                         process exit path
nvme_setup_rw [nvme_core]       NVMe driver         (from ntfs3 read)
```

A driver bug faults in its own code, in the same place, repeatedly. This does
not. Non-canonical addresses, `LIST_POISON2` (`0xdead000000000122`) and the list
checker firing are bit-flip signatures. **Switching the NTFS driver moved the
fault and did not stop it**: `ntfs3` faulted in `ntfs_file_read_iter` -> nvme;
`ntfs-3g` then faulted in `filemap_get_read_batch` and its daemon died, which
training saw as `OSError: [Errno 107] Transport endpoint is not connected`. Two
entirely different code paths, same outcome.

What is ruled out: **not the SSD** (zero NVMe controller errors — no timeout,
reset, abort or `blk_update_request` I/O error); **not the GPU** (no Xid, no
NVRM error, no MCE); **not OOM** (23-29 GB of 30 free throughout); **not the
recipe** (Python cannot fault the kernel).

⚠⚠ **THE MEMORY CONFIGURATION IS FOUR DIMMs FROM TWO DIFFERENT BRANDS**, stated
by the user 2026-08-19. On a consumer DDR4 platform this is the highest-risk
arrangement there is for exactly these symptoms, for three compounding reasons:
populating all four slots roughly doubles the electrical load per channel and
Intel derates the supported speed accordingly (often to 2133-2400 for four
dual-rank modules on Alder Lake); one kit's XMP profile is applied to sticks that
were never validated at those timings or voltage; and mixed ICs and rank
organisations need different training parameters that the controller can only
satisfy for one of them. The result is bit flips under heavy memory traffic —
which is precisely when training does its work — with nothing logged, because
the RAM is non-ECC.

It also reconciles the timeline that otherwise argued against hardware: a
10-epoch run completed on 2026-08-14 (`run_b16fast_20260814/checkpoint_9`), and
nothing has completed one epoch since the 2026-08-17 reinstall. A marginal memory
configuration does not need to change to start failing — a new kernel with
different allocation and page-cache behaviour is enough to tip it over.

**The exact configuration, from `dmidecode -t memory` on 2026-08-19:** 4 x 8 GB,
all single-rank, `Configured Memory Speed: 3200 MT/s` on every stick, at a
reported `Configured Voltage: 1.2 V`.

| module | SPD `Speed` (JEDEC) | configured |
|---|---|---|
| A-DATA `AX4U32008G16A-BW50` x2 | 2667 MT/s | **3200** |
| Kingston `KF3200C16D4/8GX` x2 | **2400 MT/s** | **3200** |

So XMP is on and the Kingston pair is running **33% above** the JEDEC speed its
own SPD declares. Worse, four single-rank DIMMs means **2 ranks per channel**,
and Intel's validated ceiling for 12th-gen DDR4 at 2 DIMMs per channel is
**2666 MT/s** — which is exactly the 2667 the A-DATA SPD reports. The board is
running 3200 in a configuration Intel guarantees only to 2666.

The reported 1.2 V matters too: both kits are XMP-3200 CL16 parts rated at
**1.35 V**, so 3200 at 1.2 V is undervolted, which produces intermittent flips
under load and perfect stability at idle. ⚠ `dmidecode` frequently reports the
SPD nominal 1.2 V even when the BIOS has applied 1.35 V, so confirm in the BIOS
rather than trusting that field.

**Fix order: (1) disable XMP/EXPO in BIOS and run at 2666 or 2400 — free, two minutes,
highest value. (2) If it still faults, run two matched sticks only, from one kit,
in the slots the ASRock manual names for dual channel. (3) Command Rate 2T and
the higher of the two kits' DRAM voltages are the usual stabilisers for a
four-DIMM mixed set.** ⚠ A clean memtest86+ does NOT exonerate this
configuration: marginal four-DIMM setups routinely pass memtest and still corrupt
memory under real multi-threaded load. Treat memtest failure as proof and memtest
success as weak evidence.

⚠ **The RAM is non-ECC** — `EDAC ie31200: No ECC support`. A clean MCE/EDAC log
is therefore *not* evidence of healthy memory; the hardware cannot report it.

Why it always looks like a storage problem: training streams hundreds of GB
through the page cache, so nearly all kernel memory traffic is in the I/O path,
and a random flip lands there. That is why every fault has an I/O flavour without
being an I/O bug.

**What to do, in order.** (1) `sudo apt install memtest86+`, reboot, run at least
one full pass on 32 GB — hours, ideally overnight. That settles it. (2) If
memtest is clean, disable XMP/EXPO in the BIOS (ASRock B760M Pro RS/D4, i7-12700F,
DDR4) and retest; an unstable memory profile produces exactly this. (3) Only then
suspect kernel 7.0.0-29, which is new — but a kernel bug rarely scatters across
five subsystems like this.

Until then, treat any checkpoint this machine writes as suspect: a flip during a
3 GB checkpoint write is silent.

⚠⚠ **Historical, kept because the diagnosis was refined in stages: the storage
stack faults under training I/O.**
Stage-1 died three times in 24 minutes and `journalctl -k` shows **two kernel
Oopses**, both in a `pt_data_worker` — a DataLoader worker reading the dataset:

```
RIP: nvme_setup_rw+0x9c/0x2c0 [nvme_core]     <- faulting instruction
     nvme_setup_cmd+0xad/0x230 [nvme_core]
     nvme_queue_rqs+0xff/0x220 [nvme]
     ...
     ntfs_file_read_iter+0x182/0x260 [ntfs3]  <- WHERE IT CAME FROM
RIP: __rmqueue_pcplist+0x54/0x2e0             <- page allocator, LIST_POISON2
```

**Read the call trace, not just the RIP: the request originates in `ntfs3`**, on a
read of a file from the dataset volume, and is malformed by the time `nvme_core`
tries to set it up. There is **not one NVMe controller error anywhere in the log**
— no timeout, no reset, no abort, no `blk_update_request: I/O error` — which is
what a failing SSD would produce. So this is the kernel NTFS driver on a volume
already recorded as dirty, not dying hardware, and `nvme-cli`/SMART is not the
thing to chase (it is not installed on the host either).

The second Oops is fallout from the first. This is **not** an OOM (23 of 30 GB free,
`/dev/shm` at 131 MB of 16 GB) and **not** something a batch size or a recipe
change can fix. A process that hits it holds its VRAM and ignores `kill -9`;
only a reboot clears it.

**It is the best explanation on record for the two "unexplained" hard hangs**
(2026-08-18 22:25 and 2026-08-19 01:20), which left nothing at all in
`journalctl -b -1` — a block-layer Oops can take the box down before anything is
written. Do not attribute those to power loss any more.

**The fix is to stop reading training data through `ntfs3`, not to tune the
recipe.** A reboot is required first regardless: the dead process's CUDA context
leaks 8,832 MiB that `kill -9` cannot reclaim, and the kernel is `Tainted: G D W O`.
After that, three options, in the order they are worth trying:

1. **Copy the dataset onto ext4, pre-resized.** The pipeline does
   `Resize(512)` (shorter side, aspect preserved) then `CenterCrop(448)`, so a
   copy already at shorter-side 512 makes the resize a no-op. At ~377k images
   that is roughly 25 GB as JPEG q95 or ~113 GB as PNG, against 262 GB free on
   `/home` — where the 571 GB original could never fit. This removes `ntfs3` from
   the training path entirely **and** cuts the dataloader's dominant cost, since
   JPEG decode of the full 7 MP original is 66% of item time. ⚠ JPEG re-encode is
   a lossy, systematic pixel change; PNG avoids it at 4.5x the size. Do the copy
   itself over **ntfs-3g**, not `ntfs3` — reading 377k files through the driver
   that just Oopsed is how you crash the machine again.
2. **`chkdsk` from Windows**, then keep using `ntfs3`. Repairs the volume the
   driver is choking on. `ntfsfix` clears the dirty bit without repairing and is
   not a substitute.
3. **Mount with `ntfs-3g` (FUSE) and train from it.** No kernel NTFS code in the
   path, so no Oops — but it was measured to degrade training badly on this host
   (0.23 -> 1.0-1.4 s/it with 10-14 minute stalls). A fallback, not a plan.

Only one kernel is installed (`7.0.0-29-generic`), so there is no older kernel to
A/B against yet.
Full record: `docs/handoff/PLAN-2026-08-19-stage1-run.md`.

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
#   5 failed (re-measured 2026-08-19), and 2 modules excluded before collection
#   — test_blip2_negative_sampling.py and test_encoder_ablation.py, both of which
#   import model.lavis and therefore torchvision. Collection errors abort the run,
#   so ignore them explicitly to see the real result:
CUDA_VISIBLE_DEVICES="" python -m pytest tests/ -q \
    --ignore=tests/test_blip2_negative_sampling.py --ignore=tests/test_encoder_ablation.py
# The 5 failures are baseline, neither of them a real defect:
#   test_native_independence   4 -- missing private env config
#   test_stage1_eval_hook      1 -- missing torchvision
# test_loss_weight_gating went green again on 2026-08-19 when lambda_itc/itm/lm
# returned to 0.0; it had failed only while the Q-Former was briefly re-enabled.
# Re-measured 2026-08-30 with tests/explainability/ added: 740 tests,
# 732 passed, 5 failed, 3 skipped. Still the same five.
#
# ON THE TRAINING HOST the baseline is different and better, because the full
# stack is installed there: no --ignore is needed and test_stage1_eval_hook
# passes. Measured 2026-08-30 in ~/.venvs/meta-cxr-stage1-311 with
# CUDA_VISIBLE_DEVICES="": 759 tests, 753 passed, 4 failed, 2 skipped. The four
# are all test_native_independence, still the missing private
# configs/env_config.yaml. Use this number, not the CPU-box one, when judging
# whether a change broke something on the host.
# ⚠ It was SIX for a while: test_encoder_finetune.py::TestShippedConfig::
#   test_encoder_finetune_is_on_with_patterns asserted `len(patterns) == 5`,
#   the SHALLOW unfreeze of run_20260820_ft, and 814b778 deepened the config to
#   8 without updating it. The config was right and the test was stale. Fixed
#   2026-08-30 by pinning the pattern SET instead of its count -- a count tells
#   you only that the number moved, and is what let this rot unnoticed.
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
# To get an F1 that MEANS anything, add --label-framing study_presence to BOTH
# lines (see "Which question the metrics answer" below). Mismatched framings are
# refused rather than silently scored, unlike --uncertain-policy.
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
  **The two encoders share ONE coordinate frame, verified on the host
  2026-08-30**, so a saliency map from either can be overlaid on the same image
  without an offset. `preprocessor_config.json` for
  `flaviagiammarino/pubmed-clip-vit-base-patch32` is `do_resize: true,
  size: 224, do_center_crop: true, crop_size: 224, resample: 3`, and the
  dataset hands it a SQUARE 448x448 tensor — so the resize goes to 224x224 and
  the centre crop is a no-op. A pure downscale of the same crop, no second
  crop. `14*32 == 7*64 == 448`, checked at import by
  `training/explainability/projection.assert_shared_coordinate_frame`. Change
  the crop size and that stops holding.
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
- **The vision-language block is OFF, and this time the decision is settled by
  measurement (2026-08-19).** `lambda_itc/itm/lm` are all `0.0`;
  `lambda_teacher_cls/distill` stay at `0.5`. `forward()` therefore skips the
  Q-Former's **image** path — query tokens and cross-attention keep their BLIP-2
  initialisation — while the Q-Former's **text** tower still runs and still
  trains, because MHCAC's teacher reads it. The skip logic is unchanged and
  covered by `tests/test_loss_weight_gating.py`.

  **This matches the reference implementation exactly.** In
  `DasithEdirisinghe/META-CXR` the whole vision-language block of
  `blip2_qformer.py` is commented out; Stage-1's loss is
  `cls + 0.3*contrastive + 0.7*orth + 0.3*sparsity` (weights this repo already
  matches), and MHCAC there is called with `text_embeddings` from
  `Qformer.bert(...)` and with raw `image_embeds` — never with a query-token
  output. So the published Stage 1 trains the text tower and not the image path,
  which is what this configuration now does.

  ⚠ **The cost, stated plainly: this checkpoint is NOT valid for the Stage-2
  `meta_cxr_qformer` soft-token modes**, because the cross-attention that would
  produce those soft tokens never sees a medical image during Stage 1. The
  reference is in the same position. `medgemma_direct` is unaffected.

  The history, so nobody re-enables it a fourth time: switched off 2026-08-13 for
  cost; on again, off again 2026-08-16 at exactly chance; on again 2026-08-18;
  off 2026-08-19 after `scripts/check_itc_gate.py` and a controlled
  pinned-temperature probe both returned chance. The gate table is below.

  ⚠ **It forced batch 16 -> 8, and it cost Swin — batch is back to 16 x 4 as of
  2026-08-19, Swin stays off.** Swin was not restored with the batch: turning it
  off also restores `_native_stream_layouts`, where PubMedCLIP keeps its CLS
  token and each encoder keeps its native scale, which this repo had already
  measured to be the better configuration. The VRAM table below is the evidence
  for the batch, and it was measured with the VL block ON; with it off this
  configuration is strictly smaller. Measured on the 5060 Ti,
  2026-08-18, all with `lambda_itc/itm/lm: 0.1`:

  | encoders | batch | encoder FT | peak VRAM | s/it | outcome |
  |---|---|---|---|---|---|
  | 3 (swin on) | 8 | off | 11,238 then OOM | — | OOM in the **backward** pass, iteration 1 |
  | 3 (swin on) | 4 | off | 11,083 | 0.40 | fits; ITM negatives fall to 3 |
  | 2 (swin off) | 8 | off | **14,460 / 15,850 (93.4%)** | 0.42 | measured 2026-08-18 |
  | 2 (swin off) | 8 | **ON (deep)** | **OOM** | — | ⚠ measured 2026-08-31 |
  | 2 (swin off) | 8 | off | **14,763** | **0.43** | ⚠ re-measured 2026-08-31, 750 iters |

  ⚠⚠ **The 2026-08-18 rows predate encoder fine-tuning and no longer describe
  the shipped config.** `model.encoder_finetune` releases 53.12M encoder
  parameters, and with the Q-Former also on, batch 8 now **OOMs** — iteration 0
  completes at 9,885 MiB and iteration 1 dies, which is the retained-gradient
  signature, not an oversized activation. The two features cannot both be on at
  batch 8 on this card.

  Which one to drop is a modelling decision, not a memory one: ITC and ITM draw
  their negatives from the live microbatch, so halving the batch to keep the
  unfreeze leaves ITM three candidates and makes the Q-Former probe answer
  nothing. **Drop the unfreeze, keep batch 8** — that restores the exact
  configuration the 14,460 measurement was taken on, and it re-measures at
  14,763 MiB / 0.43 s/it.

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
  | ~525 updates, 2026-08-31, learned | 0.00944 | **127.49** | **127.76** | **-0.0118** |

  ⚠⚠ **A FOURTH measurement, 2026-08-31, and it is chance again.** Run
  `run_20260831_qformer` was launched specifically to retest this with
  `lambda_itc/itm/lm: 0.1`, batch 8, encoder fine-tuning off — the exact
  configuration that produced the only encouraging training curve on record.
  At iteration 4,200 (~525 optimizer updates) the val gate returned
  `loss_itc` 5.5570 against a chance of ln(256) = 5.5452, ranks 127.49 / 127.76
  against a chance rank of 127.5, `delta_nats` **-0.0118**. The JSON carries
  `studies_scanned` 392 and `valid_fraction` 0.6531, so it is the fixed
  measurement, not the retracted one.

  The training curve did exactly what it did last time and must not be read as
  evidence: `loss_itc` sat ~1.4 nats below chance around iteration 1,650, then
  returned to 5.5625 — chance — by iteration 7,600. The temperature collapsed
  again (0.00944), which this file already records as a symptom rather than a
  cause.

  **Four independent measurements, four times chance. Stop proposing this.**

  **THE CAUSE IS NOW IDENTIFIED FROM THE LITERATURE, AND IT IS BATCH SIZE.**
  Checked 2026-08-31 against the papers rather than guessed:

  | | BLIP-2 stage 1 | this repo |
  |---|---|---|
  | contrastive batch | **1,680-2,320** | **8** |
  | negatives | in-batch, **no queue** | 256-entry queue off the LIVE encoder |
  | pretraining data | 129M images | ~222k studies |
  | hardware | 16x A100 40G, <6 days | 1x RTX 5060 Ti 16G |

  That is a **210x smaller batch** and ~580x less data. Softmax contrastive
  gives a positive pair the gradient "be more similar than the hardest of N-1
  negatives"; at N=8 the hardest negative is usually easy, so the gradient is
  weak and noisy. Chance is the predicted outcome, not a mystery.

  Note also that BLIP-2 **removed** the momentum queue precisely because a
  frozen image encoder lets you fit big batches. This repo kept a queue *and*
  has a tiny batch — the worst of both, and the queue is filled from the live
  encoder, so its keys are stale.

  ⚠ **BLIP-2's own ablation says stage-1 representation learning is NOT
  optional if you want the soft-token bridge**: "Without representation
  learning, the Q-Former fails to bridge the modality gap", and OPT-style
  decoders then show "catastrophic forgetting where performance drastically
  degrades as training proceeds". So `meta_cxr_qformer` cannot be rescued by
  skipping stage 1 — it either gets a real contrastive batch or it does not
  work.

  **Three ways out, in the order they are worth trying.** None has been tried.

  1. **GradCache + the frozen-feature cache.** GradCache decouples the
     contrastive backward from the encoder along the batch dimension, giving
     near-constant memory for an arbitrarily large effective batch. This repo
     is unusually well placed for it: the encoders are frozen and
     `run.feature_cache_dir` already exists, so the Q-Former can train alone on
     cached features and an effective batch of ~1,000+ becomes reachable on one
     16 GB card. This attacks the measured cause directly.
  2. **Replace softmax ITC with SigLIP's pairwise sigmoid loss.** It removes
     the global normalisation, so each pair is scored independently and small
     batches degrade far less; it is also cheaper in memory. Its sweet spot is
     still ~32k, so treat it as a multiplier on (1) rather than a substitute.
  3. **Close the route.** Keep `lambda_itc/itm/lm: 0.0`, match the reference
     implementation, and commit to `medgemma_direct`. Cheapest and honest —
     but it retires `meta_cxr_qformer` permanently.

  **Cost, measured and then corrected — MetricLogger is RIGHT.** An epoch at
  batch 8 is **3h26m at 0.4445 s/it** (27,844 iters), so a 10-epoch run is
  **~34 h**. Checkpoint writes are negligible: 3.38 GB in 1.4 s, 2.37 GB/s.

  ⚠ A "0.82 s/it actual vs 0.43 reported" claim briefly stood here and was
  WRONG. It came from a monitoring grep hardcoded to `epoch: \[0\]`, which
  kept returning a stale epoch-0 line after the run moved to epoch 1; one
  epoch's iterations were then divided by two epochs' wall clock. The run's own
  timestamps settle it — `Start training epoch 0` 01:00:37, epoch summary
  04:26:54 (`Total time: 3:26:16, 0.4445 s / it`), `Start training epoch 1`
  04:26:57, killed 07:21:27 at epoch 1 iteration 23,750, which is the same
  0.44 s/it. **When monitoring a multi-epoch run, never pin the epoch index in
  the progress grep.**

  Aborted during epoch 1 on the gate result.

  Sources: BLIP-2 (arXiv 2301.12597), SigLIP (arXiv 2303.15343), GradCache
  (arXiv 2101.06983).

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

`EXTRA_OPTS` appends one-off `--options KEY=VALUE` pairs, last, so they beat
what the supervisor sets itself. It exists because varying a single knob
otherwise meant editing the tracked YAML, which then misdescribes every other
run made from it. Extending a finished run is the motivating case:

```bash
OUT=$HOME/<run> LOG=$HOME/<run>.log WID=<id> EXTRA_OPTS="run.max_epoch=15" \
    setsid nohup bash ~/supervise.sh >>$HOME/<run>.supervise.log 2>&1 &
```

⚠ **`LinearWarmupCosineLRScheduler` is stateless** — it computes the LR from
`cur_epoch`, `max_epoch` and `steps_per_epoch` with no counter to restore. So
raising `max_epoch` on a resume reshapes the whole curve, including the part
already trained, and the join is a **warm restart, not a continuation**.
Measured for the 10 -> 15 case at 3,480 updates/epoch: the finished run ended at
`min_lr` 2.00e-5 and the resume re-enters at **4.06e-5, a 2.03x jump**, then
decays back. That is a legitimate SGDR-style restart and it is *not* the same
as a clean 15-epoch cosine (which would have run epoch 9 at 4.84e-5 rather than
2.21e-5). Describe it by its real name.

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

#### Read-only run monitoring — `scripts/train_healthcheck.sh`

This is the compact check used by cron and manual SSH triage; it never restarts
or kills a run. Set `RUN_DIR` and `LOG` to the active artifact directory and raw
log. Set **`EXPECT_RUNNING=1` for a scheduled experiment**: if the Python process
vanishes (for example after the Stage-2 `NVRM Xid 8` observed 2026-08-31), the
result is ALERT/exit 3 instead of ordinary IDLE/exit 4.

Progress is the newest artifact **write**, across both formats: Stage 1
`*.pth`, and Stage 2 `adapter_model.safetensors` / `trainer_state.pt` under
`checkpoints/last`. Do not narrow this back to `*.pth`; that made a healthy
Stage-2 recovery save invisible. `find -printf %T@` includes fractional seconds,
so the script strips the fraction before shell arithmetic — the former direct
subtraction raised `arithmetic syntax error` on real Stage-1 checkpoints.

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

### Full-pipeline inference on the TEST split — measured 2026-08-31

The first end-to-end run of both stages on test, from
`run_20260820_ft/checkpoint_best`. ⚠ **No Q-Former and no Stage-2 fine-tune are
involved**, and neither is available: `lambda_itc/itm/lm` are `0.0` so the
Q-Former's image path never trained, and no Stage-2 adapter exists on the host.

**Stage 1, classification, n=3,269.** Thresholds calibrated on val
(`--selection plateau --plateau-fraction 0.95 --min-positive 5`), scored on test
under `study_presence` + `marginal_presence`. **Reproduces the recorded numbers
exactly**, which is the useful part — the pipeline is intact:

| macro_auroc | micro_auroc | macro_auprc | pos_macro_f1 | precision | recall | specificity |
|---:|---:|---:|---:|---:|---:|---:|
| 0.7643 | 0.8166 | 0.3203 | 0.3542 | 0.2931 | 0.5373 | 0.8020 |

**Stage 2, generation, n=300 test studies, ZERO-SHOT.**
`google/medgemma-1.5-4b-it` out of the box via
`scripts/generate_stage2_reports.py`; 0 failures, 26.7 min, peak 10.65 GiB,
generated median 66 words against a reference median of 57.

| BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | ROUGE-L | METEOR | CIDEr | BERTScore-F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.3066 | 0.1752 | 0.1091 | 0.0704 | 0.2373 | 0.0556 | — | 0.8112 |

⚠ CIDEr is **0.0556**, i.e. near zero, while BERTScore is 0.81. That gap is the
expected signature of a zero-shot model producing fluent, clinically-plausible
prose that does not reuse the corpus' n-grams; it is not evidence of a working
report generator. **These are baseline numbers for an un-finetuned model and
must never be quoted as this project's Stage-2 result.** Clinical metrics
(CheXbert/RadGraph/RadCliQ) are reported as unavailable, not as zero.

**Stage-2 explanation, n=200 test studies.** Both gates passed before any map
was written: ablation (mismatched image, n=100) **+0.1523 [+0.1226, +0.1825]**,
established; randomization rho **0.923 -> 0.871 -> 0.704 -> 0.410 -> 0.359 ->
0.144** across 1..34 layers, degrades. `parse_coverage` **0.683** over 1,155
sentences with `lexicon_v2`. Eight example overlays rendered by
`scripts/render_explanation_examples.py`; they are patient images and stay on
the host.

### Stage-2 explainability — `training/explainability/` (branch `feat/stage2-explainability`)

Post-hoc XAI over the language head. **It is an OBSERVER: nothing in Stage 1 or
Stage 2 imports it, and deleting the directory leaves both pipelines byte-for-byte
unchanged.** Keep it that way -- a layer that can alter what it explains is not
an explanation layer. Only `attention_capture.py` (NOT YET WRITTEN, needs the GPU
host) touches a live model; it installs and removes every hook in `finally`.

⚠⚠ **Q-Former soft tokens carry NO geometry, so stage 1 of the pipeline alone
cannot produce a heatmap for the `meta_cxr_qformer*` modes.** The 32 soft tokens
are `query_output.last_hidden_state`: each has already cross-attended over all
246 visual tokens, and those cross-attention weights are **not** in the cached
Stage-2 record (`.sensitive_stage1_cache/` holds only `[32, 768]`). Rolling
attention back to them says which *soft token* a sentence used, never which
*image region*. `projection.assert_spatial_projection_supported` raises
`SpatialProjectionUnsupported` rather than rendering a picture that would look
meaningful and not be. **`medgemma_direct` is the only route here that yields a
real spatial map**, because MedGemma's own vision tower has a patch grid.

⚠ `rollout.py` imports `torch` and nothing else -- no model, no `transformers`,
no cross-import from this repo. That is what makes its arithmetic checkable
against hand-computed matrices on a CPU box, which is the only machine the
planning session has. Do not add an import there for convenience.

⚠ **The rollout formula is Chefer et al. 2021 as this repo reads it, and has NOT
been checked against the project's own design document** (that file was never
reachable in the session that wrote the module). Check it before quoting a
number. If it differs, the change is confined to `fuse_heads` and
`_rollout_chefer`, and the hand-computed tests will catch it.

Two details that cost a test each, both pinned:
`fuse_heads` clamps **before** averaging over heads (mean-then-clamp lets a
negative head cancel a positive one: 2.0 becomes 0.0, same shape, no error), and
a span that receives no attributed mass returns **zeros, not a uniform
distribution** -- "this sentence did not use the image" is a real answer.

Sentence labels come from `LexiconSentenceLabeler` (`lexicon_v1`), a thin adapter
over `safety.claims.LexiconClaimParser`. **This repository implements no trained
clinical labeler and this one must never be presented as one.** Its limit is
measured, not assumed: `parse_coverage` rides on every study record and is
aggregated **by sentence, not by study** -- quote it beside any sentence-level
result. Unparsed sentences are kept, flagged `spatially_meaningful: false`.

Output is `.jsonl` (already git-ignored), one line per study, written through
`scripts/evaluate_explanation.py`'s `_assert_private_output_location`; maps are
`.npz` at native grid resolution (14x14 / 7x7), never upsampled PNG.

**Measured on GPU 2026-08-30, `google/medgemma-1.5-4b-it`, RTX 5060 Ti.**
Loading is bf16 with `attn_implementation={"text_config": "eager",
"vision_config": "sdpa"}`, every parameter frozen, the vision tower run under
`no_grad` and its features scattered into the input embeddings, and gradients
taken with `torch.autograd.grad` rather than `.backward()`. Peak 9.9 GiB of
15.5; attention comes back `[34, 8, S, S]` with the gradient reaching all 34
layers. **NF4 is not needed and should not be used here** — the Mucs-7 worry
about differentiating through it does not arise on this hardware.

Each of those four choices cost an OOM to find, so none is decorative:
`output_attentions=True` propagates into SigLIP (27 layers x 16 heads x 4096^2
fp32 ≈ 27 GiB retained); eager SigLIP retains its softmax for backward even
without it; `.backward()` allocates gradients for ~4B parameters; and the
logits are `[1, S, 262208]`.

**MedGemma's frame is not Stage-1's frame.** `Gemma3ImageProcessor` resizes to
896x896 with NO aspect preservation (measured: a 2544x3056 study, aspect
0.8325, comes out 1.0) and does not crop, while Stage 1 resizes the shorter
side to 512 preserving aspect and then centre-crops 448 — which **discards
36.3%** of a portrait radiograph that MedGemma sees in full. Its grid is 16x16
at 56 px (SigLIP tiles 896/14 = 64x64, the projector pools 4x to
`mm_tokens_per_image` = 256). Carry both maps back to the ORIGINAL image before
comparing them; `projection.Stage1CropGeometry` mirrors torchvision's exact
arithmetic and is pinned against five real outputs.

**Geometry check: 4/4.** With a neutral target naming no location, a synthetic
bright square in each of the four quadrants put the hottest quadrant of the
projected map in that same quadrant every time (chance 1/4). An earlier run
scored 4/4 too but was worthless — its target said "left upper zone" while the
square sat top-left, so a map following the WORDS would have looked identical.
Keep the target neutral.

⚠⚠ **The ablation gate requires a bootstrap CI that excludes zero, not a
threshold on a point estimate — and that is not hypothetical.** On 12 real test
studies (frontal, 30–90 findings tokens, seed 16):

| control | mean ΔNLL | 95% CI | studies worse | established |
|---|---:|---|---:|---|
| zeroed visual embeddings | +0.0535 | [-0.0283, +0.1314] | 7/12 | **no** |
| another study's image | **+0.1868** | **[+0.0761, +0.3045]** | 10/12 | **yes** |

The zero-ablation clears the 0.05 threshold and is still indistinguishable from
no effect. A threshold-only gate called it a pass; `score_ablation` now does
not. **The mismatched-image control is the sharper test and the one to quote**:
a zero vector is not "no information", it is an out-of-distribution point, and
on a synthetic non-radiograph zeroing actually made the target EASIER to
predict (-0.84). Substituting another patient's image asks the question that
matters — is THIS image being used — and stays in distribution.

⚠⚠ **`scripts/explain_stage2.py` explained the BASE model until 2026-09-01 —
it had no `--adapter` flag at all.** Every XAI number recorded before that date
therefore describes `google/medgemma-1.5-4b-it` out of the box, not this
project's Stage 2, and the two full val runs (`~/xai_val`, `~/xai_val_v2`) are
zero-shot baselines. That is still a legitimate result for the two sanity gates,
which test the method rather than the fine-tune; it is not a result about the
fine-tuned model. `--adapter` now merges a Stage-2 QLoRA adapter into the bf16
base (`merge_and_unload`, so the attention modules stay the plain Gemma ones the
hooks and `randomize_layers` expect), raises if `adapter_config.json` is absent,
and `summary.json` records `mode` + `adapter`. ⚠ The adapter was trained against
an NF4 base and is merged into a bf16 one — the standard QLoRA deployment path,
but not bit-identical to what was trained. Say so.

⚠⚠ **`native_anchor_guided` (Stage-2 arm B) IS NOT WIRED, and it fails
SILENTLY.** `build_records` (`training/dataio/manifest.py:229`) emits only
`sample_key`, `ref`, `image_path`, `anchor_view`, `auxiliary_views` — no
`pred_groups`. `context_from_record` (`stage2/prompts/records.py:41`) turns the
absent key into an empty tuple, and nothing in `builder.py`/`validation.py`
raises when a guided mode receives zero cues. The only producer of `pred_groups`
is `build_stage1_records`, whose own docstring forbids native mode from calling
it. So running `--prompt-config configs/experiment_native_anchor_guided.yaml`
today trains on prompts **identical to arm A**, differing only by RNG — a 70-hour
no-op. `tests/test_stage2_prompts.py::test_the_two_experiment_arms_differ_in_one_line`
does not catch this: it compares the two YAML files, not the two prompts.
**Arm B was dropped from the pipeline on 2026-09-01** at the user's decision;
arm A (`native_anchor_only`) is the official Stage-2 run. If it is ever revived,
the agreed cue source is `marginal_presence = sigmoid(m) x q_pos` with thresholds
calibrated on validation — NOT `classify_with_thresholds`
(`train_eval_figure9_llm_variants_200.py:352`), which softmaxes the
classification logits alone and never touches the mention gate.

**The deliverable is `scripts/explain_stage2.py`.** One JSONL line per study
(sentence text, `lexicon_v1` labels, map path, `mean_token_nll`,
`spatially_meaningful`), one NPZ of maps per study at the native 16x16 grid,
and a `summary.json`. Run end-to-end on GPU 2026-08-30: gate established at
+0.1429 [+0.0284, +0.2668] over 8 studies, 6 studies explained, peak 10.7 GiB.

The gate runs FIRST and aborts the run, because a directory of
plausible-looking maps produced by a model that is not using the image is worse
than no output. `--skip-ablation-gate` and `--no-gradient-weight` exist, warn,
and are recorded in the summary; both default to strict.

⚠ **`parse_coverage` was 0.606 pooled over 33 sentences on the first real run**,
with one study at 0.125 — one labelled sentence in eight. Four sentences in ten
carry no label at all, so every sentence-level conclusion is bounded by that
number. The command prints it on exit for exactly this reason. Pooled by
sentence is the figure to quote; the mean of per-study fractions (0.674 on the
same run) flatters it.

Output paths go through `evaluate_explanation.py`'s `_assert_private_output_location`;
map filenames are sequential and identifier-free; the JSONL carries a blake2
`sample_key`, and the join to real ids is written only under `--write-key-map`.

**FULL VAL RUN, n=1513 — these are the numbers to quote.** 2026-08-30,
23.1 min wall, 0 errors, 1,513 of 1,513 frontal val studies with valid
findings, 7,786 sentences, 15 MB of output. Peak 11.31 GiB of 15.48; every
study used the shared graph and the per-sentence fallback never fired.

| | n | result |
|---|---:|---|
| ablation, mismatched image | **100** | **+0.1788 [+0.1400, +0.2185]**, 83% worse, established |
| randomization, final rho | 1 | **-0.0030**, degrades |
| `parse_coverage` (pooled by sentence) | **7,786** | **0.483** |
| `spatially_meaningful` sentences | 7,786 | 3,758 = 48.3% |
| sentences/study | 1,513 | min 1, median 5, max 14 |
| `mean_token_nll` | 7,777 | median 2.451, p5 0.547, p95 8.624 |

⚠ **154 of 1,513 studies came out at zero coverage** — every sentence
unlabelled. Median per-study coverage is 0.500 while the pooled figure is
0.483; quote the pooled one.

⚠ Earlier smoke figures (n=6: coverage 0.606, ablation +0.1868/+0.1429) are
**not results** and must not appear in a report. Every log line and summary key
now carries its n for exactly this reason (`n_studies`,
`n_ablation_studies`, `n_randomization_studies`).

**Second gate: weight randomization (Adebayo et al. 2018), and it passes.**
`cascading_randomization` re-initialises the language layers from the last
backwards and rank-correlates each map with the original. Measured on GPU
2026-08-30, one real val study:

| layers randomized | 1 | 2 | 4 | 8 | 16 | 34 |
|---|---:|---:|---:|---:|---:|---:|
| Spearman vs original (full val run) | 0.981 | 0.963 | 0.886 | 0.560 | 0.510 | **-0.003** |
| Spearman vs original (earlier, n=1) | 0.825 | 0.824 | 0.842 | 0.609 | 0.376 | 0.130 |

All six points are written to `summary.json` under `randomization_gate`
(`steps` + `correlations`), not just the pass/fail, and the runner prints the
whole curve on exit.

⚠ **Report the first point.** rho is **0.981** after randomising the last
layer — re-initialising it changes the map almost not at all. Two readings, and
this repo cannot currently separate them: the last layer genuinely carries
little of the attribution (Chefer's recurrence multiplies over all 34, so one
broken factor is diluted), or the method is insensitive to the top of the stack.
The collapse from layer 8 onward is what makes the gate pass; the flat start is
a property to disclose, not to omit.

That is the shape a valid method should have -- stable while only the last few
layers are broken, then progressive collapse to near zero. A map that stayed
flat here would be a function of the input and the architecture, not an
explanation. The gate aborts the run, like the ablation one, and both are
recorded in `summary.json`. On the same run the ablation was **+0.2376
[+0.1087, +0.3807], 100% of studies worse**.

**MedGemma's grid has its own import-time assert.**
`assert_shared_coordinate_frame(STAGE1_GRIDS)` checks 14*32 == 7*64 == 448 and
says nothing about `medgemma_direct`, which lives in a different square. From
the checkpoint config: SigLIP `image_size` 896 / `patch_size` 14 -> 64x64 =
4096 patches, `mm_tokens_per_image` 256 -> the projector pools **4x** to 16x16,
so one LM image token covers 896/16 = **56 px**. The attributed patch size is
that pooled 56, not SigLIP's 14. A 16x16 grid at 28 px tiles 448 exactly, so
the Stage-1 assert would have passed it while it was wrong by 2x for MedGemma
-- hence a second assert.

**Memory: the shared graph scales.** val tops out at 14 sentences per study.
Through the runner, eight of the longest val reports (7-13 sentences,
121+ findings tokens) peaked at **11.31 GiB of 15.48**, 4.17 GiB spare, with no
fallback triggered. `--graph-mode per-sentence` exists and does NOT reduce the
peak (11.23 vs 11.27 on a matched set) because the peak tracks sequence length,
not sentence count -- it costs +47% wall time and is worth having only as the
`auto` OOM fallback. ⚠ The runner's default 30-90 findings-token filter was
selecting the worst case away; the longest val studies carry 121-138 tokens.

⚠⚠ **`parse_coverage` is 0.483 over the full val split (n=7,786 sentences);
the composition below was measured on a 300-study subsample (1,556 sentences,
coverage 0.493). But most of the miss is not a labeler
failure**, which is what decides whether a better labeler is worth building:

⚠⚠ **`lexicon_v2` WAS FITTED TO VAL, so 0.6477 is in-sample.** The process was
literally: run v1 on val (0.4827) -> bucket the sentences it failed to label ->
read the word frequencies of the `unclassified` bucket ON VAL -> add exactly
those terms (`heart size`, `low lung volumes`, `tube`/`line`/`device`) -> pick
the nine extended findings from the `outside_14` bucket measured ON VAL -> re-run
on val (0.6477). Both numbers are real and both must be reported together with
that description; **quoting 0.6477 alone would be a held-out claim the work does
not support.**

The lexicon is now FROZEN. Both lexicons carry a content hash written into every
JSONL record and both summaries — v1 `f809b80b33579b43` (14 labels), v2
`3fcf80ad1c9b6c78` (23 labels) — and `tests/explainability/
test_sentence_attribution.py::test_the_lexicons_are_frozen` pins them, so any
further edit is a red test rather than a quietly better number. Changing a
lexicon is only legitimate for a reason INDEPENDENT of val results (a
clinician's correction, a wrong term); never because coverage looked low. The
honest estimate for v2 is a single measurement on the **test** split, which has
not been run.

**Both full val runs exist and are directly comparable**, `~/xai_val` (v1) and
`~/xai_val_v2` (v2) on the training host, 1,513 studies each, 15 MB each,
0 errors:

| | `lexicon_v1` | `lexicon_v2` |
|---|---:|---:|
| `parse_coverage` | 0.4827 | **0.6477** |
| labelled sentences of 7,786 | 3,758 | **5,043** |
| `spatially_meaningful` | 48.3% | **64.8%** |
| zero-coverage studies | 154 | **128** |
| peak VRAM | 11.31 GiB | 11.31 GiB |
| ablation, n=100 | +0.1788 | +0.1788 |
| randomization, final rho | -0.0030 | -0.0030 |

The two gates reproduce to the digit across runs, which they must: neither
depends on the labeler. Label instances under v2: **6,384 `chexpert_14`, 611
`extended`**.

**`lexicon_v2` extends the taxonomy, and it works — coverage 0.483 -> 0.648.**
Full val, n=1,513 studies / 7,786 sentences. **v2 is the default on both
commands and in `attribute_sentences` as of 2026-08-30.** Pass
`--labeler lexicon_v1` to reproduce anything recorded before that date — every
artifact writes the labeler it used, so no existing output became ambiguous.

| bucket, share of unparsed | v1 (4,028 unparsed) | v2 (2,743 unparsed) |
|---|---:|---:|
| `normal` | 41.0% | **45.6%** |
| `technical` | 26.9% | **32.6%** |
| `unclassified` | 15.8% | 15.5% |
| `outside_14` | 10.8% | **3.8%** |
| `missed_14` | 5.4% | **2.5%** |

`missed_14` fell 217 -> 68 sentences and `outside_14` 437 -> 105. What is left
unparsed is 78.2% `normal` + `technical`, i.e. sentences with nothing to label.

⚠⚠ **Only 61% of the gain is verifiable, and the split must be reported.**
Of the 1,285 sentences v2 newly labels:

| | sentences | share of gain | coverage |
|---|---:|---:|---:|
| CheXpert-14 only — Stage 1 predicts these | **782** | 60.9% | +0.100 |
| extended only — **nothing predicts these** | **473** | 36.8% | +0.065 |
| both tiers | 30 | 2.3% | — |

So `lexicon_v2` buys +0.100 of verifiable coverage and +0.065 of description.
Every label carries a `tier` into the JSONL for exactly this reason: a sentence
labelled `Aortic Abnormality` has gained coverage, not verification, and must
never be counted as a checked claim.

Extended findings actually hit, full val: `Aortic Abnormality` 124,
`Postsurgical Change` 124, `Scarring` 93, `Degenerative Change` 82,
`Hyperinflation` 80, `Upper Abdomen` 63, `Hernia` 25, `Spinal Deformity` 15,
`Chest Wall Deformity` 5.

⚠ **`safety/claims.py` is deliberately untouched.** Its 14 labels correspond
one-to-one with Stage 1's classification head, and `safety/pipeline.py`
reconciles each claim against that classifier; adding labels there would create
claims with nothing to check them against. The extension lives in
`training/explainability/`, where nothing is being verified.

⚠ The nine extra labels are a **proposal** and need a clinician's sign-off
before any published claim, exactly like the kappa table.

Measured over the **whole val split**: n=1,513 studies, 7,786 sentences,
4,028 of them unparsed. The diagnostic reaches the same cohort and the same
0.4827 coverage as the GPU run by an independent path, which is a useful check
on both.

| bucket | share of the 4,028 unparsed | 300-study subsample |
|---|---:|---:|
| `normal` — normality/negation naming no finding | **41.0%** | 39.5% |
| `technical` — priors, projection, positioning, headers | **26.9%** | 26.0% |
| `unclassified` | **15.8%** | 16.5% |
| `outside_14` — real finding outside the 14-label taxonomy | 10.8% | 12.7% |
| `missed_14` — wording for one of the 14 the lexicon should have caught | **5.4%** | 5.3% |

The subsample was representative; every bucket moved by under 2 points.

Adding header and request phrases moved mass from `unclassified` into
`technical` and left `missed_14` unmoved — 42 sentences before and after on the
subsample. A reclassification that drained the bucket the decision rests on
would have been worse than no change.

So better synonyms *within* the 14 labels address about 5% of the misses.
⚠ **5.4% is the lower bound; 10.4% is the upper, and both are now measured.**
Of the 636 remaining `unclassified` sentences on full val, **202 (31.8%)**
contain a word that would name one of the 14 labels if the lexicon carried the
synonym — the bucket's commonest terms are `lung`(91), `volumes`(75),
`enlarged`(40), `heart`(36), `tube`(30), i.e. low-lung-volume, Cardiomegaly and
Support-Devices phrasings. Counting all of them gives (217+202)/4028 = **10.4%**,
and that is generous, since `aorta`(34)/`aortic`(34)/`tortuous`(23) in the same
bucket are `outside_14`, not missed labels.

**So better synonyms inside the 14 labels address 5–10% of the misses, against
`normal` + `technical` at 67.9%.** In absolute terms that is 217–419 sentences
of 7,786, i.e. 2.8–5.4% of all sentences. That is the number the
"is a better labeler worth building" decision rests on.
`scripts/diagnose_parse_coverage.py` writes the sentences it sorted, grouped,
so the classifier can be checked; that file is report text and stays on the host.

⚠ Every ablation and coverage number in this file states its **n**. The n=6
smoke figures are not results.

207 CPU tests in `tests/explainability/` (namespace package -- no
`__init__.py`), plus the GPU checks above, which are not automated.

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

**Which question the metrics answer — `--label-framing`, added 2026-08-20.**
The paragraph above diagnoses the symptom; this is the cause and the fix.
`training/evaluation/label_framing.py` names the two readings of the label matrix
and records the choice in every output file, exactly as `uncertain_policy` does
for the uncertain class:

All numbers below are `positive_macro_f1` on the `run_20260819_xmpoff` test
split (3,269 studies), thresholds calibrated on validation with the matching
flags:

| | `masked_polarity` (default, historical) | `study_presence` + `q_pos` | `study_presence` + `m x q_pos` |
|---|---|---|---|
| A blank cell means | masked — polarity *given* the finding was mentioned | not present | not present |
| Per-label prevalence | 0.13–1.00, twelve labels above 0.55 | **0.019–0.344** | same |
| `all_positive` | **0.8397** | 0.2280 | 0.2280 |
| `all_negative` / `majority_class` | 0.0000 / 0.8200 | 0.0000 / 0.0000 | 0.0000 / 0.0000 |
| `threshold_half` | 0.8713 | 0.2797 | 0.3173 |
| **Model** | 0.8717 | 0.3074 | **0.3224** |
| Model over `all_positive` | **+0.032** | +0.079 | **+0.094** |
| `macro_auroc` | 0.7857 | 0.6767 | **0.7441** |
| `macro_specificity` | 0.4197 | 0.6359 | **0.8214** |
| Degenerate labels | 3 (`No Finding`, `Pleural Other`, `Fracture`) | **none** | **none** |

`masked_polarity` is the right denominator for the **loss** and a broken one for
**F1**: a constant beats the model by 0.032 there, calibrating thresholds buys
+0.0004 over a flat 0.5, and `majority_class` alone scores 0.8200.
`study_presence` asks the clinical question — *does this study have finding X* —
so blank/negative/uncertain all mean "not present", prevalence returns to normal,
and both trivial constants collapse to 0.0000.
**Quote F1 only under `study_presence`.** `macro_auroc` is framing-dependent too
because the populations differ; always report which framing it came from.

**The mention gate is worth using and the measurement is unambiguous:
`m x q_pos` beats `q_pos` on AUROC for 14 of 14 labels, mean +0.0831.** Biggest
gains are exactly where "was it mentioned" carries the signal — `No Finding`
+0.2603, `Pleural Other` +0.2501, `Fracture` +0.1798, `Support Devices` +0.0948
— and those first three are precisely the labels that are degenerate under
`masked_polarity`. `micro_auroc` goes 0.6384 -> 0.8183, `macro_specificity`
0.6359 -> 0.8214. Per-label table: `Test/stage1_test/mention_gate_contribution.csv`
(git-ignored, on the dev box).

⚠ `m` is **not** a calibrated mention probability — the gate trains with
inverse-frequency weights (`n_not_mentioned / n_mentioned x kappa_gate`, kappa 2,
capped at 10), so its odds are inflated. Harmless when thresholds are calibrated
on validation, which absorbs it; do not read `m = 0.5` as "50% chance it was
mentioned".

⚠ Under `study_presence` the framing has already folded uncertain into "not
present", so `--uncertain-policy` no longer decides anything. Say so; do not
report both flags as if they were independent.

⚠ `--score marginal_presence` multiplies the mention gate into the score —
`P(present) = sigmoid(mention) x q_pos` — which is what `study_presence` actually
asks about, since `classification_logits` is `q`, conditional on mention.
It needs `mention_probabilities` in the `.npz`, written only by runs whose eval
hook collected the gate (added to `image_text_pretrain.py` on 2026-08-20). Older
files raise `ScoreUnavailableError` rather than falling back. Scoring
`study_presence` with the default `conditional_positive` is legitimate but is a
**floor**, not an estimate: it discards the gate the model was trained to
provide (`lambda_gate: 0.5`).

Pinned by `tests/test_label_framing.py` (9 tests).

**Fitting the threshold: `--selection plateau --plateau-fraction 0.95
--min-positive 5`.** Once F1 means something it is worth maximising, and on this
validation split the way you pick the point matters as much as the objective.
Two changes, both chosen by 5-fold x 10-repeat cross-validation *inside*
validation before test was touched:

- `--selection plateau` stands at the **median of every threshold scoring >= 95%
  of the peak** instead of at the peak itself. The peak on 1,808 val studies is
  fitted to which studies happened to land there; the middle of the near-optimal
  region transfers. CV 0.3246 vs argmax 0.3202.
- `--min-positive 5` instead of 20. At 20, `Pleural Other` (15 val positives) and
  `Fracture` (18) fell back to a flat 0.5 — and **`Fracture` was then never
  predicted positive at all, F1 0.0000**. Those two labels alone were **59% of
  the entire gap to the achievable ceiling**. Calibrated properly they land at
  0.231 / 0.244, against test-oracle thresholds of 0.250 / 0.247.

Test result, `run_20260819_xmpoff`, `study_presence` + `marginal_presence`:
**positive_macro_f1 0.3224 -> 0.3397**. Paired bootstrap over 2,000 study
resamples: ΔF1 **+0.0174 [+0.0102, +0.0243]**, Δrecall **+0.0747 [+0.0619,
+0.0880]**, Δprecision **-0.0211 [-0.0303, -0.0137]** — all three significant.
Be honest that it is a **trade**, 3.5:1 in recall's favour, not a free gain.
The ceiling for thresholding alone is 0.3532 (test-oracle thresholds, diagnostic
only, never a reportable number), so 0.3397 captures 97% of it.

**Three things that did NOT help — measured, so do not retry them blind.**

| tried | val-CV macro F1 | verdict |
|---|---|---|
| `m x q` (shipped) | **0.3247** | best |
| `m^a x q^(1-a)`, a fitted per label | 0.3228 | no gain |
| logistic stack on `[logit m, logit q]` | 0.3225 | no gain (AUROC +0.004, F1 flat) |
| `q` alone | 0.3155 | worse, as expected |
| ensemble mean(ep4, ep6, ep9) | 0.3266 | below ep9 alone |
| ensemble mean(ep6, ep9) | 0.3280 | below ep9 alone |
| `checkpoint_9` instead of `checkpoint_best` (ep6) | 0.3291 | **+0.0041, CI crosses zero** (wins 66% of 100 folds) — not enough to switch |

The plain product is already the right way to combine the two heads, and the
epochs are interchangeable, which is consistent with the flat val curve (train
loss 1.856 -> 1.834 over epochs 5-9 while val loss moved 1.8916 -> 1.8902).

⚠⚠ **Precision cannot be bought with a threshold on this model.** Fitting a
precision floor on validation and applying it to test, macro precision saturates
around 0.40 while recall collapses:

| val precision floor | test F1 | test precision | test recall |
|---|---|---|---|
| none (F1-optimal plateau) | **0.3409** | 0.2778 | **0.5044** |
| 0.40 | 0.2646 | 0.3330 | 0.3506 |
| 0.50 | 0.2321 | 0.3975 | 0.2329 |
| 0.70 | 0.1045 | **0.4074** | 0.0815 |

A floor of 0.70 on val delivers 0.41 on test and throws away 84% of the recall.
That ceiling belongs to the **model**, not the decision rule, so anyone asking
for higher precision needs a training change, not a calibration flag.

**Encoder fine-tuning WORKS — run_20260820_ft, 2026-08-21.** The 2026-08-20
conclusion that "precision cannot be bought with a threshold; that ceiling
belongs to the model" was right, and unfreezing the top of both vision encoders
moved it. `model.encoder_finetune` released 31.85M of 181.3M frozen encoder
parameters (ResNet50 `layer4` + projector, CLIP vision blocks 10-11 +
`post_layernorm`) at `init_lr_enc` 1e-5 with `keep_batchnorm_eval`, and every
kappa went to 1 in the same run. 10 epochs, `rc=0`, **15h33m**, 0 restarts,
0 kernel faults, `max mem` 8,976 MiB.

Paired bootstrap over the same 3,269 test studies, `study_presence` +
`marginal_presence`, thresholds calibrated on val with the plateau rule:

| | run_20260819_xmpoff (frozen) | run_20260820_ft | Δ, 95% CI |
|---|---:|---:|:---:|
| `macro_auroc` | 0.7441 | **0.7643** | **+0.0201 [+0.0136, +0.0263]** |
| `positive_macro_f1` | 0.3397 | **0.3542** | **+0.0144 [+0.0053, +0.0236]** |
| `positive_macro_precision` | 0.2776 | **0.2931** | **+0.0154 [+0.0073, +0.0238]** |
| `positive_macro_recall` | 0.5006 | **0.5373** | **+0.0368 [+0.0205, +0.0535]** |

**All four significant and all four in the same direction** — this is a better
model, not a moved operating point, and it is qualitatively different from the
plateau threshold change above, which traded 0.021 precision for 0.075 recall.
AUROC improved on **14 of 14 labels**, biggest at `Pleural Other` +0.0800,
`Lung Lesion` +0.0394, `Pneumothorax` +0.0302. The precision ceiling moved with
it: a 0.70 val floor now delivers **0.5240** on test against 0.4074 before.

⚠ **kappa and the unfreeze went in together, so their contributions are not
separated.** The indirect argument is that kappa only moves the operating point
and validation-fitted thresholds absorb that, whereas AUROC — which kappa cannot
change in principle — is what moved. Attribute it to the unfreeze, but say that
it is an inference, and run an ablation before publishing either as a cause.

Cost: **+37% per epoch** (1.55 h vs 1.13 h), 15h33m vs 12h35m. VRAM +5.3%, so
44% of the card is still free.

**DEEPENING THE UNFREEZE DOES NOT HELP — measured 2026-08-22,
`run_20260821_deep`.** The shallow release was the only lever that ever moved
this model rather than its operating point, so the config also released ResNet50
`layer3` and CLIP vision blocks 8-9: **53.12M of 181.3M**, 158 parameters, 8
patterns. **Nothing else changed** — kappa, batch 16 x 4, `init_lr_enc` 1e-5, 10
epochs and every loss weight identical to `run_20260820_ft` — so this is a clean
ablation of unfreeze *depth*, unlike that run where kappa moved with it. The run
completed cleanly: 10/10 epochs, `rc=0`, **14h03m**, 0 restarts, 0 kernel faults,
`max mem` **9,839 MiB** (the 9,847 the smoke predicted).

Paired bootstrap over the same 3,269 test studies, 2,000 resamples,
`study_presence` + `marginal_presence`, each run using thresholds calibrated on
its own validation split with the plateau rule:

| | run_20260820_ft (shallow) | run_20260821_deep | Δ, 95% CI |
|---|---:|---:|:---:|
| `macro_auroc` | 0.7643 | 0.7692 | +0.0049 [-0.0003, +0.0100] |
| `micro_auroc` | 0.8166 | 0.8187 | **+0.0021 [+0.0001, +0.0040]** |
| `positive_macro_f1` | 0.3542 | 0.3518 | -0.0023 [-0.0128, +0.0088] |
| `positive_macro_precision` | 0.2931 | 0.3008 | +0.0077 [-0.0031, +0.0204] |
| `positive_macro_recall` | 0.5373 | 0.4436 | **-0.0937 [-0.1106, -0.0769]** |
| `macro_specificity` | 0.8020 | 0.8395 | **+0.0375 [+0.0343, +0.0407]** |

`macro_auroc` does not clear zero. The only large significant moves are recall
down and specificity up, which is the **operating point shifting**, not a better
model — the same signature as `run_20260821_ext` below, and the opposite of the
shallow unfreeze, where all four moved together. `micro_auroc` clears zero at
+0.0021, which is too small to act on. Per-label AUROC: deep wins **10 of 14**,
mean +0.0049 (binomial P ≈ 0.09) — against **14 of 14** for the shallow
unfreeze. `macro_auprc` 0.3203 -> 0.3269.

⚠ **`run_20260820_ft` / epoch 9 remains the Stage-1 model.** Nothing about the
deep run justifies switching, and its own `checkpoint_best` is again epoch 9,
the last epoch, at val loss **1.8739** (5-9: 1.8837, 1.8794, 1.8836, 1.8824,
1.8739). Val loss is lower than the shallow run's at every scored epoch, and
that did **not** translate into a better test model — one more reason not to
select on val loss alone here.

⚠ The F1 drop is mostly a calibration artefact, not a model difference:
`Fracture`'s threshold calibrated to **0.407** on deep against 0.235 on shallow
(18 val positives), which on test gives precision 0.043 / recall 0.011, while
its AUROC is unchanged (0.6670 vs 0.6657). Read a per-label F1 swing on the rare
labels as threshold noise until the AUROC agrees with it.

Reproduce with `scripts/calibrate_thresholds.py` then `scripts/evaluate_stage1.py`
using `--label-framing study_presence --score marginal_presence --selection
plateau --plateau-fraction 0.95 --min-positive 5`; the shallow run reproduces its
recorded 0.3542 / 0.7643 exactly under that invocation, which is what makes the
comparison valid. Artifacts live in `~/eval_deep/` on the training host
(git-ignored — they are patient-data derivatives).

Cost: the deep run finished in **14h03m** against the shallow run's 15h33m, and
its `max mem` was 9,839 MiB against 8,976 (+9.6%, 61% of the card). ⚠ Do not
read that wall clock as "deeper is faster" — the two runs met different
dataloader conditions and neither was a controlled throughput measurement.

⚠ **`init_lr_enc` 1e-5 applied to the shallow and the deep slices alike, and
that is the first thing to fix if anyone retries this.** Standard practice is
layer-wise LR decay — lower LR the deeper you go, because early features are more
general and more easily destroyed — and this repo cannot express it:
`runner_base.py` builds a single `encoder_decay`/`encoder_no_decay` pair, routed
by membership in `encoder_finetune_param_names`. The result above is therefore
evidence against *this* deep configuration, not against depth in general; a
retry needs a per-pattern LR group, which is a code change, not a config one.

⚠ **`setup_output_dir` nests the run directory when `$OUT` already exists**
(`runner_base.py:588` appends a timestamp), and `scripts/supervise_stage1.sh`
does `mkdir -p "$OUT"` at line 65. So pointing `OUT` at the inner
`mimic_cxr_full_blip2` directory for a **fresh** run makes LAVIS create
`mimic_cxr_full_blip2/mimic_cxr_full_blip2_<timestamp>/` inside it, and the
supervisor's resume detection then never finds `$OUT/checkpoint_last.pth`.
Pointing `OUT` at the *outer* directory avoids the nesting but breaks resume the
same way, one level up. The working launch for a fresh run is therefore two
steps — start attempt 1 directly with `run.output_dir=<outer>`, then attach the
supervisor to it with `OUT=<outer>/mimic_cxr_full_blip2 ADOPT_PID=<main pid>`:

```bash
setsid nohup env CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled <venv>/bin/python \
    -m pretraining.train --cfg-path pretraining/configs/mimic_cxr_full.yaml \
    --options run.output_dir=$HOME/<run> > $HOME/<run>.log 2>&1 < /dev/null &
# the MAIN pid is the one with ppid 1 -- pgrep returns DataLoader workers too:
ps -eo pid,ppid,cmd | grep "[p]retraining.train" | awk '$2==1 {print $1}'
OUT=$HOME/<run>/mimic_cxr_full_blip2 LOG=$HOME/<run>.log WID=<id> \
    BATCH=16 ACCUM=4 ADOPT_PID=<main pid> \
    setsid nohup bash ~/supervise.sh >>$HOME/<run>.supervise.log 2>&1 &
```

Every supervisor restart after that resumes correctly, because
`run.resume_ckpt_path` then exists and `setup_output_dir` takes its resume-in-
place branch (`runner_base.py:580`) instead of the timestamping one.

⚠ **`run_20260820_ft`'s `checkpoint_best` is epoch 9, the LAST epoch, and val
loss was still falling there** (1.8939 -> 1.8925 -> 1.8939 -> 1.8913 ->
**1.8843**). Run 01 was flat and peaked mid-run at epoch 6. The two loss values
are not comparable across the runs because kappa changed the loss scale — the
comparable thing is the shape. `run_20260821_ext` below settles what that
falling curve was worth: nothing.

Full record: `Test/stage1_test_02/README.md` (git-ignored, on the dev box).

**MORE EPOCHS DO NOT HELP — measured 2026-08-21, `run_20260821_ext`.** The
sentence that used to stand here ("this model has not converged; more epochs is
the cheapest remaining lever") was an inference from a falling val loss and it is
now falsified. Resuming `run_20260820_ft` with `run.max_epoch=15` ran epochs
10-14 in **7h32m**, `rc=0`, 0 restarts, 0 kernel faults — and produced nothing.

- **No epoch beat epoch 9 on val loss.** `checkpoint_best.pth` was never
  rewritten (mtime unchanged for the whole 7.5 h). Train loss kept falling
  1.848 -> 1.836 while the train/val gap widened +0.0363 -> +0.0517.
- **Validation macro AUROC said otherwise, and it was wrong.** Scored under
  `study_presence` + `marginal_presence`, epoch 14 beat epoch 9 by
  **+0.0082, 95% CI [+0.0015, +0.0148]**, P = 0.995 on 1,808 val studies. That
  is a genuine signal by the usual test, and it **did not replicate**: on test
  the same comparison is **-0.0005 [-0.0043, +0.0031]**.
- Test, paired bootstrap over the same 3,269 studies: `macro_auroc` -0.0005,
  `positive_macro_f1` -0.0067, `positive_macro_precision` +0.0015 — **all three
  CIs cross zero**. The only significant moves are `recall` **-0.0826** and
  `specificity` **+0.0244**, which is the recalibrated threshold shifting the
  operating point, not a better model. AUROC improved on **6 of 14** labels,
  i.e. a coin flip; the encoder unfreeze improved 14 of 14.

Three transferable lessons, all cheap to reuse:

1. **A val result whose CI barely excludes zero on ~1,800 studies is "not
   established", not "improved".** Treat [+0.0015, +0.0148] as a null.
2. **Check macro against micro before believing a macro gain.** Here val macro
   AUROC rose while val micro AUROC *fell* (0.8461 -> 0.8406): the apparent gain
   lived in a few rare labels, which are exactly the noisiest estimates.
3. **Decompose the score.** Removing the mention gate showed epoch 14 was
   *worse* than epoch 9 on `q` alone (0.7304 vs 0.7354) — the whole difference
   was luck in the product `m x q`, which is not a mechanism.

**Run 02 / epoch 9 stands as the final Stage-1 model.** Full record:
`Test/stage1_test_03/README.md` (git-ignored, on the dev box).

**The matched training-side lever, not yet run:
`model.loss.lambda_mention_conditioned_cls`.** It trains the joint
`-log(1-m)` / `-log(m) - log(q[y])`, which is *exactly* the quantity
`study_presence` + `marginal_presence` scores; the shipped
`lambda_cls` + `lambda_gate` pair trains the two factors separately and nothing
reconciles them. The 2026-08-16 verdict that it "did not work" was reached under
`masked_polarity`, whose metric masks blank cells and therefore cannot see the
joint at all — that judgment does not carry over and should be re-tested. It
requires `lambda_cls: 0.0` and `lambda_gate: 0.0` (the constructor raises
otherwise) and a full ~12.5 h run.

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

⚠⚠ **`target_valid` admits one-line reports, and Stage 2 trains on them.**
Measured on the val frontal cohort 2026-08-30, n=1,513, every row marked
`target_filter_reason = VALID`:

| `findings_token_count` | min 5 | p1 8 | p5 12 | median 52 | max 146 |
|---|---|---|---|---|---|
| studies at or below 10 tokens | **64 (4.2%)** | | | | |
| studies at or below 15 tokens | **91 (6.0%)** | | | | |

Found from the other end: 154 of those 1,513 studies produce ZERO labelled
sentences in the Stage-2 explanation run, and they turn out to be one-line
reports -- median **1 sentence** and **59 characters**, against 5 and 318 for
the rest, with 55% carrying exactly one sentence. Their content is
overwhelmingly technique and header fragments — **61.3% technical, 7.7%
unclassified** once header phrases are matched, from 35.1% and 33.6% before —
and they contain LESS missed pathology than the general population, **1.8%
against 5.7%**, unchanged by that reclassification.

Harmless for the explanation layer -- they simply score zero coverage. Not
harmless for Stage 2, which trains and is scored on the same cohort, so
roughly 6% of its generation targets are one-line technique fragments. That
dilutes the loss and every NLG metric computed from it.

Not fixed here: the length threshold lives in
`preporcessing/preprocess_mimic_cxr.py` and changing it means rebuilding the
manifests, which is its own piece of work. Recorded so it is not rediscovered
from scratch.

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

⚠⚠ **An evaluate-only run used to score RANDOM WEIGHTS, silently. Fixed
2026-08-20 — `run.resume_ckpt_path` is now mandatory for `run.evaluate=True`.**
`eval_epoch` reloads `checkpoint_best.pth` only when `cur_epoch == "best"`, and
the evaluate-only path passes `"provided"`, while `train()` guarded its
checkpoint load with `if not self.evaluate_only`. So `run.evaluate=True` built
the model, skipped every weight load, and produced a complete val+test metrics
report from random initialisation in 108 s with no error and nothing in the log.
The only tell was that the mention gate came out near-constant (0.334–0.668,
per-label means identical on val and test to three decimals).

`RunnerBase._load_eval_weights()` now loads `run.resume_ckpt_path` — weights
only, no optimizer/scaler/epoch — and **raises** when neither it nor
`load_finetuned`/`load_pretrained` is set. The correct invocation:

```bash
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled python -m pretraining.train \
    --cfg-path pretraining/configs/mimic_cxr_full.yaml \
    --options run.evaluate=True run.test_splits=[val,test] \
      run.resume_ckpt_path=<run>/mimic_cxr_full_blip2/checkpoint_best.pth
```

Note `run.test_splits` is what an evaluate-only run reads (`valid_splits` is
ignored — `runner_base.py:593`), so listing both splits gets val and test in one
pass; files land as `{split}_predictions_epoch_provided.npz`. Setting
`resume_ckpt_path` also makes `setup_output_dir` resolve **in place**, next to
the checkpoint. Confirm it worked by grepping the log for
`Loading evaluation weights from` — **zero matches means the run is worthless.**
Pinned by `tests/test_evaluate_only_weights.py`.

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
- **Never run `ruff check --fix` on a directory. Only on the files you edited.**
  On this branch it twice reformatted `scripts/evaluate_stage1.py` and
  `scripts/export_stage2_prompt_samples.py` — files the change had nothing to do
  with — and both times they were committed before anyone noticed. The second
  time was after a commit message explicitly warning about the first. Scope it:
  `ruff check --fix path/to/file.py`, then `git diff --stat <branch-point>` and
  read the file list before committing.
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
