#!/usr/bin/env bash
# Stage-1 supervisor with a PROGRESS watchdog.
#
# Lives in the repo as of 2026-08-18 so it is versioned with the recipe it
# launches. The copy on the training host (~/supervise.sh) is a deployment of
# this file: `cp scripts/supervise_stage1.sh ~/supervise.sh`.
#
# Why it exists: the run has died four different ways (Xid 31 MMU fault, a bare
# SIGABRT, a DataLoader IPC deadlock, and a whole-machine hang) and only some of
# them showed up as a process exit. The deadlock left the process alive and idle
# forever, so restarting on exit alone is not enough -- this watches for forward
# progress and kills the whole process group when it stops.
#
# ⚠ STARTUP IS NOT A STALL. The first version applied the stall timeout from the
# moment of launch, and would have killed a healthy run ~12 minutes in while it
# was still downloading blip2_pretrained.pth (1.9 GB) -- there are no "Train:"
# lines during startup, so the watchdog cannot tell "not begun" from "hung" by
# line count alone. Startup gets its own, much larger budget.
set -u

REPO="${REPO:-$HOME/Meta-CXR}"
PY="${PY:-$HOME/.venvs/meta-cxr-stage1-311/bin/python}"
CFG="${CFG:-pretraining/configs/mimic_cxr_full.yaml}"
OUT="${OUT:-$HOME/run_20260818_lr2e5}"
LOG="${LOG:-$OUT.log}"
WID="${WID:-stage1-lr2e5-20260818}"

# stdout is BLOCK-BUFFERED here: the run writes to a file, not a tty, so
# MetricLogger print() lines land in ~8 KB bursts -- measured 2026-08-18 at one
# flush per ~75 s (0 -> 17 train lines in a single flush). The line count is
# therefore a COARSE progress signal, and a run that merely slows down stretches
# the gap proportionally. Hence a generous line timeout plus two sharper signals.
STALL_MINUTES="${STALL_MINUTES:-20}"      # no new train line once training has begun
STARTUP_MINUTES="${STARTUP_MINUTES:-45}"  # before the FIRST train line ever appears
# GPU idle is the signal that actually identifies the failures seen on this box:
# the DataLoader IPC deadlock and the ntfs-3g stall both sat at 0% utilisation
# while the process stayed alive. Unbuffered and immediate, unlike the log.
GPU_IDLE_MINUTES="${GPU_IDLE_MINUTES:-6}"
# 0 = run until the training completes or a real fault stops it. The old default
# of 8 could retire a healthy multi-day run purely on restart count; "supervise
# it continuously" means the attempt budget must not be the thing that ends it.
# The give-up rule below is progress-based instead, which is the honest test.
MAX_ATTEMPTS="${MAX_ATTEMPTS:-0}"
# Consecutive failures that produced NO new checkpoint WRITE before giving up.
MAX_FAILS_WITHOUT_PROGRESS="${MAX_FAILS_WITHOUT_PROGRESS:-3}"
BATCH="${BATCH:-8}"
ACCUM="${ACCUM:-8}"
MIN_BATCH="${MIN_BATCH:-2}"
RESTART_SLEEP="${RESTART_SLEEP:-30}"
ADOPT_PID="${ADOPT_PID:-}"                # attach to an already-running train pid

mkdir -p "$OUT"
cd "$REPO" || exit 1
: >>"$LOG"

say() { echo "[supervise $(date '+%F %T')] $*" | tee -a "$LOG"; }

# PROGRESS IS A CHECKPOINT *WRITE*, NOT A CHECKPOINT *FILE*. This used to count
# files with `ls | wc -l`, which was correct only while checkpoints were written
# once per epoch under distinct names. Since run.save_every_iters landed,
# checkpoint_last.pth is REWRITTEN IN PLACE every 1000 iterations, so the file
# count sits at 1 for an entire ~4 h epoch: three restarts inside one epoch --
# exactly the window the mid-epoch checkpointing was added to protect -- would
# have tripped the "nothing is being learned" abort while the run was in fact
# making steady progress. The newest mtime moves on every write and does not.
ckpt_stamp() {
    local newest
    newest=$(find "$OUT" -maxdepth 1 -name '*.pth' -printf '%T@\n' 2>/dev/null \
             | sort -n | tail -1)
    echo "${newest:-0}"
}

# `grep -c` prints 0 AND exits 1 when there is no match, so the old
# `grep -c ... || echo 0` emitted TWO lines ("0\n0"). Every `[ "$n" -gt ... ]`
# against that raised "integer expression expected" and evaluated false, which
# happened to look like "no progress" during startup and hid the error in stderr.
train_lines() {
    local n
    n=$(grep -c "Train: data epoch" "$LOG" 2>/dev/null | head -1)
    case "$n" in (''|*[!0-9]*) n=0 ;; esac
    echo "$n"
}

say "supervisor starting: OUT=$OUT LOG=$LOG batch=$BATCH accum=$ACCUM max_attempts=${MAX_ATTEMPTS:-unlimited}"

attempt=0
last_stamp=$(ckpt_stamp)
fails_without_progress=0

while [ "$MAX_ATTEMPTS" -eq 0 ] || [ "$attempt" -lt "$MAX_ATTEMPTS" ]; do
    attempt=$((attempt + 1))
    adopted=0

    if [ -n "$ADOPT_PID" ] && kill -0 "$ADOPT_PID" 2>/dev/null; then
        train_pid="$ADOPT_PID"
        adopted=1
        ADOPT_PID=""
        say "attempt $attempt: ADOPTING already-running pid $train_pid"
    else
        opts=(run.output_dir="$OUT"
              run.batch_size_train="$BATCH"
              run.batch_size_eval="$BATCH"
              run.accum_grad_iters="$ACCUM"
              run.wandb_run_id="$WID"
              run.wandb_resume=allow)
        if [ -f "$OUT/checkpoint_last.pth" ]; then
            opts+=(run.resume_ckpt_path="$OUT/checkpoint_last.pth")
            say "attempt $attempt: RESUMING from checkpoint_last (batch $BATCH x accum $ACCUM)"
        else
            say "attempt $attempt: FRESH start (batch $BATCH x accum $ACCUM)"
        fi
        setsid env CUDA_VISIBLE_DEVICES=0 \
            PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
            "$PY" -m pretraining.train --cfg-path "$CFG" --options "${opts[@]}" \
            >>"$LOG" 2>&1 &
        train_pid=$!
    fi
    train_pgid=$(ps -o pgid= -p "$train_pid" 2>/dev/null | tr -d ' ')
    say "  pid=$train_pid pgid=$train_pgid adopted=$adopted"

    # --- watchdog ------------------------------------------------------------
    stalled=0
    prev_lines=$(train_lines)
    started=0; [ "$prev_lines" -gt 0 ] && started=1
    quiet_min=0
    idle_min=0
    while kill -0 "$train_pid" 2>/dev/null; do
        sleep 60
        now_lines=$(train_lines)
        util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
        case "$util" in (''|*[!0-9]*) util=100 ;; esac
        if [ "$started" -eq 1 ] && [ "$util" -le 5 ]; then
            idle_min=$((idle_min + 1))
        else
            idle_min=0
        fi
        if [ "$now_lines" -gt "$prev_lines" ]; then
            prev_lines=$now_lines; quiet_min=0
            if [ "$started" -eq 0 ]; then
                started=1
                say "  first training line seen -- stall watchdog ${STALL_MINUTES}m, GPU-idle watchdog ${GPU_IDLE_MINUTES}m"
            fi
        else
            quiet_min=$((quiet_min + 1))
        fi
        # A mid-epoch checkpoint write is forward progress even when the log is
        # mid-buffer, so it resets the stall clock. It does NOT reset the GPU
        # clock: a hung run cannot write a checkpoint, but a run that wrote one
        # and then deadlocked would otherwise get a free 20 minutes.
        now_stamp=$(ckpt_stamp)
        if [ "$now_stamp" != "$last_stamp" ]; then
            last_stamp="$now_stamp"; quiet_min=0
            fails_without_progress=0
        fi
        if [ "$started" -eq 1 ]; then limit=$STALL_MINUTES; else limit=$STARTUP_MINUTES; fi
        if [ "$quiet_min" -ge "$limit" ] || [ "$idle_min" -ge "$GPU_IDLE_MINUTES" ]; then
                if [ "$idle_min" -ge "$GPU_IDLE_MINUTES" ]; then
                    say "  GPU IDLE: utilisation <=5% for ${idle_min}m while alive -- dumping state and killing pgid $train_pgid"
                elif [ "$started" -eq 1 ]; then
                    say "  STALL: no new train line for ${quiet_min}m -- dumping state and killing pgid $train_pgid"
                else
                    say "  STARTUP TIMEOUT: never reached the training loop in ${quiet_min}m -- killing pgid $train_pgid"
                fi
                ps -eo pid,stat,wchan:24,comm | grep -E "python|pt_data_worker|ntfs" >>"$LOG" 2>&1
                nvidia-smi >>"$LOG" 2>&1
                kill -9 -"$train_pgid" 2>/dev/null
                stalled=1
                break
        fi
    done

    if [ "$adopted" -eq 1 ]; then
        # Not our child, so there is no exit status to collect. Success is only
        # claimable from the log; anything else is treated as a failure.
        if tail -n 50 "$LOG" | grep -q "Training time"; then rc=0; else rc=98; fi
    else
        wait "$train_pid" 2>/dev/null; rc=$?
    fi
    [ "$stalled" -eq 1 ] && rc=99
    say "  exited rc=$rc"

    if [ "$rc" -eq 0 ]; then
        say "TRAINING COMPLETED"
        exit 0
    fi

    # --- OOM fallback --------------------------------------------------------
    # Halving the batch and doubling the accumulation leaves the effective batch
    # and therefore the LR schedule untouched. An OOM is a capacity finding, not
    # a fault, so it does NOT consume the no-progress budget: the next attempt
    # is a genuinely different configuration.
    if tail -n 400 "$LOG" | grep -qE "OutOfMemoryError|CUDA out of memory"; then
        if [ "$BATCH" -gt "$MIN_BATCH" ]; then
            BATCH=$((BATCH / 2)); ACCUM=$((ACCUM * 2))
            say "  OOM detected -> retrying at batch $BATCH x accum $ACCUM (effective unchanged)"
            sleep "$RESTART_SLEEP"
            continue
        fi
        say "  OOM at batch $BATCH, the floor -- ABORTING, needs a human"
        exit 1
    fi

    # --- give up if nothing is being learned ---------------------------------
    now_stamp=$(ckpt_stamp)
    if [ "$now_stamp" != "$last_stamp" ]; then
        last_stamp="$now_stamp"; fails_without_progress=0
    else
        fails_without_progress=$((fails_without_progress + 1))
        if [ "$fails_without_progress" -ge "$MAX_FAILS_WITHOUT_PROGRESS" ]; then
            say "$fails_without_progress consecutive failures with no checkpoint write -- ABORTING, needs a human"
            exit 1
        fi
        say "  failure $fails_without_progress/$MAX_FAILS_WITHOUT_PROGRESS with no checkpoint write"
    fi
    say "  restarting in ${RESTART_SLEEP}s"
    sleep "$RESTART_SLEEP"
done

say "exhausted $MAX_ATTEMPTS attempts -- ABORTING"
exit 1
