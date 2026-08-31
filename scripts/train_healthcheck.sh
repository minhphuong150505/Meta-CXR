#!/usr/bin/env bash
# Periodic health report for a Stage-1/Stage-2 run. Read-only: it never
# restarts, kills or changes anything -- scripts/supervise_stage1.sh is the
# thing that intervenes, this is the thing that tells you whether to look.
#
# Prints a compact report and exits with a status the caller can branch on:
#   0 OK      2 WARN (worth a look)      3 ALERT (act now)      4 IDLE (nothing running)
# Set EXPECT_RUNNING=1 for a scheduled experiment: a vanished process is then
# an ALERT instead of ordinary IDLE.
#
# What it checks, and why each one is here rather than the obvious alternative:
#
#   * GPU utilisation, not the log. Both known failure modes on this host -- the
#     DataLoader IPC deadlock and the ntfs-3g stall -- sat at 0% util with the
#     process alive and the log quiet. Utilisation is unbuffered and immediate.
#   * checkpoint WRITE time, not file count. `checkpoint_last.pth` is rewritten
#     in place every save_every_iters, so `ls | wc -l` sits at 1 for a whole
#     multi-hour epoch and looks like no progress.
#   * kernel faults. This machine corrupted memory under load in Aug 2026 and
#     the signature was Oopses scattered across unrelated subsystems. XMP is off
#     now; a fault reappearing is the single most important thing to catch.
#   * mount + disk. /mnt/drive1tb is absent from fstab, so a reboot loses the
#     dataset; /home filling up kills a run at its next checkpoint.
#
# NO PATIENT DATA: it reports counts, rates and metric names only. It never
# echoes a log line that could carry report text, a path, or an identifier.

set -uo pipefail
RUN_DIR="${RUN_DIR:-}"
LOG="${LOG:-}"
EXPECT_RUNNING="${EXPECT_RUNNING:-0}"
STALL_MIN="${STALL_MIN:-45}"     # no checkpoint write for this long -> WARN
GPU_IDLE_PCT="${GPU_IDLE_PCT:-5}"

status=0
note() { printf '  %-22s %s\n' "$1" "$2"; }
bump() { [ "$1" -gt "$status" ] && status="$1"; return 0; }

echo "=== train healthcheck $(date '+%F %T %Z') on $(hostname) ==="

# ---- process ----------------------------------------------------------------
# Match a REAL python invocation of one of the entrypoints, and exclude this
# script's own shell. `pgrep -f` alone matches the shell when the script is
# piped into `bash -c` or `bash -s` -- which is exactly how a cron or an ssh
# heredoc runs it -- so a machine with nothing training reports a live process.
# Requiring bin/python in the command line, and skipping self and parent, is
# what makes IDLE actually mean idle.
run_row=$(ps -eo pid,ppid,etimes,cmd --no-headers 2>/dev/null | awk -v self="$$" -v par="$PPID" '
  $1 != self && $1 != par && $2 != self &&
  /bin\/python/ &&
  (/pretraining\.train/ || /run_medgemma_qlora\.py/ || /explain_stage2\.py/) { print; exit }')
run_live=0
if [ -z "$run_row" ]; then
  if [ "$EXPECT_RUNNING" = "1" ]; then
    note "process" "NONE RUNNING (but EXPECT_RUNNING=1)"
    note "  ^^ ALERT" "the scheduled run exited or never started"
    bump 3
  else
    note "process" "NONE RUNNING"
    bump 4
  fi
else
  run_live=1
  main=$(echo "$run_row" | awk '{print $1}')
  et=$(echo "$run_row" | awk '{print $3}')
  what=$(echo "$run_row" | grep -oE "pretraining\.train|run_medgemma_qlora\.py|explain_stage2\.py" | head -1)
  note "process" "pid=${main} ${what} elapsed=$(( et / 3600 ))h$(( (et % 3600) / 60 ))m"
fi

# ---- GPU --------------------------------------------------------------------
if command -v nvidia-smi >/dev/null; then
  read -r util mem tot <<<"$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total \
      --format=csv,noheader,nounits | head -1 | tr -d ',')"
  read -r temp pwr plim fan <<<"$(nvidia-smi \
      --query-gpu=temperature.gpu,power.draw,power.limit,fan.speed \
      --format=csv,noheader,nounits | head -1 | tr -d ',')"
  note "gpu" "util=${util}% mem=${mem}/${tot} MiB ${temp}C ${pwr}/${plim}W fan ${fan}%"

  # Thermals matter on a multi-day run: this card sustains ~74 C at 100% load
  # with no throttling, so anything near the mid-80s is a change worth seeing.
  if [ "${temp:-0}" -ge 87 ]; then
    note "  ^^ ALERT" "GPU at ${temp}C; this card runs 74C healthy, throttling is imminent or active"
    bump 3
  elif [ "${temp:-0}" -ge 80 ]; then
    note "  ^^ WARN" "GPU at ${temp}C, above its measured 74C steady state"
    bump 2
  fi
  # The authoritative signal is the driver's own throttle flags, not the number.
  thr=$(nvidia-smi --query-gpu=clocks_throttle_reasons.hw_thermal_slowdown,clocks_throttle_reasons.sw_thermal_slowdown \
        --format=csv,noheader 2>/dev/null | grep -ci "Active" || true)
  if [ "${thr:-0}" -gt 0 ] && nvidia-smi --query-gpu=clocks_throttle_reasons.hw_thermal_slowdown,clocks_throttle_reasons.sw_thermal_slowdown --format=csv,noheader 2>/dev/null | grep -qv "Not Active"; then
    note "  ^^ ALERT" "driver reports THERMAL SLOWDOWN active"
    bump 3
  fi
  if [ "$run_live" -eq 1 ] && [ "${util:-0}" -le "$GPU_IDLE_PCT" ]; then
    note "  ^^ ALERT" "GPU <=${GPU_IDLE_PCT}% with a live process: the DataLoader-deadlock / ntfs-3g-stall signature"
    bump 3
  fi
fi

# ---- CPU and drives ---------------------------------------------------------
# Thresholds are the ones the hardware declares (coretemp high=80 crit=100,
# NVMe high=89.8 crit=94.8), not numbers picked here. Measured steady state
# under a GPU-bound run: package 55 C, most cores 35-42 C, NVMe 38-39 C -- the
# CPU only feeds dataloader workers, so a hot package means something changed.
if command -v sensors >/dev/null; then
    pkg=$(sensors 2>/dev/null | awk '/^Package id 0:/ {gsub(/[+°C]/,"",$4); print int($4); exit}')
    [ -n "$pkg" ] && note "cpu package" "${pkg}C (high 80, crit 100)"
    if [ "${pkg:-0}" -ge 90 ]; then
        note "  ^^ ALERT" "CPU at ${pkg}C, past its declared high of 80"
        bump 3
    elif [ "${pkg:-0}" -ge 80 ]; then
        note "  ^^ WARN" "CPU at ${pkg}C, at its declared high"
        bump 2
    fi
    hot=$(sensors 2>/dev/null | awk '/^Composite:/ {gsub(/[+°C]/,"",$2); if ($2+0>m) m=$2+0} END {print int(m)}')
    [ -n "$hot" ] && [ "$hot" -gt 0 ] && note "nvme hottest" "${hot}C (high 89.8, crit 94.8)"
    if [ "${hot:-0}" -ge 85 ]; then
        note "  ^^ ALERT" "NVMe at ${hot}C, approaching its critical point"
        bump 3
    fi
fi

# ---- progress: newest checkpoint WRITE ---------------------------------------
if [ -n "$RUN_DIR" ] && [ -d "$RUN_DIR" ]; then
  # Stage 1 rewrites checkpoint_last.pth. Stage 2 rewrites the recovery
  # adapter/trainer state under checkpoints/last. Track both formats by mtime;
  # counting files cannot show progress when the same path is overwritten.
  newest=$(find "$RUN_DIR" -type f \( \
      -name '*.pth' -o \
      -name 'adapter_model.safetensors' -o \
      -name 'trainer_state.pt' \
    \) -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1)
  if [ -n "$newest" ]; then
    newest_epoch=${newest%% *}
    newest_epoch=${newest_epoch%%.*}
    age=$(( ($(date +%s) - newest_epoch) / 60 ))
    note "last checkpoint" "${age} min ago  ($(basename "${newest#* }"))"
    if [ "$run_live" -eq 1 ] && [ "$age" -gt "$STALL_MIN" ]; then
      note "  ^^ WARN" "no checkpoint write in ${age} min (threshold ${STALL_MIN})"
      bump 2
    fi
  else
    note "last checkpoint" "none yet"
  fi
  checkpoint_count=$(find "$RUN_DIR" -type f \( \
      -name '*.pth' -o \
      -name 'adapter_model.safetensors' -o \
      -name 'trainer_state.pt' \
    \) | wc -l)
  note "checkpoints" "${checkpoint_count} files, $(du -sh "$RUN_DIR" 2>/dev/null | cut -f1)"
fi

# ---- training numbers, by name, from the log --------------------------------
if [ -n "$LOG" ] && [ -f "$LOG" ]; then
  note "log" "$(stat -c%s "$LOG" | awk '{printf "%.1f MB", $1/1048576}'), last write $(( ($(date +%s) - $(stat -c%Y "$LOG")) / 60 )) min ago"
  # Epoch AND iteration, and the epoch index is NEVER pinned. A monitoring
  # grep hardcoded to epoch 0 kept returning a stale line after a run moved to
  # epoch 1, and one epoch's iterations divided by two epochs' wall clock
  # produced a factor-of-two throughput error that stood in CLAUDE.md until the
  # run's own timestamps contradicted it.
  last=$(tr '\r' '\n' < "$LOG" | grep -oE "epoch: \[[0-9]+\] +\[ *[0-9]+/[0-9]+\]" | tail -1)
  [ -n "$last" ] && note "position" "$last"
  for k in time: loss_cls loss_mpc loss_distill loss_itc loss_itm loss_lm; do
    v=$(grep -oE "$k [0-9.]+" "$LOG" | tail -1 | awk '{print $2}')
    [ -n "$v" ] && note "$k" "$v"
  done
  m=$(grep -oE "max mem: [0-9]+" "$LOG" | tail -1 | awk '{print $3}')
  [ -n "$m" ] && note "max mem" "${m} MiB"
fi

# ---- the thing that actually killed runs here -------------------------------
faults=$(journalctl -b -k --no-pager 2>/dev/null | grep -icE "Oops|BUG:|LIST_POISON|general protection|Call Trace")
note "kernel faults" "${faults:-0} this boot"
if [ "${faults:-0}" -gt 0 ]; then
  note "  ^^ ALERT" "memory-corruption signature returned; XMP should be OFF on this host"
  bump 3
fi

# ---- storage ----------------------------------------------------------------
free=$(df -BG --output=avail /home 2>/dev/null | tail -1 | tr -dc '0-9')
note "/home free" "${free} GB"
if [ "${free:-999}" -lt 20 ]; then note "  ^^ ALERT" "under 20 GB; the next checkpoint write may fail"; bump 3
elif [ "${free:-999}" -lt 50 ]; then note "  ^^ WARN" "under 50 GB"; bump 2; fi
if mountpoint -q /mnt/drive1tb 2>/dev/null; then
  note "/mnt/drive1tb" "mounted ($(findmnt -no FSTYPE /mnt/drive1tb))"
else
  note "/mnt/drive1tb" "NOT MOUNTED"
  if [ "$run_live" -eq 1 ]; then
    note "  ^^ ALERT" "dataset gone mid-run; it is not in fstab and every reboot loses it"
    bump 3
  else
    note "  ^^ note" "not in fstab; remount before the next run (needs sudo, see CLAUDE.md)"
  fi
fi

case $status in
  0) echo "=== OK ===" ;;
  2) echo "=== WARN: worth a look ===" ;;
  3) echo "=== ALERT: act now ===" ;;
  4) echo "=== IDLE: no run in progress ===" ;;
esac
exit $status
