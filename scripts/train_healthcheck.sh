#!/usr/bin/env bash
# Periodic health report for a Stage-1/Stage-2 run. Read-only: it never
# restarts, kills or changes anything -- scripts/supervise_stage1.sh is the
# thing that intervenes, this is the thing that tells you whether to look.
#
# Prints a compact report and exits with a status the caller can branch on:
#   0 OK      2 WARN (worth a look)      3 ALERT (act now)      4 IDLE (nothing running)
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
if [ -z "$run_row" ]; then
  note "process" "NONE RUNNING"
  bump 4
else
  main=$(echo "$run_row" | awk '{print $1}')
  et=$(echo "$run_row" | awk '{print $3}')
  what=$(echo "$run_row" | grep -oE "pretraining\.train|run_medgemma_qlora\.py|explain_stage2\.py" | head -1)
  note "process" "pid=${main} ${what} elapsed=$(( et / 3600 ))h$(( (et % 3600) / 60 ))m"
fi

# ---- GPU --------------------------------------------------------------------
if command -v nvidia-smi >/dev/null; then
  read -r util mem tot <<<"$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total \
      --format=csv,noheader,nounits | head -1 | tr -d ',')"
  note "gpu" "util=${util}% mem=${mem}/${tot} MiB"
  if [ "$status" -ne 4 ] && [ "${util:-0}" -le "$GPU_IDLE_PCT" ]; then
    note "  ^^ ALERT" "GPU <=${GPU_IDLE_PCT}% with a live process: the DataLoader-deadlock / ntfs-3g-stall signature"
    bump 3
  fi
fi

# ---- progress: newest checkpoint WRITE ---------------------------------------
if [ -n "$RUN_DIR" ] && [ -d "$RUN_DIR" ]; then
  newest=$(find "$RUN_DIR" -name '*.pth' -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1)
  if [ -n "$newest" ]; then
    age=$(( ($(date +%s) - ${newest%% *}) / 60 ))
    note "last checkpoint" "${age} min ago  ($(basename "${newest#* }"))"
    if [ "$status" -ne 4 ] && [ "$age" -gt "$STALL_MIN" ]; then
      note "  ^^ WARN" "no checkpoint write in ${age} min (threshold ${STALL_MIN})"
      bump 2
    fi
  else
    note "last checkpoint" "none yet"
  fi
  note "checkpoints" "$(find "$RUN_DIR" -name '*.pth' | wc -l) files, $(du -sh "$RUN_DIR" 2>/dev/null | cut -f1)"
fi

# ---- training numbers, by name, from the log --------------------------------
if [ -n "$LOG" ] && [ -f "$LOG" ]; then
  note "log" "$(stat -c%s "$LOG" | awk '{printf "%.1f MB", $1/1048576}'), last write $(( ($(date +%s) - $(stat -c%Y "$LOG")) / 60 )) min ago"
  last=$(grep -oE "epoch: \[[0-9]+\]" "$LOG" | tail -1)
  [ -n "$last" ] && note "epoch" "$last"
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
  if [ "$status" -ne 4 ]; then
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
