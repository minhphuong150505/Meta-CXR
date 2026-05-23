#!/usr/bin/env bash
# Helpers dùng chung cho run_stage1.sh và run_stage2.sh.

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

require_kaggle_cli() {
  if [ "${EUID:-0}" -eq 0 ]; then
    log "ERROR: Không chạy script này bằng sudo. Hãy chạy: bash run_stage1.sh hoặc bash run_stage2.sh"
    exit 1
  fi

  local kaggle_venv="${KAGGLE_VENV:-$HOME/.venvs/kaggle-cli}"
  if [ -x "$kaggle_venv/bin/kaggle" ]; then
    export PATH="$kaggle_venv/bin:$PATH"
  fi

  if ! command -v kaggle >/dev/null 2>&1; then
    log "ERROR: Không tìm thấy lệnh kaggle."
    log "  Chạy trước: bash setup_vm.sh"
    log "  Hoặc: export PATH=\"$HOME/.venvs/kaggle-cli/bin:\$PATH\""
    exit 1
  fi

  local kaggle_config_dir="${KAGGLE_CONFIG_DIR:-$HOME/.kaggle}"
  local kaggle_json="$kaggle_config_dir/kaggle.json"
  if [ ! -f "$kaggle_json" ]; then
    log "ERROR: Thiếu Kaggle API token: $kaggle_json"
    log "  Lấy token ở Kaggle Settings -> Create New Token, rồi copy file kaggle.json vào đường dẫn trên."
    exit 1
  fi

  chmod 600 "$kaggle_json" 2>/dev/null || true
}

# poll_kernel SLUG MAX_HOURS
# Poll kaggle kernel status mỗi POLL_INTERVAL_SECS giây.
# Exit 0 khi complete, 1 khi error/cancel, 2 khi timeout.
poll_kernel() {
  local slug="$1"
  local max_hours="${2:-13}"
  local interval="${POLL_INTERVAL_SECS:-300}"
  local max_iters=$(( max_hours * 3600 / interval ))
  local i=0
  local raw status

  log "Polling kernel $slug (interval=${interval}s, max=${max_hours}h)"
  while [ $i -lt $max_iters ]; do
    raw=$(kaggle kernels status "$slug" 2>&1 || true)
    status=$(echo "$raw" | grep -oE '"[a-zA-Z]+"' | head -n1 | tr -d '"')
    log "  iter=$i status=$status"

    case "$status" in
      complete)
        log "Kernel finished successfully."
        return 0
        ;;
      error|cancelRequested|cancelAcknowledged)
        log "Kernel failed with status: $status"
        log "Raw output: $raw"
        return 1
        ;;
      running|queued|"")
        ;;
      *)
        log "Unknown status '$status', continuing to poll."
        ;;
    esac

    sleep "$interval"
    i=$((i + 1))
  done

  log "Timeout after ${max_hours}h"
  return 2
}

# upload_gcs SRC DST — retry 3 lần.
upload_gcs() {
  local src="$1"
  local dst="$2"
  local attempt
  for attempt in 1 2 3; do
    log "gsutil -m cp -r $src $dst (attempt $attempt)"
    if gsutil -m cp -r "$src" "$dst"; then
      return 0
    fi
    sleep 30
  done
  log "gsutil upload failed after 3 attempts."
  return 1
}
