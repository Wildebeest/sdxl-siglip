#!/usr/bin/env bash
set -euo pipefail

# Kill a running remote training started by scripts/remote_train.sh
# Uses train.pid if present; falls back to pkill by command line.

usage() {
  cat << 'USAGE'
Usage: scripts/remote_kill.sh --host user@host [--remote-dir "~/sdxl-siglip"] [--ssh "ssh -p 22"] \
       [--accept-new-hostkey] [--no-accept-new-hostkey] [--grace 10] [--force]

Options:
  --host         SSH target (required)
  --remote-dir   Remote repo dir (default: ~/sdxl-siglip)
  --ssh          SSH command (default: ssh)
  --accept-new-hostkey    Auto-accept new host keys (default)
  --no-accept-new-hostkey Disable auto-accept; use strict hostkey checks
  --grace        Seconds to wait after SIGTERM before SIGKILL (default: 10)
  --force        Send SIGKILL immediately after SIGTERM attempt
USAGE
}

HOST=""
REMOTE_DIR="~/sdxl-siglip"
SSH_CMD="ssh"
GRACE=10
FORCE=0
SSH_OPTS=()
ACCEPT_NEW=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST="$2"; shift 2;;
    --remote-dir) REMOTE_DIR="$2"; shift 2;;
    --ssh) SSH_CMD="$2"; shift 2;;
    --accept-new-hostkey) ACCEPT_NEW=1; shift;;
    --no-accept-new-hostkey) ACCEPT_NEW=0; shift;;
    --grace) GRACE="$2"; shift 2;;
    --force) FORCE=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2;;
  esac
done

if [[ -z "$HOST" ]]; then
  echo "Error: --host is required" >&2
  usage
  exit 2
fi

# Configure SSH options
if [[ $ACCEPT_NEW -eq 1 ]]; then
  SSH_OPTS+=( -o StrictHostKeyChecking=accept-new -o UpdateHostKeys=yes )
fi

REMOTE_CMD='set -euo pipefail
cd '"$REMOTE_DIR"' 2>/dev/null || { echo "Remote dir not found: '"$REMOTE_DIR"'"; exit 3; }

kill_pid() {
  local pid="$1"
  if [[ -z "$pid" ]]; then return 0; fi
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "PID $pid not running"
    return 0
  fi
  echo "Sending SIGTERM to $pid"
  kill -TERM "$pid" || true
}

force_kill() {
  local pid="$1"
  if [[ -z "$pid" ]]; then return 0; fi
  if kill -0 "$pid" 2>/dev/null; then
    echo "Sending SIGKILL to $pid"
    kill -KILL "$pid" || true
  fi
}

pick_pid_file() {
  # Prefer newest per-run pid under runs/, then fallback to repo root
  local f
  f=$(ls -1t runs/*/train.pid 2>/dev/null | head -n1 || true)
  if [[ -z "$f" && -f train.pid ]]; then
    f="train.pid"
  fi
  echo "$f"
}

PID_FILE=$(pick_pid_file)
if [[ -n "$PID_FILE" && -f "$PID_FILE" ]]; then
  echo "Using PID file: $PID_FILE"
  PID=$(cat "$PID_FILE" || true)
  echo "PID: $PID"
  kill_pid "$PID"
  sleep '"$GRACE"'
  if kill -0 "$PID" 2>/dev/null; then
    echo "PID $PID still running after '"$GRACE"'s"
    '"[[ $FORCE -eq 1 ]] && echo force || echo attempting"' >/dev/null
    force_kill "$PID"
  fi
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "Stopped process $PID"
    rm -f "$PID_FILE"
  fi
else
  echo "No PID file found; falling back to pkill by command line"
  # Try to catch foreground or detached runs
  pkill -f "uv run python train_baseline.py" || true
  pkill -f "python train_baseline.py" || true
fi

# Show remaining python processes for visibility
echo "Remaining training-like PIDs (if any):"
pgrep -fa "train_baseline.py" || true
'

$SSH_CMD "${SSH_OPTS[@]}" "$HOST" bash -lc "$REMOTE_CMD"
