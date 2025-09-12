#!/usr/bin/env bash
set -euo pipefail

# Tail the latest (or specified) remote training log

usage() {
  cat << 'USAGE'
Usage: scripts/remote_tail.sh --host user@host [--remote-dir "~/sdxl-siglip"] [--ssh "ssh -p 22"] [--log <file>] [--lines 200]

Options:
  --host         SSH target (required)
  --remote-dir   Remote repo dir (default: ~/sdxl-siglip)
  --ssh          SSH command (default: ssh)
  --log          Specific log file (otherwise pick newest train_*.log)
  --lines        Number of lines to show initially (default: 200)
USAGE
}

HOST=""
REMOTE_DIR="~/sdxl-siglip"
SSH_CMD="ssh"
LOG_FILE=""
LINES=200

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST="$2"; shift 2;;
    --remote-dir) REMOTE_DIR="$2"; shift 2;;
    --ssh) SSH_CMD="$2"; shift 2;;
    --log) LOG_FILE="$2"; shift 2;;
    --lines) LINES="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2;;
  esac
done

if [[ -z "$HOST" ]]; then
  echo "Error: --host is required" >&2
  usage
  exit 2
fi

REMOTE_CMD='set -euo pipefail
cd '"$REMOTE_DIR"' 2>/dev/null || { echo "Remote dir not found: '"$REMOTE_DIR"'"; exit 3; }
file='"$LOG_FILE"'
if [[ -z "$LOG_FILE" ]]; then
  file=$(ls -1t train_*.log 2>/dev/null | head -n1 || true)
fi
if [[ -z "$file" ]]; then
  echo "No log files found (train_*.log)"
  exit 4
fi
echo "Tailing: $file"
tail -n '"$LINES"' -f "$file"
'

$SSH_CMD "$HOST" bash -lc "$REMOTE_CMD"

