#!/usr/bin/env bash
set -euo pipefail

# Tail the latest (or specified) remote training log

usage() {
  cat << 'USAGE'
Usage: scripts/remote_tail.sh --host user@host [--remote-dir "~/sdxl-siglip"] [--ssh "ssh -p 22"] \
       [--accept-new-hostkey] [--no-accept-new-hostkey] [--log <file>] [--lines 200]

Options:
  --host         SSH target (required)
  --remote-dir   Remote repo dir (default: ~/sdxl-siglip)
  --ssh          SSH command (default: ssh)
  --accept-new-hostkey    Auto-accept new host keys (default)
  --no-accept-new-hostkey Disable auto-accept; use strict hostkey checks
  --log          Specific log file (otherwise pick newest train_*.log)
  --lines        Number of lines to show initially (default: 200)
USAGE
}

HOST=""
REMOTE_DIR="~/sdxl-siglip"
SSH_CMD="ssh"
LOG_FILE=""
LINES=200
SSH_OPTS=()
ACCEPT_NEW=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST="$2"; shift 2;;
    --remote-dir) REMOTE_DIR="$2"; shift 2;;
    --ssh) SSH_CMD="$2"; shift 2;;
    --accept-new-hostkey) ACCEPT_NEW=1; shift;;
    --no-accept-new-hostkey) ACCEPT_NEW=0; shift;;
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

# Configure SSH options
if [[ $ACCEPT_NEW -eq 1 ]]; then
  SSH_OPTS+=( -o StrictHostKeyChecking=accept-new -o UpdateHostKeys=yes )
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

$SSH_CMD "${SSH_OPTS[@]}" "$HOST" bash -lc "$REMOTE_CMD"
