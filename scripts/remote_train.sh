#!/usr/bin/env bash
set -euo pipefail

# Remote SDXL+SigLIP training bootstrapper
# - Installs Astral's uv on the remote host if missing
# - Rsyncs the current local repo to the remote directory
# - Copies local .env to remote (if present)
# - Runs `uv sync`
# - Smoke tests SigLIP tokenizer load
# - Starts training with provided --train-urls

usage() {
  cat << 'USAGE'
Usage: scripts/remote_train.sh --host user@host --train-urls "wds_pattern" \
       [--remote-dir "~/sdxl-siglip"] [--background] [--no-background] [--ssh "ssh -p 22"] \
       [--accept-new-hostkey] [--no-accept-new-hostkey] \
       [--extra-args "--batch_size 1 --max_steps 1000 --wandb_project sdxl-siglip"]

Required:
  --host           SSH target, e.g. ubuntu@10.0.0.5
  --train-urls     WebDataset shard pattern, e.g. /data/shards/{0000..0999}.tar

Optional:
  --remote-dir     Directory on remote host (default: ~/sdxl-siglip)
  --ssh            SSH command (default: ssh)
  --accept-new-hostkey      Auto-accept new host keys (default)
  --no-accept-new-hostkey   Disable auto-accept; use strict hostkey checks
  --background     Run training under nohup in background (default)
  --no-background  Run training attached in foreground
  --extra-args     Extra args forwarded to train_baseline.py

Notes:
  - Expects a local .env; will scp to remote repo root.
  - Ensures uv is on PATH (~/.local/bin and ~/.cargo/bin included).
USAGE
}

HOST=""
TRAIN_URLS=""
REMOTE_DIR="~/sdxl-siglip"
SSH_CMD="ssh"
BACKGROUND=1
EXTRA_ARGS=""
SSH_OPTS=()
ACCEPT_NEW=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST="$2"; shift 2;;
    --train-urls) TRAIN_URLS="$2"; shift 2;;
    --remote-dir) REMOTE_DIR="$2"; shift 2;;
    --ssh) SSH_CMD="$2"; shift 2;;
    --accept-new-hostkey) ACCEPT_NEW=1; shift;;
    --no-accept-new-hostkey) ACCEPT_NEW=0; shift;;
    --background) BACKGROUND=1; shift;;
    --no-background) BACKGROUND=0; shift;;
    --extra-args) EXTRA_ARGS="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2;;
  esac
done

if [[ -z "$HOST" || -z "$TRAIN_URLS" ]]; then
  echo "Error: --host and --train-urls are required" >&2
  usage
  exit 2
fi

# Configure SSH options
if [[ $ACCEPT_NEW -eq 1 ]]; then
  # Safer than disabling checks: accept new keys, refuse changed ones.
  SSH_OPTS+=( -o StrictHostKeyChecking=accept-new -o UpdateHostKeys=yes )
fi

RSYNC_SSH="$SSH_CMD ${SSH_OPTS[*]}"

# Derive scp options from SSH_CMD (port, identity, -o flags)
SCP_OPTS=("${SSH_OPTS[@]}")
SSH_PORT=""
if [[ -n "$SSH_CMD" ]]; then
  read -r -a _SSH_WORDS <<< "$SSH_CMD"
  for ((i=0; i<${#_SSH_WORDS[@]}; i++)); do
    w="${_SSH_WORDS[$i]}"
    case "$w" in
      -p)
        if (( i+1 < ${#_SSH_WORDS[@]} )); then SSH_PORT="${_SSH_WORDS[$((i+1))]}"; fi; ((i++));;
      -i)
        if (( i+1 < ${#_SSH_WORDS[@]} )); then SCP_OPTS+=( -i "${_SSH_WORDS[$((i+1))]}" ); fi; ((i++));;
      -o)
        if (( i+1 < ${#_SSH_WORDS[@]} )); then SCP_OPTS+=( -o "${_SSH_WORDS[$((i+1))]}" ); fi; ((i++));;
    esac
  done
fi
if [[ -n "$SSH_PORT" ]]; then
  SCP_OPTS+=( -P "$SSH_PORT" )
fi

if [[ ! -f .env ]]; then
  echo "Warning: local .env not found; continuing without copying secrets" >&2
fi

REMOTE_DIR_REMOTE="${REMOTE_DIR/#\~/\$HOME}"
REMOTE_DIR_TILDE="$REMOTE_DIR"
REMOTE_BOOTSTRAP='set -euo pipefail
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv on remote..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi
mkdir -p '"$REMOTE_DIR_REMOTE"'
'

echo "[1/7] Bootstrapping remote host and directory..."
$SSH_CMD "${SSH_OPTS[@]}" "$HOST" bash -lc "$REMOTE_BOOTSTRAP"

echo "[2/7] Rsyncing repo to remote..."
RSYNC_EXCLUDES=(
  --exclude '.git'
  --exclude '.venv'
  --exclude '__pycache__'
  --exclude '*.log'
)
rsync -az --delete "${RSYNC_EXCLUDES[@]}" -e "$RSYNC_SSH" ./ "$HOST":"$REMOTE_DIR_TILDE/"

if [[ -f .env ]]; then
  echo "[3/7] Copying .env to remote..."
  scp "${SCP_OPTS[@]}" .env "$HOST":"$REMOTE_DIR_TILDE/.env"
fi

echo "[4/7] Verifying uv and repo on remote..."
$SSH_CMD "${SSH_OPTS[@]}" "$HOST" bash -lc 'set -e; UVBIN="$HOME/.local/bin/uv"; if [[ -x "$UVBIN" ]]; then echo "uv at $UVBIN"; else echo "uv not in $UVBIN; attempting \"command -v uv\""; command -v uv; fi; test -d '"$REMOTE_DIR_REMOTE"' && echo "repo dir OK"'

echo "[5/7] Syncing dependencies with uv..."
$SSH_CMD "${SSH_OPTS[@]}" "$HOST" bash -lc 'export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"; cd '"$REMOTE_DIR_REMOTE"'; ("$HOME/.local/bin/uv" sync) || uv sync'

echo "[6/7] Smoke test: load SigLIP tokenizer..."
$SSH_CMD "${SSH_OPTS[@]}" "$HOST" bash -lc 'export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"; cd '"$REMOTE_DIR_REMOTE"'; ("$HOME/.local/bin/uv" run python - <<"PY") || uv run python - <<"PY"
from transformers import AutoTokenizer
try:
    tok = AutoTokenizer.from_pretrained("google/siglip-so400m-patch14-384")
    print("SigLIP tokenizer OK:", type(tok).__name__)
except Exception as e:
    import sys
    print("SigLIP tokenizer FAILED:", e)
    sys.exit(1)
PY'

if [[ "$BACKGROUND" -eq 1 ]]; then
  echo "[7/7] Starting background training with nohup..."
START_BG='set -euo pipefail
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
cd '"$REMOTE_DIR_REMOTE"'
ts=$(date +%Y%m%d_%H%M%S)
rnd=${RANDOM}
run_id="${ts}_${rnd}"
run_dir="'$REMOTE_DIR_REMOTE'/runs/${run_id}"
mkdir -p "$run_dir"
log="$run_dir/train.log"
ln -sfn "$run_dir" latest
echo "Run ID: $run_id"
echo "Logging to $log"
set +o allexport
if [ -f .env ]; then
  set -o allexport; . ./.env; set +o allexport
fi
TRAIN_URLS_POS="$1"; shift
ARGS=("$@")
# ensure per-run output dir unless provided by user
case " ${ARGS[*]} " in *" --output_dir "* ) :;; *) ARGS=("${ARGS[@]}" "--output_dir" "runs/${run_id}");; esac
# keep W&B files under the run dir unless overridden
export WANDB_DIR="${run_dir}/wandb"
mkdir -p "$WANDB_DIR"
if [ -z "${WANDB_NAME:-}" ]; then export WANDB_NAME="run-${run_id}"; fi
set +o braceexpand; set -o noglob
nohup "$HOME/.local/bin/uv" run python train_baseline.py --train_urls "$TRAIN_URLS_POS" "${ARGS[@]}" > "$log" 2>&1 &
echo $! > "$run_dir/train.pid"
echo "Started PID $(cat "$run_dir/train.pid")"
echo "$run_dir/train.log"
'
  # Split EXTRA_ARGS into words locally and pass as bash -c positional args
  # shellcheck disable=SC2206
  EXTRA_ARR=( $EXTRA_ARGS )
  LOGFILE=$($SSH_CMD "${SSH_OPTS[@]}" "$HOST" bash -lc "$START_BG" bash "$TRAIN_URLS" "${EXTRA_ARR[@]}" | tail -n1)
  echo "Remote training started. Tail logs with:"
  echo "$SSH_CMD ${SSH_OPTS[*]} $HOST bash -lc 'cd $REMOTE_DIR_REMOTE; tail -f $LOGFILE'"
else
  echo "[7/7] Starting foreground training (attached)..."
START_FG='set -euo pipefail
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
cd '"$REMOTE_DIR_REMOTE"'
set +o allexport
if [ -f .env ]; then
  set -o allexport; . ./.env; set +o allexport
fi
TRAIN_URLS_POS="$1"; shift
ARGS=("$@")
# create run dir and set outputs
ts=$(date +%Y%m%d_%H%M%S)
rnd=${RANDOM}
run_id="${ts}_${rnd}"
run_dir="'$REMOTE_DIR_REMOTE'/runs/${run_id}"
mkdir -p "$run_dir"
ln -sfn "$run_dir" latest
# attach still writes a log alongside the console
log="$run_dir/train.log"
echo "Run ID: $run_id"
echo "Logging to $log (and console)"
# ensure per-run output dir unless provided by user
case " ${ARGS[*]} " in *" --output_dir "* ) :;; *) ARGS=("${ARGS[@]}" "--output_dir" "runs/${run_id}");; esac
# keep W&B files under the run dir unless overridden
export WANDB_DIR="${run_dir}/wandb"
mkdir -p "$WANDB_DIR"
if [ -z "${WANDB_NAME:-}" ]; then export WANDB_NAME="run-${run_id}"; fi
set +o braceexpand; set -o noglob
"$HOME/.local/bin/uv" run python train_baseline.py --train_urls "$TRAIN_URLS_POS" "${ARGS[@]}" 2>&1 | tee "$log"
'
  # shellcheck disable=SC2206
  EXTRA_ARR=( $EXTRA_ARGS )
  $SSH_CMD "${SSH_OPTS[@]}" "$HOST" bash -lc "$START_FG" bash "$TRAIN_URLS" "${EXTRA_ARR[@]}"
  echo "Remote training finished."
fi
