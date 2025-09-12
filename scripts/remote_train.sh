#!/usr/bin/env bash
set -euo pipefail

# Remote SDXL+SigLIP training bootstrapper
# - Installs Astral's uv on the remote host if missing
# - Clones repo git@github.com:Wildebeest/sdxl-siglip.git (or custom URL)
# - Copies local .env to remote
# - Runs `uv sync`
# - Starts training with provided --train-urls

usage() {
  cat << 'USAGE'
Usage: scripts/remote_train.sh --host user@host --train-urls "wds_pattern" \
       [--remote-dir "~/sdxl-siglip"] [--repo "git@github.com:Wildebeest/sdxl-siglip.git"] \
       [--branch main] [--background] [--no-background] [--ssh "ssh -p 22"] \
       [--extra-args "--batch_size 1 --max_steps 1000 --wandb_project sdxl-siglip"]

Required:
  --host           SSH target, e.g. ubuntu@10.0.0.5
  --train-urls     WebDataset shard pattern, e.g. /data/shards/{0000..0999}.tar

Optional:
  --remote-dir     Directory on remote host (default: ~/sdxl-siglip)
  --repo           Git URL (default: git@github.com:Wildebeest/sdxl-siglip.git)
  --branch         Branch to checkout (default: main)
  --ssh            SSH command (default: ssh)
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
REPO_URL="git@github.com:Wildebeest/sdxl-siglip.git"
BRANCH="main"
SSH_CMD="ssh"
BACKGROUND=1
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST="$2"; shift 2;;
    --train-urls) TRAIN_URLS="$2"; shift 2;;
    --remote-dir) REMOTE_DIR="$2"; shift 2;;
    --repo) REPO_URL="$2"; shift 2;;
    --branch) BRANCH="$2"; shift 2;;
    --ssh) SSH_CMD="$2"; shift 2;;
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

if [[ ! -f .env ]]; then
  echo "Warning: local .env not found; continuing without copying secrets" >&2
fi

REMOTE_BOOTSTRAP='set -euo pipefail
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv on remote..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

mkdir -p "$REMOTE_DIR"
if [ ! -d "$REMOTE_DIR/.git" ]; then
  echo "Cloning repo..."
  git clone --branch "$BRANCH" "$REPO_URL" "$REMOTE_DIR"
else
  echo "Updating repo..."
  cd "$REMOTE_DIR"
  git fetch origin "$BRANCH"
  git checkout "$BRANCH"
  git pull --ff-only
fi

cd "$REMOTE_DIR"
echo "Syncing dependencies with uv..."
uv sync
'

echo "[1/5] Bootstrapping remote host and repo..."
$SSH_CMD "$HOST" REMOTE_DIR="$REMOTE_DIR" REPO_URL="$REPO_URL" BRANCH="$BRANCH" bash -lc "$REMOTE_BOOTSTRAP"

if [[ -f .env ]]; then
  echo "[2/5] Copying .env to remote..."
  scp .env "$HOST":"$REMOTE_DIR/.env"
fi

echo "[3/5] Verifying uv and repo on remote..."
$SSH_CMD "$HOST" bash -lc 'export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"; command -v uv && test -d '"$REMOTE_DIR"' && echo OK'

if [[ "$BACKGROUND" -eq 1 ]]; then
  echo "[4/5] Starting background training with nohup..."
START_BG='set -euo pipefail
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
cd '"$REMOTE_DIR"'
ts=$(date +%Y%m%d_%H%M%S)
log="train_${ts}.log"
echo "Logging to $log"
set +o allexport
if [ -f .env ]; then
  set -o allexport; . ./.env; set +o allexport
fi
nohup uv run python train_baseline.py --train_urls '"$TRAIN_URLS"' '"$EXTRA_ARGS"' > "$log" 2>&1 &
echo $! > train.pid
echo "Started PID $(cat train.pid)"
echo "$log"
'
  LOGFILE=$($SSH_CMD "$HOST" bash -lc "$START_BG")
  echo "[5/5] Remote training started. Tail logs with:"
  echo "$SSH_CMD $HOST bash -lc 'cd $REMOTE_DIR; tail -f $LOGFILE'"
else
  echo "[4/5] Starting foreground training (attached)..."
START_FG='set -euo pipefail
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
cd '"$REMOTE_DIR"'
set +o allexport
if [ -f .env ]; then
  set -o allexport; . ./.env; set +o allexport
fi
uv run python train_baseline.py --train_urls '"$TRAIN_URLS"' '"$EXTRA_ARGS"'
'
  $SSH_CMD "$HOST" bash -lc "$START_FG"
  echo "[5/5] Remote training finished."
fi
