#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  start_mixed_precision_tmux.sh \
    --session ultradsp_mp_l3_20260607 \
    --model-keys llama3_8b \
    --precisions W5A5,W4A3,W3A4,W4A5,W5A4,W3A5,W5A3 \
    --gpus 0,1,2,3 \
    --run-dir <REMOTE_HOME>/rebuttal/runs/mixed_precision_ultradsp_20260607 \
    --base-master-port 9900

The script writes the concrete command to RUN_DIR/launcher_logs/<session>.cmd.sh
and starts it in a detached tmux session. Comma-separated lists are used so SSH
launches do not need fragile quoted strings with spaces.
USAGE
}

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
SESSION=""
MODEL_KEYS=""
PRECISIONS="W5A5,W4A3,W3A4,W4A5,W5A4,W3A5,W5A3"
GPUS="0,1,2,3"
RUN_DIR="${RUN_DIR:-${REPO_DIR}/runs/mixed_precision_ultradsp_$(date +%Y%m%d_%H%M%S)}"
BASE_MASTER_PORT="9300"
START_DELAY_PER_GPU="60"
PATH_PREFIX="${PATH_PREFIX:-<REMOTE_HOME>/rebuttal/ostquant_venv/bin:<REMOTE_HOME>/miniconda3/bin:/usr/local/bin:/usr/bin:/bin}"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --session) SESSION="$2"; shift 2 ;;
    --model-keys) MODEL_KEYS="$2"; shift 2 ;;
    --precisions) PRECISIONS="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --run-dir) RUN_DIR="$2"; shift 2 ;;
    --base-master-port) BASE_MASTER_PORT="$2"; shift 2 ;;
    --start-delay-per-gpu) START_DELAY_PER_GPU="$2"; shift 2 ;;
    --repo-dir) REPO_DIR="$2"; shift 2 ;;
    --path-prefix) PATH_PREFIX="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [ -z "$SESSION" ] || [ -z "$MODEL_KEYS" ]; then
  echo "--session and --model-keys are required." >&2
  usage >&2
  exit 2
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 3
fi

MODEL_KEYS_SPACED="${MODEL_KEYS//,/ }"
PRECISIONS_SPACED="${PRECISIONS//,/ }"
GPUS_SPACED="${GPUS//,/ }"
CMD_DIR="$RUN_DIR/launcher_logs"
CMD_FILE="$CMD_DIR/${SESSION}.cmd.sh"
mkdir -p "$CMD_DIR"

{
  echo '#!/usr/bin/env bash'
  echo 'set -euo pipefail'
  printf 'cd %q\n' "$REPO_DIR"
  printf 'export PATH=%q\n' "$PATH_PREFIX"
  printf 'export MODEL_KEYS=%q\n' "$MODEL_KEYS_SPACED"
  printf 'export PRECISIONS=%q\n' "$PRECISIONS_SPACED"
  printf 'export GPUS=%q\n' "$GPUS_SPACED"
  printf 'export RUN_DIR=%q\n' "$RUN_DIR"
  printf 'export BASE_MASTER_PORT=%q\n' "$BASE_MASTER_PORT"
  printf 'export START_DELAY_PER_GPU=%q\n' "$START_DELAY_PER_GPU"
  echo 'export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"'
  echo 'export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"'
  echo 'export PYTHONUNBUFFERED=1'
  echo '/usr/bin/bash experiments/mixed_precision_ultradsp/scripts/run_mixed_precision_ultradsp_host.sh'
} > "$CMD_FILE"
chmod +x "$CMD_FILE"

tmux new-session -d -s "$SESSION" /usr/bin/bash "$CMD_FILE"
echo "[tmux started] session=$SESSION command=$CMD_FILE"
