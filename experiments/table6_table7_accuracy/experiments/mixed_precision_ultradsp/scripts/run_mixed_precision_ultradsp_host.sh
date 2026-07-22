#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
RUN_DIR="${RUN_DIR:-${REPO_DIR}/runs/mixed_precision_ultradsp_$(date +%Y%m%d_%H%M%S)}"
MODEL_KEYS="${MODEL_KEYS:-llama2_7b llama3_8b}"
PRECISIONS="${PRECISIONS:-W5A5 W4A3 W3A4 W4A5 W5A4 W3A5 W5A3}"
GPUS="${GPUS:-0 1 2 3}"
BASE_MASTER_PORT="${BASE_MASTER_PORT:-9300}"
START_DELAY_PER_GPU="${START_DELAY_PER_GPU:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ONE_SCRIPT="$SCRIPT_DIR/run_one_precision_ultradsp.sh"
HOST_LOG_DIR="$RUN_DIR/host_logs/$(hostname)"
mkdir -p "$HOST_LOG_DIR"

read -r -a GPU_ARR <<< "$GPUS"
GPU_COUNT="${#GPU_ARR[@]}"
if [ "$GPU_COUNT" -lt 1 ]; then
  echo "No GPUs configured."
  exit 2
fi

run_queue_for_gpu() {
  local gpu_index="$1"
  local gpu_id="${GPU_ARR[$gpu_index]}"
  local job_index=0

  if [ "$START_DELAY_PER_GPU" -gt 0 ]; then
    local delay=$((gpu_index * START_DELAY_PER_GPU))
    echo "[delay] gpu=$gpu_id sleep=${delay}s"
    sleep "$delay"
  fi

  for model_key in $MODEL_KEYS; do
    for precision in $PRECISIONS; do
      if [ $((job_index % GPU_COUNT)) -eq "$gpu_index" ]; then
        local port=$((BASE_MASTER_PORT + job_index))
        echo "[queue] gpu=$gpu_id port=$port model=$model_key precision=$precision"
        if CUDA_VISIBLE_DEVICES="$gpu_id" \
          MASTER_PORT="$port" \
          MODEL_KEY="$model_key" \
          PRECISION="$precision" \
          RUN_DIR="$RUN_DIR" \
          "$ONE_SCRIPT"; then
          echo "[queue done] gpu=$gpu_id model=$model_key precision=$precision"
        else
          status=$?
          echo "[queue failed] gpu=$gpu_id model=$model_key precision=$precision exit=$status"
          return "$status"
        fi
      fi
      job_index=$((job_index + 1))
    done
  done
}

for gpu_index in "${!GPU_ARR[@]}"; do
  gpu_id="${GPU_ARR[$gpu_index]}"
  (
    run_queue_for_gpu "$gpu_index"
  ) > "$HOST_LOG_DIR/gpu_${gpu_id}.log" 2>&1 &
done

wait
echo "[host done] $(hostname) -> $RUN_DIR"
