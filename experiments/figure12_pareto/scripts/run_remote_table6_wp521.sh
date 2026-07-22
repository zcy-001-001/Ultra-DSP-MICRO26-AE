#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="${BASE_DIR:-<REMOTE_HOME>/codex_runs/ultradsp_wp521}"
REPO_DIR="${REPO_DIR:-${BASE_DIR}/OSTQuant_table6_clean}"
RUN_DIR="${RUN_DIR:-${BASE_DIR}/runs/table6_wp521_$(date +%Y%m%d_%H%M%S)}"

MODEL_LLAMA2="${MODEL_LLAMA2:-<MODEL_DIR>/llama2}"
MODEL_LLAMA3="${MODEL_LLAMA3:-<MODEL_DIR>/llama3}"
NPROC="${NPROC:-1}"
EVAL_NPROC="${EVAL_NPROC:-1}"

export CRYPTOGRAPHY_OPENSSL_NO_LEGACY="${CRYPTOGRAPHY_OPENSSL_NO_LEGACY:-1}"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base
# <REMOTE_HOST>'s miniconda base is kept as the parent runtime, while the existing
# OSTQuant venv carries lm_eval and the locally built fast_hadamard_transform.
OSTQUANT_VENV="${OSTQUANT_VENV:-<REMOTE_HOME>/rebuttal/ostquant_venv}"
if [ -d "$OSTQUANT_VENV" ]; then
  # shellcheck disable=SC1091
  source "$OSTQUANT_VENV/bin/activate"
fi

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export PATH="${CUDA_HOME}/bin:${PATH}"

export HF_HOME="${HF_HOME:-<REMOTE_HOME>/.cache/huggingface}"
# Downstream datasets may be absent from the remote cache; allow dataset fetches
# by default while keeping model weights fixed to the local /data-hdd paths.
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTHONUNBUFFERED=1

mkdir -p "$(dirname "$RUN_DIR")"
cd "$REPO_DIR"

echo "[wp521] start=$(date -Is)"
echo "[wp521] repo=$REPO_DIR"
echo "[wp521] run_dir=$RUN_DIR"
echo "[wp521] model_llama2=$MODEL_LLAMA2"
echo "[wp521] model_llama3=$MODEL_LLAMA3"

MODEL_LLAMA2="$MODEL_LLAMA2" \
MODEL_LLAMA3="$MODEL_LLAMA3" \
RUN_DIR="$RUN_DIR" \
NPROC="$NPROC" \
EVAL_NPROC="$EVAL_NPROC" \
bash experiments/table6_overpacking/scripts/run_table6_wp521_task_split.sh

# Previous full-regeneration entry point kept for auditability.  It reruns
# BF16/Ultra-DSP/DSP-Packing/DB-MixQ, whose Table 6 accuracy is already present
# in expected/table6_expected_from_existing_artifacts.json, so the rebuttal path
# above measures only the missing WP521 row.
# bash experiments/table6_overpacking/scripts/run_table6_wp521_only.sh
# bash experiments/table6_overpacking/scripts/run_table6_full_regeneration.sh

echo "[wp521] done=$(date -Is)"
echo "[wp521] summary=${RUN_DIR}/table6_wp521_summary.json"
