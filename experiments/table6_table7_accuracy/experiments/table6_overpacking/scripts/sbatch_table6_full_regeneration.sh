#!/usr/bin/env bash
#SBATCH -p acd_u
#SBATCH -J ostq-table6-full
#SBATCH -o <REMOTE_WORKSPACE>/codes/Quantization/INT/third_party/OSTQuant_table6_clean/logs/table6-full-%j.out
#SBATCH -e <REMOTE_WORKSPACE>/codes/Quantization/INT/third_party/OSTQuant_table6_clean/logs/table6-full-%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4

set -euo pipefail

REPO_DIR="${REPO_DIR:-<REMOTE_WORKSPACE>/codes/Quantization/INT/third_party/OSTQuant_table6_clean}"
ENV_DIR="${ENV_DIR:-<REMOTE_WORKSPACE>/codes/Quantization/INT}"
RUN_ROOT="${RUN_ROOT:-${REPO_DIR}/runs}"
RUN_DIR="${RUN_DIR:-${RUN_ROOT}/table6_full_regen_${SLURM_JOB_ID:-manual}}"
# Keep training single-rank; eval still uses model dispatch across the allocated GPUs.
NPROC="${NPROC:-1}"
EVAL_NPROC="${EVAL_NPROC:-1}"
MODEL_LLAMA2="${MODEL_LLAMA2:-<REMOTE_WORKSPACE>/models/meta-llama/Llama-2-7b-hf}"
MODEL_LLAMA3="${MODEL_LLAMA3:-<REMOTE_WORKSPACE>/models/meta-llama/Llama-3-8B}"
MASTER_PORT="${MASTER_PORT:-$((19000 + (${SLURM_JOB_ID:-0} % 20000)))}"

mkdir -p "${REPO_DIR}/logs" "${RUN_DIR}"
cd "${REPO_DIR}"

if ! command -v module >/dev/null 2>&1 && [ -f /etc/profile.d/modules.sh ]; then
  # shellcheck disable=SC1091
  . /etc/profile.d/modules.sh
fi
if command -v module >/dev/null 2>&1; then
  module purge
  module load gcc/13.3
  module load cuda/12.6
fi

export REPO_DIR ENV_DIR RUN_DIR NPROC EVAL_NPROC MODEL_LLAMA2 MODEL_LLAMA3 MASTER_PORT
export HF_HOME="${HF_HOME:-<REMOTE_WORKSPACE>/hf_cache}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTHONUNBUFFERED=1

echo "[table6] host=$(hostname) job=${SLURM_JOB_ID:-manual} start=$(date -Is)"
echo "[table6] repo=${REPO_DIR}"
echo "[table6] run_dir=${RUN_DIR}"
echo "[table6] train_nproc=${NPROC} eval_nproc=${EVAL_NPROC} master_port=${MASTER_PORT}"
echo "[table6] model_llama2=${MODEL_LLAMA2}"
echo "[table6] model_llama3=${MODEL_LLAMA3}"

"${REPO_DIR}/experiments/table6_overpacking/scripts/run_table6_full_regeneration.sh"

echo "[table6] done=$(date -Is) summary=${RUN_DIR}/table6_summary.json"
