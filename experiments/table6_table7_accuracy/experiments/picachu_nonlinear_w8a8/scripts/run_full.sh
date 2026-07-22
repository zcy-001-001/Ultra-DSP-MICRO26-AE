#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
RUN_DIR="${RUN_DIR:-${REPO_DIR}/runs/picachu_nonlinear_w8a8_$(date +%Y%m%d_%H%M%S)}"
PYTHON_BIN="${PYTHON_BIN:-<REMOTE_HOME>/miniconda3/envs/nonlinear/bin/python}"

MODEL_LLAMA2="${MODEL_LLAMA2:-<MODEL_DIR>/llama2}"
MODEL_LLAMA3="${MODEL_LLAMA3:-<MODEL_DIR>/llama3}"
MODEL_KEYS="${MODEL_KEYS:-llama2_7b llama3_8b}"
METHOD_KEYS="${METHOD_KEYS:-fp16 ibert gemmlowp}"

NONLINEAR_BITS="${NONLINEAR_BITS:-8}"
NONLINEAR_INT_PROFILE="${NONLINEAR_INT_PROFILE:-w8a8_mid}"
W_BITS="${W_BITS:-16}"
A_BITS="${A_BITS:-16}"
DOWN_BITS="${DOWN_BITS:-16}"
V_BITS="${V_BITS:-16}"
K_BITS="${K_BITS:-16}"
ACT_BITS="${ACT_BITS:-16}"
RESIDUAL_BITS="${RESIDUAL_BITS:-16}"
ATTN_BITS="${ATTN_BITS:-16}"
W_GPTQ="${W_GPTQ:-True}"
NSAMPLES="${NSAMPLES:-128}"
EVAL_SAMPLES="${EVAL_SAMPLES:--1}"
PPL_BSZ="${PPL_BSZ:-4}"
RUN_LM_EVAL="${RUN_LM_EVAL:-1}"
LM_EVAL_BATCH_SIZE="${LM_EVAL_BATCH_SIZE:-16}"
LM_EVAL_LIMIT="${LM_EVAL_LIMIT:-}"
PARSE_AFTER_RUN="${PARSE_AFTER_RUN:-1}"
AUTO_INSTALL_LM_EVAL="${AUTO_INSTALL_LM_EVAL:-0}"

LM_EVAL_TASKS=(arc_easy hellaswag piqa winogrande)

mkdir -p "$RUN_DIR"
cd "$REPO_DIR"

export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-<REMOTE_HOME>/.cache/huggingface}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED=1

if ! "$PYTHON_BIN" -c "import loguru, geoopt, einops, lm_eval, peft" >/dev/null 2>&1; then
  if [ "$AUTO_INSTALL_LM_EVAL" = "1" ]; then
    "$PYTHON_BIN" -m pip install loguru geoopt einops "lm_eval==0.4.4" "peft==0.4.0"
  else
    echo "One or more eval dependencies are missing in ${PYTHON_BIN}." >&2
    echo "Install once with: ${PYTHON_BIN} -m pip install loguru geoopt einops lm_eval==0.4.4 peft==0.4.0" >&2
    echo "Or rerun with AUTO_INSTALL_LM_EVAL=1 to install them automatically." >&2
    exit 2
  fi
fi

model_path_for() {
  case "$1" in
    llama2_7b) echo "$MODEL_LLAMA2" ;;
    llama3_8b) echo "$MODEL_LLAMA3" ;;
    *) echo "Unknown model key: $1" >&2; exit 2 ;;
  esac
}

method_for() {
  case "$1" in
    fp16) echo "disabled" ;;
    ibert) echo "ibert" ;;
    gemmlowp|gemmlowp_*) echo "gemmlowp" ;;
    *) echo "Unknown method key: $1" >&2; exit 2 ;;
  esac
}

profile_for() {
  case "$1" in
    gemmlowp_*) echo "${1#gemmlowp_}" ;;
    *) echo "$NONLINEAR_INT_PROFILE" ;;
  esac
}

completed_log() {
  local log_file="$1"
  if ! [ -f "$log_file" ] || ! grep -q "PPL:" "$log_file" || grep -Eiq "PPL:[[:space:]]*nan" "$log_file"; then
    return 1
  fi
  if [ "$RUN_LM_EVAL" = "1" ]; then
    grep -q "AVERAGE" "$log_file"
  else
    return 0
  fi
}

write_command() {
  local command_file="$1"
  shift
  printf "%q " "$@" > "$command_file"
  printf "\n" >> "$command_file"
}

run_one() {
  local model_key="$1"
  local method_key="$2"
  local model_path
  local method
  local profile
  model_path="$(model_path_for "$model_key")"
  method="$(method_for "$method_key")"
  profile="$(profile_for "$method_key")"

  local out_dir="${RUN_DIR}/${model_key}/${method_key}"
  local log_file="${out_dir}/log.txt"
  mkdir -p "$out_dir"
  if completed_log "$log_file"; then
    echo "[skip] ${model_key}/${method_key}; completed log found at ${log_file}"
    return
  fi

  local cmd=(
    "$PYTHON_BIN" "$REPO_DIR/main.py"
    --output_dir "$out_dir"
    --model "$model_path"
    --rotate=False
    --train_rotate=False
    --pre_eval=False
    --bf16=True
    --use_sdpa=False
    --distribute=True
    --eval_dataset wikitext2
    --eval_samples "$EVAL_SAMPLES"
    --bsz "$PPL_BSZ"
    --w_bits "$W_BITS"
    --a_bits "$A_BITS"
    --down_bits "$DOWN_BITS"
    --v_bits "$V_BITS"
    --k_bits "$K_BITS"
    --act_bits "$ACT_BITS"
    --residual_bits "$RESIDUAL_BITS"
    --attn_bits "$ATTN_BITS"
    --w_gptq="$W_GPTQ"
    --nsamples "$NSAMPLES"
    --skip_ppl_eval=False
    --nonlinear_int_method "$method"
    --nonlinear_bits "$NONLINEAR_BITS"
    --nonlinear_int_profile "$profile"
    --nonlinear_quant_rope=False
  )
  if [ "$RUN_LM_EVAL" = "1" ]; then
    cmd+=(
      --lm_eval=True
      --tasks "${LM_EVAL_TASKS[@]}"
      --lm_eval_batch_size "$LM_EVAL_BATCH_SIZE"
    )
    if [ -n "$LM_EVAL_LIMIT" ]; then
      cmd+=(--lm_eval_limit "$LM_EVAL_LIMIT")
    fi
  fi

  write_command "${out_dir}/command.txt" "${cmd[@]}"
  echo "[run] ${model_key}/${method_key} -> ${out_dir}"
  "${cmd[@]}" 2>&1 | tee "$log_file"
}

for model_key in $MODEL_KEYS; do
  for method_key in $METHOD_KEYS; do
    run_one "$model_key" "$method_key"
  done
done

if [ "$PARSE_AFTER_RUN" = "1" ]; then
  parse_cmd=(
    "$PYTHON_BIN" "$REPO_DIR/experiments/picachu_nonlinear_w8a8/scripts/parse_results.py"
    --run-dir "$RUN_DIR"
    --out "$RUN_DIR/picachu_nonlinear_w8a8_summary.json"
    --out-md "$RUN_DIR/picachu_nonlinear_w8a8_summary.md"
    --allow-missing
  )
  if [ "$RUN_LM_EVAL" != "1" ]; then
    parse_cmd+=(--allow-missing-metrics)
  fi
  "${parse_cmd[@]}"
fi

echo "Done. Run directory: $RUN_DIR"
