#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
RUN_DIR="${RUN_DIR:-${REPO_DIR}/runs/mixed_precision_ultradsp_$(date +%Y%m%d_%H%M%S)}"
MODEL_KEY="${MODEL_KEY:-${1:-}}"
PRECISION="${PRECISION:-${2:-}}"
MODEL_LLAMA2="${MODEL_LLAMA2:-<MODEL_DIR>/llama2}"
MODEL_LLAMA3="${MODEL_LLAMA3:-<MODEL_DIR>/llama3}"
NPROC="${NPROC:-1}"
MASTER_PORT="${MASTER_PORT:-8902}"
LM_EVAL_BATCH_SIZE="${LM_EVAL_BATCH_SIZE:-16}"
MAX_STEPS="${MAX_STEPS:-100}"
NSAMPLES="${NSAMPLES:-128}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
SKIP_EVAL="${SKIP_EVAL:-0}"
FORCE="${FORCE:-0}"

if [ -z "$MODEL_KEY" ] || [ -z "$PRECISION" ]; then
  echo "Usage: MODEL_KEY=llama2_7b PRECISION=W5A5 $0"
  exit 2
fi

if [[ "$PRECISION" =~ ^W([0-9]+)A([0-9]+)$ ]]; then
  W_BITS="${W_BITS:-${BASH_REMATCH[1]}}"
  A_BITS="${A_BITS:-${BASH_REMATCH[2]}}"
else
  echo "Invalid precision: $PRECISION"
  exit 2
fi

case "$MODEL_KEY" in
  llama2_7b) MODEL_PATH="${MODEL_PATH:-$MODEL_LLAMA2}" ;;
  llama3_8b) MODEL_PATH="${MODEL_PATH:-$MODEL_LLAMA3}" ;;
  *) echo "Unknown MODEL_KEY: $MODEL_KEY"; exit 2 ;;
esac

cd "$REPO_DIR"
mkdir -p "$RUN_DIR"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_SSL_VERIFICATION="${HF_HUB_DISABLE_SSL_VERIFICATION:-1}"
export CURL_CA_BUNDLE="${CURL_CA_BUNDLE:-}"
export REQUESTS_CA_BUNDLE="${REQUESTS_CA_BUNDLE:-}"
export PYTHONUNBUFFERED=1

run_torch_np() {
  local nproc="$1"
  shift
  python -m torch.distributed.run --nnodes 1 --nproc_per_node "$nproc" --master-addr localhost --master-port "$MASTER_PORT" main.py "$@"
}

common_quant_args() {
  cat <<ARGS
--loss_type=kl_top
--post_attn=True
--rotate_ov=True
--rotate_post_rope=False
--online_qk_hadamard=False
--smooth_qk=True
--smooth_ov=True
--smooth_up_down=True
--smooth_norm_linear=True
--bf16=True
--per_device_train_batch_size=${TRAIN_BATCH_SIZE}
--max_steps=${MAX_STEPS}
--nsamples=${NSAMPLES}
--a_bits=${A_BITS}
--down_bits=${A_BITS}
--w_bits=${W_BITS}
--v_bits=${A_BITS}
--k_bits=${A_BITS}
--a_asym
False
--k_asym
False
--v_asym
False
--narrow_symmetric=True
--train_enable_wquant=False
--sub_mean
False
--distribute=True
--use_klt
ARGS
}

task_args() {
  cat <<'ARGS'
--tasks
arc_easy
hellaswag
piqa
openbookqa
ARGS
}

train_qmodel() {
  local train_dir="$1"
  mkdir -p "$train_dir"
  if [ "$FORCE" != "1" ] && [ -f "$train_dir/qmodel.pt" ] && [ -f "$train_dir/model.bin" ]; then
    echo "[skip] existing qmodel: $train_dir"
    return
  fi
  mapfile -t qargs < <(common_quant_args)
  echo "[train] $MODEL_KEY $PRECISION -> $train_dir"
  run_torch_np "$NPROC" \
    --output_dir "$train_dir" --model "$MODEL_PATH" \
    "${qargs[@]}" \
    --skip_ppl_eval=True \
    --save_qmodel_path "$train_dir/qmodel.pt" \
    2>&1 | tee "$train_dir/console.log"
}

eval_ultradsp() {
  local train_dir="$1"
  local eval_dir="$2"
  mkdir -p "$eval_dir"
  if [ "$FORCE" != "1" ] && [ -f "$eval_dir/log.txt" ] && grep -q "AVERAGE" "$eval_dir/log.txt"; then
    echo "[skip] existing eval: $eval_dir"
    return
  fi
  mapfile -t qargs < <(common_quant_args)
  mapfile -t tasks < <(task_args)
  echo "[eval] $MODEL_KEY $PRECISION Ultra-DSP exact -> $eval_dir"
  run_torch_np 1 \
    --output_dir "$eval_dir" --model "$MODEL_PATH" \
    "${qargs[@]}" \
    --train_rotate=False --resume_path "$train_dir/model.bin" \
    --load_qmodel_path "$train_dir/qmodel.pt" \
    --linear_int_mode=exact \
    --linear_int_variant=lsb3_zero \
    --linear_int_fixed_alpha=1.0 \
    --linear_int_accum_dtype=bf16 \
    --skip_ppl_eval=True \
    --lm_eval=True --lm_eval_batch_size "$LM_EVAL_BATCH_SIZE" \
    "${tasks[@]}" \
    2>&1 | tee "$eval_dir/console.log"
}

JOB_DIR="$RUN_DIR/$MODEL_KEY/$PRECISION"
train_qmodel "$JOB_DIR/train"
if [ "$SKIP_EVAL" = "1" ]; then
  echo "[skip] eval disabled by SKIP_EVAL=1"
else
  # The original single-argument call only provided eval_dir, which leaves
  # train_dir unbound under `set -u` and stops the queue after qmodel export.
  # eval_ultradsp "$JOB_DIR/eval_ultradsp"
  eval_ultradsp "$JOB_DIR/train" "$JOB_DIR/eval_ultradsp"
fi
echo "[done] $MODEL_KEY $PRECISION"
