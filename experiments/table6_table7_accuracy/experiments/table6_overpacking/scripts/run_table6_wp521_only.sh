#!/usr/bin/env bash
set -euo pipefail

# WP521-only runner for rebuttal use.  Existing Table 6/BF16/Ultra-DSP/
# DSP-Packing/DB-MixQ accuracy is reused from the checked-in expected artifact;
# this script only regenerates a W4A4 qmodel when missing and evaluates the
# WP521 raw-unsigned activation row that is absent from the paper.

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
ENV_DIR="${ENV_DIR:-<REMOTE_WORKSPACE>/codes/Quantization/INT}"
RUN_DIR="${RUN_DIR:-${REPO_DIR}/runs/table6_wp521_only_$(date +%Y%m%d_%H%M%S)}"
MODEL_LLAMA2="${MODEL_LLAMA2:-<REMOTE_WORKSPACE>/models/meta-llama/Llama-2-7b-hf}"
MODEL_LLAMA3="${MODEL_LLAMA3:-<REMOTE_WORKSPACE>/models/meta-llama/Llama-3-8B}"
NPROC="${NPROC:-1}"
EVAL_NPROC="${EVAL_NPROC:-1}"
MASTER_PORT="${MASTER_PORT:-8902}"
PYTHON_BIN="${PYTHON_BIN:-python}"
LM_EVAL_TASKS=(arc_easy hellaswag piqa openbookqa)

mkdir -p "$RUN_DIR"
cd "$REPO_DIR"

if [ -d "${ENV_DIR}/.venv" ]; then
  # shellcheck disable=SC1091
  source "${ENV_DIR}/.venv/bin/activate"
fi

export HF_HOME="${HF_HOME:-<REMOTE_WORKSPACE>/hf_cache}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTHONUNBUFFERED=1

run_torch_np() {
  local nproc="$1"
  shift
  "$PYTHON_BIN" -m torch.distributed.run --nnodes 1 --nproc_per_node "$nproc" --master-addr localhost --master-port "$MASTER_PORT" main.py "$@"
}

run_train_torch() {
  run_torch_np "$NPROC" "$@"
}

run_eval_torch() {
  run_torch_np "$EVAL_NPROC" "$@"
}

common_quant_args() {
  cat <<'ARGS'
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
--per_device_train_batch_size=4
--max_steps=100
--a_bits=4
--down_bits=4
--w_bits=4
--v_bits=4
--k_bits=4
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

train_w4a4_if_missing() {
  local model_name="$1" model_path="$2" out_dir="$3"
  mkdir -p "$out_dir"
  if [ -f "$out_dir/qmodel.pt" ] && [ -f "$out_dir/model.bin" ]; then
    echo "[skip] ${model_name} W4A4 qmodel; existing checkpoint found"
    return
  fi
  echo "[train] ${model_name} W4A4 qmodel -> ${out_dir}"
  mapfile -t qargs < <(common_quant_args)
  run_train_torch \
    --output_dir "$out_dir" --model "$model_path" \
    "${qargs[@]}" \
    --skip_ppl_eval=True \
    --save_qmodel_path "$out_dir/qmodel.pt"
}

eval_wp521() {
  local model_name="$1" model_path="$2" train_dir="$3" out_dir="$4"
  mkdir -p "$out_dir"
  if [ -f "$out_dir/log.txt" ] && grep -q "AVERAGE" "$out_dir/log.txt"; then
    echo "[skip] ${model_name} WP521; existing completed log found"
    return
  fi
  echo "[eval] ${model_name} wp521_unsigned_raw -> ${out_dir}"
  mapfile -t qargs < <(common_quant_args)
  run_eval_torch \
    --output_dir "$out_dir" --model "$model_path" \
    "${qargs[@]}" \
    --train_rotate=False --resume_path "$train_dir/model.bin" \
    --load_qmodel_path "$train_dir/qmodel.pt" \
    --linear_int_mode=approx \
    --linear_int_variant=wp521_unsigned_raw \
    --linear_int_fixed_alpha=1.0 \
    --linear_int_accum_dtype=bf16 \
    --skip_ppl_eval=True \
    --lm_eval=True --tasks "${LM_EVAL_TASKS[@]}" --lm_eval_batch_size 16
}

run_model_wp521() {
  local model_key="$1" model_path="$2"
  local base="$RUN_DIR/$model_key"
  local train_dir="$base/train_w4a4"
  train_w4a4_if_missing "$model_key" "$model_path" "$train_dir"
  eval_wp521 "$model_key" "$model_path" "$train_dir" "$base/wp521"
}

run_model_wp521 llama2_7b "$MODEL_LLAMA2"
run_model_wp521 llama3_8b "$MODEL_LLAMA3"

"$PYTHON_BIN" "$REPO_DIR/experiments/table6_overpacking/scripts/build_wp521_summary.py" \
  --run-dir "$RUN_DIR" \
  --out "$RUN_DIR/table6_wp521_summary.json" \
  --out-md "$RUN_DIR/table6_wp521_summary.md"
cp "$RUN_DIR/table6_wp521_summary.json" "$RUN_DIR/table6_summary.json"
echo "Done. Summary: $RUN_DIR/table6_wp521_summary.json"
