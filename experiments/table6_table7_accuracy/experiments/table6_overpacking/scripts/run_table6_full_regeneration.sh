#!/usr/bin/env bash
set -euo pipefail

# Full-regeneration Table 6 runner.  It does not consume prebuilt qmodel artifacts:
# each model first regenerates its W4A4 checkpoint, then evaluates all simulator rows.

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
ENV_DIR="${ENV_DIR:-<REMOTE_WORKSPACE>/codes/Quantization/INT}"
RUN_DIR="${RUN_DIR:-${REPO_DIR}/runs/table6_full_regen_$(date +%Y%m%d_%H%M%S)}"
MODEL_LLAMA2="${MODEL_LLAMA2:-<REMOTE_WORKSPACE>/models/meta-llama/Llama-2-7b-hf}"
MODEL_LLAMA3="${MODEL_LLAMA3:-<REMOTE_WORKSPACE>/models/meta-llama/Llama-3-8B}"
# Training defaults to one torchrun rank: DDP sees sharded/uneven trainable params here.
NPROC="${NPROC:-1}"
EVAL_NPROC="${EVAL_NPROC:-1}"
MASTER_PORT="${MASTER_PORT:-8902}"
SKIP_BF16="${SKIP_BF16:-0}"
MODEL_KEYS="${MODEL_KEYS:-llama2_7b llama3_8b}"
SKIP_PARSE="${SKIP_PARSE:-0}"
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
  # Use the active Python so venv-only packages (for example lm_eval and
  # fast_hadamard_transform on <REMOTE_HOST>) are visible inside torch distributed.
  "$PYTHON_BIN" -m torch.distributed.run --nnodes 1 --nproc_per_node "$nproc" --master-addr localhost --master-port "$MASTER_PORT" main.py "$@"
}

run_train_torch() {
  run_torch_np "$NPROC" "$@"
}

run_eval_torch() {
  # Evaluation uses one process and OSTQuant's model sharding.  Multiple
  # torchrun ranks each shard a full model copy and can OOM on H100.
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

run_bf16() {
  local model_name="$1" model_path="$2" out_dir="$3"
  if [ "$SKIP_BF16" = "1" ]; then
    echo "[skip] BF16 baseline for ${model_name}"
    return
  fi
  mkdir -p "$out_dir"
  if [ -f "$out_dir/log.txt" ] && grep -q "AVERAGE" "$out_dir/log.txt"; then
    echo "[skip] BF16 baseline for ${model_name}; existing completed log found"
    return
  fi
  echo "[bf16] ${model_name} -> ${out_dir}"
  # BF16 Table 6 needs downstream lm-eval accuracy only.  The legacy pre_eval
  # PPL helper expects older Llama internals (`model.decoder.layers`), so keep
  # it disabled for current transformers while still running lm_eval below.
  run_eval_torch \
    --output_dir "$out_dir" --model "$model_path" \
    --pre_eval=False --rotate=False --lm_eval=True \
    --tasks "${LM_EVAL_TASKS[@]}" \
    --skip_ppl_eval=True \
    --max_steps=200 --a_bits=16 --down_bits=16 --w_bits=16 --v_bits=16 --k_bits=16 \
    --bf16=True --distribute=True
}

train_w4a4() {
  local model_name="$1" model_path="$2" out_dir="$3"
  mkdir -p "$out_dir"
  if [ -f "$out_dir/qmodel.pt" ] && [ -f "$out_dir/model.bin" ]; then
    echo "[skip] ${model_name} W4A4 qmodel; existing checkpoint found"
    return
  fi
  echo "[train] ${model_name} W4A4 qmodel -> ${out_dir}"
  mapfile -t qargs < <(common_quant_args)
  # qmodel regeneration trains with normal fake-quant linear ops;
  # integer simulator modes are applied only during eval rows below.
  run_train_torch \
    --output_dir "$out_dir" --model "$model_path" \
    "${qargs[@]}" \
    --skip_ppl_eval=True \
    --save_qmodel_path "$out_dir/qmodel.pt"
}

eval_qmodel() {
  local label="$1" model_path="$2" train_dir="$3" out_dir="$4" mode="$5" variant="$6" fixed_alpha="$7" accum_dtype="$8" lm_bs="$9"
  mkdir -p "$out_dir"
  if [ -f "$out_dir/log.txt" ] && grep -q "AVERAGE" "$out_dir/log.txt"; then
    echo "[skip] ${label}; existing completed log found"
    return
  fi
  echo "[eval] ${label} -> ${out_dir}"
  mapfile -t qargs < <(common_quant_args)
  run_eval_torch \
    --output_dir "$out_dir" --model "$model_path" \
    "${qargs[@]}" \
    --train_rotate=False --resume_path "$train_dir/model.bin" \
    --load_qmodel_path "$train_dir/qmodel.pt" \
    --linear_int_mode="$mode" \
    --linear_int_variant="$variant" \
    --linear_int_fixed_alpha="$fixed_alpha" \
    --linear_int_accum_dtype="$accum_dtype" \
    --skip_ppl_eval=True \
    --lm_eval=True --tasks "${LM_EVAL_TASKS[@]}" --lm_eval_batch_size "$lm_bs"
}

run_model() {
  local model_key="$1" model_path="$2"
  local base="$RUN_DIR/$model_key"
  local train_dir="$base/train_w4a4"
  run_bf16 "$model_key" "$model_path" "$base/bf16"
  train_w4a4 "$model_key" "$model_path" "$train_dir"
  eval_qmodel "$model_key ultra_dsp" "$model_path" "$train_dir" "$base/ultra_dsp" exact lsb3_zero 1.0 bf16 16
  if [ "$model_key" = "llama2_7b" ]; then
    eval_qmodel "$model_key dsp_packing" "$model_path" "$train_dir" "$base/dsp_packing" approx lsb3_zero 1.0 fp16 2
  else
    eval_qmodel "$model_key dsp_packing" "$model_path" "$train_dir" "$base/dsp_packing" approx lsb3_zero 1.0 bf16 16
  fi
  eval_qmodel "$model_key db_mixq" "$model_path" "$train_dir" "$base/db_mixq" approx fixed_alpha 0.5 bf16 16
  # WP521 is a rebuttal/Figure 12 extension and is not part of paper Table 6.
  # Historical command retained for maintenance; use run_table6_wp521_only.sh
  # only when that separate extension is explicitly requested.
  # eval_qmodel "$model_key wp521" "$model_path" "$train_dir" "$base/wp521" approx wp521_unsigned_raw 1.0 bf16 16
}

# Historical all-model calls are kept here for maintenance reference. The
# MODEL_KEYS loop below allows two hosts to split the models safely.
# run_model llama2_7b "$MODEL_LLAMA2"
# run_model llama3_8b "$MODEL_LLAMA3"
for model_key in $MODEL_KEYS; do
  case "$model_key" in
    llama2_7b) run_model llama2_7b "$MODEL_LLAMA2" ;;
    llama3_8b) run_model llama3_8b "$MODEL_LLAMA3" ;;
    *) echo "Unknown MODEL_KEYS entry: $model_key" >&2; exit 2 ;;
  esac
done

if [ "$SKIP_PARSE" = "1" ]; then
  echo "Done. Parsing skipped; run parse_table6.py after all model hosts finish."
else
  "$PYTHON_BIN" "$REPO_DIR/experiments/table6_overpacking/scripts/parse_table6.py" \
    --run-dir "$RUN_DIR" \
    --out "$RUN_DIR/table6_summary.json" \
    --out-md "$RUN_DIR/table6_summary.md"
  echo "Done. Summary: $RUN_DIR/table6_summary.json"
fi
