#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ae_root="$(cd "${script_dir}/.." && pwd)"
benchmark_dir="${ae_root}/experiments/baseline_kernels/NVIDIA-RTX-6000-Ada-Generation-GPU"
out_dir="${1:-${ae_root}/results/table5_gpu/rerun}"

mkdir -p "${out_dir}"
cd "${benchmark_dir}"
python benchmark_gemv.py \
  --gpu 0 \
  --dtype int4 \
  --verify \
  --warmup 100 \
  --measure-sec 10 \
  --streaming-mb 1024 \
  --out-dir "${out_dir}"
