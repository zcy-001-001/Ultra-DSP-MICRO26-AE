#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
build_dir="${1:-${script_dir}/build_2023_2}"
platform="${PLATFORM:-xilinx_u55c_gen3x16_xdma_3_202210_1}"

if ! command -v v++ >/dev/null 2>&1; then
    echo "ERROR: v++ is not on PATH. Source the Vitis 2023.2 settings script first." >&2
    exit 2
fi

version_text="$(v++ --version 2>&1 || true)"
if [[ "${version_text}" != *"2023.2"* ]]; then
    echo "ERROR: this reproduction entry requires Vitis 2023.2." >&2
    echo "Detected: ${version_text}" >&2
    exit 2
fi

mkdir -p "${build_dir}"
cd "${build_dir}"

v++ -c --mode hls --platform "${platform}" \
    --config "${script_dir}/llama2/hls_config1.cfg" \
    --work_dir ./initial_embedding_lookup

v++ -c --mode hls --platform "${platform}" \
    --config "${script_dir}/llama2/hls_config2.cfg" \
    --work_dir ./transformer_layer_pipeline

v++ -c --mode hls --platform "${platform}" \
    --config "${script_dir}/llama2/hls_config3.cfg" \
    --work_dir ./final_norm_classifier

echo "CSYNTH_COMPLETE build_dir=${build_dir}"
