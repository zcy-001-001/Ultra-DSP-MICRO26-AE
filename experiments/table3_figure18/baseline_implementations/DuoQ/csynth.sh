#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_SH="${ROOT_DIR}/env.sh"
PLATFORM="${PLATFORM:-/opt/xilinx/platforms/xilinx_u55c_gen3x16_xdma_3_202210_1/xilinx_u55c_gen3x16_xdma_3_202210_1.xpfm}"

if [[ -f "${ENV_SH}" ]]; then
    # shellcheck disable=SC1090
    source "${ENV_SH}"
fi

mkdir -p "${ROOT_DIR}/synth"

v++ -c --mode hls \
    --platform "${PLATFORM}" \
    --config "${ROOT_DIR}/config/hls1.cfg" \
    --work_dir "${ROOT_DIR}/synth"
