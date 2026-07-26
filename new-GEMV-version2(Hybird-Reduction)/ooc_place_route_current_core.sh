#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPT_DCP="${ROOT_DIR}/tests/results/simd_chain_lut6_2_pair_v4_ooc/W4A4_P_simd_chain_lut6_2_pair_opt.dcp"
OUTPUT_DIR="${ROOT_DIR}/tests/results/simd_chain_lut6_2_pair_v4_ooc_routed"
STAGE="${1:-full}"

source "${ROOT_DIR}/env.sh"
mkdir -p "${OUTPUT_DIR}"

vivado -notrace -mode batch \
    -log "${OUTPUT_DIR}/vivado.log" \
    -journal "${OUTPUT_DIR}/vivado.jou" \
    -source "${ROOT_DIR}/ooc_place_route_current_core.tcl" \
    -tclargs "${OPT_DCP}" "${OUTPUT_DIR}" "${STAGE}"
