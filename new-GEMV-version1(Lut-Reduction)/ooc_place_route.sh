#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPT_DCP="${ROOT_DIR}/tests/results/pure_lut_reduction_ooc/W4A4_P_pure_lut_opt.dcp"
OUTPUT_DIR="${ROOT_DIR}/tests/results/pure_lut_reduction_ooc_routed"
STAGE="${1:-full}"

source "${ROOT_DIR}/env.sh"
mkdir -p "${OUTPUT_DIR}"

vivado -notrace -mode batch \
    -log "${OUTPUT_DIR}/vivado.log" \
    -journal "${OUTPUT_DIR}/vivado.jou" \
    -source "${ROOT_DIR}/ooc_place_route_from_opt.tcl" \
    -tclargs "${OPT_DCP}" "${OUTPUT_DIR}" "${STAGE}"
