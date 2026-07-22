#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ae_root="$(cd "${script_dir}/.." && pwd)"
tb_root="${ae_root}/src/rtl_testbench"
out_dir="${1:-${ae_root}/results/rtl/rerun}"
mkdir -p "${out_dir}"

if [[ -z "${XILINX_VIVADO:-}" ]]; then
  echo "XILINX_VIVADO is not set; source the Vivado settings script first" >&2
  exit 1
fi
glbl="${XILINX_VIVADO}/data/verilog/src/glbl.v"
if [[ ! -f "${glbl}" ]]; then
  echo "Vivado glbl.v not found under XILINX_VIVADO" >&2
  exit 1
fi

cases=(
  "W4A4 W4A4_P INT4_INT4_P_tb"
  "W4A4 W4A4_D INT4_INT4_D_tb"
  "W4A4 W4A4_Hybrid Hybrid_INT4_INT4_PD_tb"
  "W3A4 W3A4_P INT4_INT3_P_tb"
  "W3A4 W3A4_D INT4_INT3_D_tb"
  "W3A4 W3A4_Hybrid W3A4_PD_tb"
)

for entry in "${cases[@]}"; do
  read -r precision stem top <<<"${entry}"
  case_dir="${out_dir}/${stem}"
  mkdir -p "${case_dir}"
  pushd "${case_dir}" >/dev/null
  xvlog "${tb_root}/${precision}/${stem}.v" \
        "${tb_root}/${precision}/${stem}_tb.v" "${glbl}" --log xvlog.log
  xelab "${top}" glbl -L unisims_ver -timescale 1ns/1ps -s "sim_${stem}" --log xelab.log
  if grep -q "ERROR:" xelab.log; then
    echo "xelab reported an error for ${stem}" >&2
    exit 1
  fi
  xsim "sim_${stem}" -runall -log xsim.log
  grep -q "ALL TESTS PASSED" xsim.log
  echo "PASS ${stem}"
  popd >/dev/null
done

echo "RTL_SIM_PASS cases=${#cases[@]}"
