# Fully Pipelined GEMV Kernel — Version 1 (LUT Reduction)

This directory packages the compact, reproducible source release of the W4A4
64-by-64 GEMV kernel with a registered LUT-reduction network.

The compute core accepts one complete GEMV transaction per enabled cycle
(`II=1`) and has an eight-cycle black-box latency. The surrounding HLS DATAFLOW
kernel loads data and writes 576 independent results through AXI; its steady
transaction interval is limited by the 36 output beats.

## Directory Structure

```text
.
|-- src/                    # HLS kernel, C model, black-box JSON, and RTL
|-- config/                 # 200 MHz HLS and OOC clock configuration
|-- tests/                  # Functional and direct-core OOC test sources
|-- results/
|   |-- hls/                # gemv_kernel synthesis report
|   |-- simulation/         # Functional and II=1 PASS logs
|   |-- gemv_kernel/        # Final gemv_kernel utilization report
|   `-- resource_summary.csv
|-- env.sh                  # Portable tool-environment setup
|-- csynth.sh               # HLS synthesis entry point
|-- ooc_implement.sh        # Full-kernel OOC driver
|-- ooc_place_route.sh      # Direct-core place/route driver
|-- ooc_synth.tcl           # Vivado synth/opt/place/route flow
`-- ooc_slr_balance.tcl     # U55C SLR placement constraints
```

Only source, reproducible scripts, selected test inputs, and final evidence are
included. Vivado checkpoints, generated HLS projects, XSIM work libraries,
XO/XCLBIN files, journals, caches, and exploratory implementation runs are
intentionally omitted.

## Implementation

- `src/GEMV.cpp` and `src/GEMV.h` implement the 32-port HBM DATAFLOW kernel.
- `src/W4A4_P.v` implements the packed W4A4 multiplication primitive.
- `src/W4A4_P_wrapper.v` instantiates the 64-by-64-by-9 array and the fully
  registered six-level LUT/CARRY8 reduction.
- `src/W4A4_P.cpp` is the black-box C model.
- `src/W4A4_P.json` declares the relative C/RTL files, latency 8, and `II=1`.
- `config/hls.cfg` selects `gemv_kernel`, the testbench, and the 200 MHz target.

The main direct-core simulation checks consecutive GEMVs, all 576 results, and
clean stalling under backpressure. The generated-HLS test also checks three
consecutive nonuniform GEMVs with `ap_ready=1` on every accepted cycle.

## Requirements

- AMD/Xilinx Vitis HLS and Vivado 2023.2
- U55C part `xcu55c-fsvh2892-2L-e`
- A compatible U55C platform `.xpfm`
- Bash

Set the installation and platform paths before running:

```bash
export VITIS_SETTINGS=/path/to/Vitis/2023.2/settings64.sh
export VIVADO_SETTINGS=/path/to/Vivado/2023.2/settings64.sh
export PLATFORM=/path/to/xilinx_u55c_gen3x16_xdma_3_202210_1.xpfm
source ./env.sh
```

If the tools are already in `PATH`, `VITIS_SETTINGS` and `VIVADO_SETTINGS` may
be omitted.

## Reproduction

Run HLS synthesis:

```bash
./csynth.sh
```

Run full-kernel OOC synthesis and optimization:

```bash
OOC_STAGE=opt ./ooc_implement.sh
```

Run through placement and routing:

```bash
OOC_STAGE=full ./ooc_implement.sh
```

New outputs are created under `synth/` and `ooc_implement/`; these generated
directories are excluded from version control.

To reproduce the direct compute-core OOC result, first synthesize and optimize
the RTL, then place and route the generated checkpoint:

```bash
vivado -notrace -mode batch \
  -source tests/synth_pure_lut_array.tcl \
  -log tests/pure_lut_ooc.log \
  -journal tests/pure_lut_ooc.jou
./ooc_place_route.sh full
```

These commands write generated checkpoints and reports under `tests/results/`,
which is excluded from version control.

For the standalone compute-core functional test:

```bash
mkdir -p tests/.xsim_work
cd tests/.xsim_work
xvlog --sv ../../src/W4A4_P.v ../../src/W4A4_P_wrapper.v \
  ../test_overpacked_wrapper.v \
  "$XILINX_VIVADO/data/verilog/src/glbl.v"
xelab --debug off -L unisims_ver -L unimacro_ver -L secureip \
  test_overpacked_wrapper glbl -s w4a4_lut_reduction_sim
xsim w4a4_lut_reduction_sim -runall
```

The expected final marker is:

```text
PASS: full 64x64x9 array accepts consecutive GEMVs and stalls cleanly
```
