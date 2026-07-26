# Fully Pipelined GEMV Kernel — Version 2 (Hybrid Reduction)

This directory packages the compact, reproducible source release of the W4A4
64-by-64 GEMV kernel with a hybrid registered reduction network.

The compute core accepts one complete GEMV transaction per enabled cycle
(`II=1`) and has an eight-cycle black-box latency. The surrounding HLS DATAFLOW
kernel preserves 576 independent outputs; its full-kernel transaction interval
is 36 cycles because one 512-bit AXI output port must emit 36 beats.

The directory name retains the requested `Hybird-Reduction` spelling. The
architecture itself is referred to as hybrid reduction throughout the code and
documentation.

## Directory Structure

```text
.
|-- src/                       # HLS kernel and hybrid-reduction RTL
|-- config/                    # 200 MHz HLS and OOC clock configuration
|-- tests/                     # Pair, DSP-chain, array, and II=1 tests
|-- results/
|   |-- hls/                   # gemv_kernel synthesis report
|   |-- simulation/            # Functional and II=1 PASS logs
|   |-- gemv_kernel/           # Final gemv_kernel utilization report
|   `-- resource_summary.csv
|-- env.sh                     # Portable tool-environment setup
|-- csynth.sh                  # HLS synthesis entry point
|-- ooc_implement.sh           # Full-kernel OOC driver
|-- ooc_place_route_current_core.sh
|                              # Direct-core place/route driver
|-- ooc_synth.tcl              # Vivado synth/opt/place/route flow
`-- ooc_slr_balance.tcl        # U55C SLR placement constraints
```

Only source, reproducible scripts, selected test inputs, and final evidence are
included. Multi-gigabyte DCPs, generated HLS projects, XSIM work libraries,
XO/XCLBIN files, journals, caches, and superseded exploration runs are omitted.

## Implementation

- `src/GEMV.cpp` and `src/GEMV.h` implement the 32-port HBM DATAFLOW kernel.
- `src/W4A4_P.v` implements packed W4A4 multiplication.
- `src/W4A4_P_wrapper.v` contains the LUT6_2/CARRY8 pair adders, DSP FOUR12
  cascade, registered LUT tail, SLR-local activation decode, and backpressure
  control.
- `src/W4A4_P.cpp` is the black-box C model.
- `src/W4A4_P.json` declares relative source paths, latency 8, and `II=1`.
- `config/hls.cfg` selects the kernel/testbench and the 200 MHz target.

The test sources cover the pair adder, DSP-chain alignment, all 576 outputs,
consecutive transactions, stalls, and the generated HLS compute wrapper.

## Requirements

- AMD/Xilinx Vitis HLS and Vivado 2023.2
- U55C part `xcu55c-fsvh2892-2L-e`
- A compatible U55C platform `.xpfm`
- Bash

Configure the toolchain:

```bash
export VITIS_SETTINGS=/path/to/Vitis/2023.2/settings64.sh
export VIVADO_SETTINGS=/path/to/Vivado/2023.2/settings64.sh
export PLATFORM=/path/to/xilinx_u55c_gen3x16_xdma_3_202210_1.xpfm
source ./env.sh
```

If the tools are already in `PATH`, the two settings variables may be omitted.

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

New outputs are written under `synth/` and `ooc_implement/` and are excluded
from version control.

To reproduce the direct compute-core OOC result, first synthesize and optimize
the RTL, then place and route the generated checkpoint:

```bash
vivado -notrace -mode batch \
  -source tests/synth_simd_chain_array.tcl \
  -log tests/hybrid_core_ooc.log \
  -journal tests/hybrid_core_ooc.jou
./ooc_place_route_current_core.sh full
```

These commands write generated checkpoints and reports under `tests/results/`,
which is excluded from version control.

Run the exhaustive pair-level test:

```bash
mkdir -p tests/.pair_work
cd tests/.pair_work
xvlog --sv ../test_pair_add_variants.v \
  "$XILINX_VIVADO/data/verilog/src/glbl.v"
xelab --debug off -L unisims_ver -L unimacro_ver -L secureip \
  test_pair_add_variants glbl -s pair_lut6_2_sim
xsim pair_lut6_2_sim -runall
```

Run the full direct-core functional test:

```bash
mkdir -p tests/.array_work
cd tests/.array_work
xvlog --sv ../../src/W4A4_P.v ../../src/W4A4_P_wrapper.v \
  ../test_overpacked_wrapper.v \
  "$XILINX_VIVADO/data/verilog/src/glbl.v"
xelab --debug off -L unisims_ver -L unimacro_ver -L secureip \
  test_overpacked_wrapper glbl -s hybrid_array_sim
xsim hybrid_array_sim -runall
```

The expected array marker is:

```text
PASS: full 64x64x9 array accepts consecutive GEMVs and stalls cleanly
```
