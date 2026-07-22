# Ultra-DSP Black-Box Artifact

This directory contains the 200 MHz Vitis HLS black-box artifact for the W4A4 Ultra-DSP processing element used in low-bit GEMV inference evaluation.

## Directory Layout

```text
.
|-- README.md
|-- env.sh
|-- csynth_200.sh
|-- hardware_implement_200.sh
|-- config/
|   `-- hls_200MHz.cfg
`-- src/
    |-- GEMV.cpp
    |-- GEMV.h
    |-- Hybrid_INT4_INT4_PD.cpp
    |-- Hybrid_INT4_INT4_PD.json
    |-- Hybrid_INT4_INT4_PD.v
    |-- Hybrid_INT4_INT4_PD_wrapper.v
    `-- u55C.cfg
```

## Files

`src/GEMV.cpp` implements the Vitis HLS GEMV kernel wrapper. It packs W4A4 activation and weight streams into the Ultra-DSP black-box interface, selects prefilling or decoding mode at runtime, and accumulates per-PE products into GEMV outputs.

`src/GEMV.h` defines the 64 x 64 DSP array shape, AXI port organization, tensor-buffer sizes, and the external kernel interface.

`src/Hybrid_INT4_INT4_PD.v` contains the single-DSP INT4 x INT4 packed processing element. It realizes the phase-adaptive W4A4 datapath used by the black-box wrapper.

`src/Hybrid_INT4_INT4_PD_wrapper.v` instantiates the single-DSP PE across the 64 x 64 array and exposes the ap_ctrl_chain-style RTL interface expected by Vitis HLS black-box integration.

`src/Hybrid_INT4_INT4_PD.cpp` is the C reference model for the black-box function. It is used by the HLS black-box metadata for functional correspondence.

`src/Hybrid_INT4_INT4_PD.json` maps the C black-box function ports to the RTL wrapper ports and records expected latency, initiation interval, and resource usage metadata.

`src/u55C.cfg` is the Vitis link configuration for the Alveo U55C target. It defines the kernel instance and HBM connectivity for activations, outputs, and 32 weight-column ports.

`config/hls_200MHz.cfg` is the only retained HLS configuration. It targets 200 MHz and packages `gemv_kernel` as an XO object with the RTL black box.

`env.sh` optionally sources Vitis, Vivado, and XRT setup scripts from `XILINX_VITIS_SETTINGS`, `XILINX_VIVADO_SETTINGS`, and `XRT_SETUP`. Set `PLATFORM` before synthesis or implementation.

`csynth_200.sh` runs 200 MHz HLS synthesis and writes the generated XO and logs under `synth_200/`.

`hardware_implement_200.sh` runs HLS synthesis if needed and links the generated XO into a U55C hardware xclbin under `implement_200/`.


## Usage

```bash
cd Black-Box
source ./env.sh
./csynth_200.sh
./hardware_implement_200.sh
```

Set the U55C platform file through `PLATFORM`:

```bash
PLATFORM=<u55c-platform.xpfm> ./csynth_200.sh
PLATFORM=<u55c-platform.xpfm> ./hardware_implement_200.sh
```

Generated directories such as `synth_200/` and `implement_200/` are intentionally excluded from the review artifact.
