# Table 3 GEMV Implementations

This directory contains the HLS, RTL black-box, configuration, and build inputs
for every implementation used to produce the 14 rows of Table 3. Generated
Vitis/Vivado projects and logs are excluded; the routed reports used by the
paper remain under `../../../results/table3_figure18/evidence/table3/`.

## Directory map

| Directory | Table 3 method and format | Main contents |
|---|---|---|
| `WP521/` | WP521, W4A4 | GEMV HLS wrapper, WP521 RTL/C model, black-box JSON, HLS/link configuration, and build scripts |
| `DeepBurning/` | DB-MixQ, W4A4 | GEMV HLS wrapper, DeepBurning/DB-MixQ RTL/C model, black-box JSON, HLS/link configuration, and build scripts |
| `DSP-Packing/` | DSP-Packing, W4A4 | GEMV HLS wrapper, FPL packing RTL/C model, two HLS configurations, and OOC/full-build scripts |
| `DuoQ/` | DuoQ, W4A4 | GEMV HLS wrapper, DuoQ RTL/C model, two HLS configurations, and OOC/full-build scripts |
| `UDP-general/INT4_INT4/` | UDP, W4A4 | UDP W4A4 GEMV and packed processing element |
| `UDP-general/INT5_INT5/` | UDP, W5A5 | UDP W5A5 GEMV and packed processing element |
| `UDP-general/INT3_INT4/` | UDP, W4A3 | UDP W3A4 source used for the W4A3 symmetric product row |
| `UDP-general/INT4_INT5/` | UDP, W4A5 | UDP W4A5 GEMV and packed processing element |
| `UDP-general/INT3_INT5/` | UDP, W5A3 | UDP W3A5 source used for the W5A3 symmetric product row |
| `Ultra-DSP-MAX/W4A4/` | Ultra-DSP, W4A4 | Maximum-packing W4A4 GEMV and processing element |
| `Ultra-DSP-MAX/W5A5/` | Ultra-DSP, W5A5 | Maximum-packing W5A5 GEMV and processing element |
| `Ultra-DSP-MAX/W3A4/` | Ultra-DSP, W4A3 | Maximum-packing W3A4 source used for the W4A3 symmetric product row |
| `Ultra-DSP-MAX/W4A5/` | Ultra-DSP, W4A5 | Maximum-packing W4A5 GEMV and processing element |
| `Ultra-DSP-MAX/W3A5/` | Ultra-DSP, W5A3 | Maximum-packing W3A5 source used for the W5A3 symmetric product row |

Each implementation directory contains:

- `src/GEMV.cpp` and `src/GEMV.h`: the HLS GEMV kernel and interface.
- `src/*.v`, `src/*.cpp`, and `src/*.json`: the RTL black box, C reference
  model, and Vitis HLS black-box metadata.
- `config/*.cfg`: the 200 MHz HLS configuration.
- `src/u55C.cfg`: the Alveo U55C link and HBM mapping configuration.
- `csynth.sh`, `ooc_implement*.sh`, and, where present,
  `hardware_implement.sh`: synthesis and implementation entry points.

The black-box JSON files use repository-relative source paths. Every local
`env.sh` delegates to the shared portable `env.sh` in this directory.

## Requirements

- AMD/Xilinx Vivado, Vitis, and Vitis HLS 2023.2.
- XRT when generating a U55C hardware image.
- A compatible Alveo U55C platform file.
- Bash and Python 3 for the orchestration and report parsing scripts.

Configure the toolchain once:

```bash
export XILINX_VITIS_SETTINGS=<VITIS_2023_2>/settings64.sh
export XILINX_VIVADO_SETTINGS=<VIVADO_2023_2>/settings64.sh
export XRT_SETUP=<XRT_INSTALL>/setup.sh
export PLATFORM=<U55C_PLATFORM>.xpfm
```

## Reproducing Table 3

Run commands from this directory. The following commands regenerate the six
W4A4 implementations:

```bash
bash WP521/ooc_implement.sh
bash DeepBurning/ooc_implement.sh
bash DSP-Packing/ooc_implement1.sh
bash DuoQ/ooc_implement1.sh
bash UDP-general/INT4_INT4/ooc_implement.sh
bash Ultra-DSP-MAX/W4A4/ooc_implement.sh
```

Regenerate the remaining UDP rows:

```bash
for precision in INT5_INT5 INT3_INT4 INT4_INT5 INT3_INT5; do
  bash "UDP-general/${precision}/ooc_implement.sh"
done
```

Regenerate all five Ultra-DSP rows:

```bash
bash Ultra-DSP-MAX/run_all_ooc.sh
```

Each OOC script writes a fresh 200 MHz Vitis/Vivado project under its own
implementation directory. To reconstruct the paper-facing CSV from a completed
report tree, use the parser documented in the parent
[`README.md`](../README.md). For a fast package-only check, re-parse the
packaged routed reports as described in the root
[`REPRODUCE.md`](../../../REPRODUCE.md).
