# Ultra-DSP

[📊 Results](results/RESULTS.md) · [🧪 Experiments](experiments/README.md) · [🔁 Reproduction](REPRODUCE.md)

## 📖 Introduction

Ultra-DSP is a software-hardware co-design framework for low-bit large language
model inference on FPGAs. It improves DSP utilization by separating sign and
magnitude handling, packing multiple low-bit multiply-accumulate lanes into a
single DSP datapath, and applying exact correction logic to recover each output
lane.

The project contains three main technical components:

1. **Lossless DSP overpacking:** RTL modules for packed signed arithmetic across
   multiple weight and activation precisions.
2. **Layout and parallelism search:** analysis notebooks and scripts for packing
   layouts, Pareto frontiers, efficiency, and resource trade-offs.
3. **Phase-adaptive acceleration:** prefilling, decoding, and hybrid processing
   elements, together with an FPGA GEMV system and CPU/GPU baselines.

## 📁 Repository Structure

```text
.
|-- src/
|   |-- rtl/                     # Packed-DSP P, D, and Hybrid RTL
|   |-- rtl_testbench/           # W3A4/W4A4 simulation testbenches
|   `-- fpga_system/             # U55C HLS/RTL black-box GEMV system
|-- experiments/
|   |-- exactness/               # Arithmetic exactness experiments
|   |-- figure12_pareto/         # Accuracy-throughput Pareto generation
|   |-- batch_sweep/             # Batch-size comparison and FPGA model
|   |-- figures14_15/            # End-to-end performance simulator
|   |-- figure17/                # RTL resource synthesis workflow
|   |-- table3_figure18/         # Table 3 GEMV sources and OOC report analysis
|   |-- table6_table7_accuracy/  # OSTQuant accuracy workflows
|   `-- ...                      # Remaining tables, figures, and ablations
|-- results/
|   |-- README.md                # Result directory index
|   `-- RESULTS.md               # Paper tables and figures summary
|-- scripts/                     # Package utilities and result generators
|-- REPRODUCE.md                 # Complete reproduction commands
`-- README.md
```

### `src/`

Contains reusable RTL, matching testbenches, and the FPGA system implementation.
Precision-specific modules are organized by weight and activation bit width.

### `experiments/`

Contains the scripts, notebooks, configurations, and input definitions used for
the paper's tables, figures, and ablation studies. Each experiment directory has
its own README with the relevant commands and dependencies. The complete
WP521, DB-MixQ, DSP-Packing, DuoQ, UDP, and Ultra-DSP GEMV implementations for
Table 3 are under
[`experiments/table3_figure18/baseline_implementations/`](experiments/table3_figure18/baseline_implementations/).

### `results/`

Contains numerical summaries, CSV/JSON files, generated plots, executed
notebooks, Vivado/Vitis reports, and experiment logs. See
[`results/RESULTS.md`](results/RESULTS.md) for the paper-oriented summary and
[`results/README.md`](results/README.md) for the directory index.

### `scripts/`

Contains shared utilities for assembling figures, parsing reports, computing
the analytical FPGA model, and checking the package layout.

## 🚀 Getting Started

### 1. Prepare Python

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
pip install numpy scipy matplotlib jupyter nbconvert
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install numpy scipy matplotlib jupyter nbconvert
```

### 2. Inspect the packaged artifact

Run the package checker from the repository root:

```bash
python scripts/verify_artifact.py
```

### 3. Open the paper results

The concise numerical summary is available at
[`results/RESULTS.md`](results/RESULTS.md). Generated figures, reports, and raw
result files are grouped by paper item under `results/`.

## 🛠️ Usage

### RTL Simulation

Use Vivado 2023.2 or another compatible Verilog simulator. Add a module from
`src/rtl/` and the matching testbench from `src/rtl_testbench/`, then run a
behavioral simulation.

The packaged six-case Vivado workflow can be launched with:

```bash
bash scripts/run_rtl_sim.sh results/rerun/rtl
```

On Windows:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_rtl_sim.ps1 `
  -VivadoBin <VIVADO_BIN> `
  -OutDir results/rerun/rtl
```

### Analysis and Figures

Run an experiment from its own directory or use the commands collected in
[`REPRODUCE.md`](REPRODUCE.md). For example, regenerate the Figure 12 Pareto
data with:

```bash
python experiments/figure12_pareto/scripts/generate_pareto.py \
  --out-dir results/rerun/figure12
```

Execute the ILP notebooks with Jupyter:

```bash
jupyter nbconvert --to notebook --execute \
  experiments/ilp_solver/pareto.ipynb \
  --output results/rerun/ilp_notebooks/pareto.executed.ipynb
```

### Table 3 GEMV Implementations

The 14 Table 3 rows can be regenerated from the HLS kernels, packed-processing
element RTL, black-box metadata, configurations, and shell scripts in
[`experiments/table3_figure18/baseline_implementations/`](experiments/table3_figure18/baseline_implementations/).
The directory README maps each paper row to its implementation folder and gives
the 200 MHz OOC commands. A fast reproduction can instead re-parse the packaged
routed reports; both workflows are listed in
[`REPRODUCE.md`](REPRODUCE.md#5-tables-34-and-figure-18).

### FPGA System

The U55C HLS/RTL black-box GEMV implementation is under `src/fpga_system/`.
Vivado, Vitis, Vitis HLS, XRT, and a compatible platform file are required for
hardware generation. Environment and build details are documented in
[`src/fpga_system/README.md`](src/fpga_system/README.md).

### CPU/GPU and Accuracy Experiments

CPU/GPU baselines are under `experiments/baseline_kernels/`. Model-accuracy
workflows for Tables 6 and 7 are under
`experiments/table6_table7_accuracy/`. Run outputs should be written to a new
subdirectory under `results/rerun/` so that the packaged result files remain
unchanged.

For the complete table-by-table and figure-by-figure commands, see
[`REPRODUCE.md`](REPRODUCE.md).
