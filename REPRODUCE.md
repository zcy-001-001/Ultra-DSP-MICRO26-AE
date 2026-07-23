# Reproducing the Ultra-DSP Evaluation

This guide covers Tables 2-8, Figures 12-20, exactness, RTL, and
phase-adaptivity checks. Commands use repository-relative paths and configurable
environment placeholders.

## 1. Conventions and Environments

Run commands from the repository root unless a section says otherwise. Keep
optional reruns under `results/rerun/`; do not overwrite canonical files.

| Placeholder | Value |
|---|---|
| `<MODEL_DIR>` | Licensed Llama base model or checkpoint directory |
| `<RESULT_DIR>` | Table 6 run directory containing per-model and per-method logs |
| `<VIVADO_2023_2>` | Vivado/Vitis 2023.2 installation root |
| `<VIVADO_BIN>` | Vivado executable or executable directory |
| `<REPORT_ROOT>` | Existing external Vivado report tree |
| `<U55C_PLATFORM>` | Compatible U55C platform file |
| `<REMOTE_HOST>` | Linux compute host |

Fast checks require Python 3.10+. Analysis plots additionally need NumPy,
SciPy, Matplotlib, Jupyter, and nbconvert. Hardware paths require the tools
listed in [README.md](README.md).

## 2. Fast Package Verification

```bash
python scripts/verify_artifact.py
```

This validates existing reports and summaries only. It does not run Vivado,
GPU jobs, or OSTQuant training.

## 3. Exactness and RTL

### 3.1 49-pair exactness

```bash
python experiments/exactness/scripts/measure_absolute_precision.py \
  --groups 100 --samples-per-group 10000 --seed 1305 \
  --out-dir results/rerun/exactness
```

Pass criterion: 49 Ultra-DSP rows, with `ep=0`, `mse=0`, and `error_count=0`.
The canonical package is `results/exactness/`.

### 3.2 W3A4/W4A4 RTL

Linux:

```bash
source <VIVADO_2023_2>/settings64.sh
bash scripts/run_rtl_sim.sh results/rerun/rtl
```

Windows:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_rtl_sim.ps1 `
  -VivadoBin <VIVADO_BIN> `
  -OutDir results/rerun/rtl
```

Pass criterion: `RTL_SIM_PASS cases=6`. The archived summary is
`results/rtl/rtl_six_case_vivado2023_2.md`.

## 4. Table 2 and Figure 17

Run arithmetic/layout checks:

```bash
python experiments/component_ablation/scripts/check_v0_normal_signed.py
python experiments/overlap_depth_sweep/scripts/audit_overlap_constraints.py \
  --out results/rerun/overlap_constraint_audit.csv
```

Optional Vivado 2023.2 synthesis:

```bash
vivado -mode batch \
  -source experiments/figure17/scripts/run_figure17_synth.tcl

python experiments/component_ablation/scripts/collect_vivado_utilization.py \
  --report-dir results/figure17/rerun_reports/component \
  --out results/rerun/figure17_component.csv

python experiments/area_ablation/scripts/collect_vivado_utilization.py \
  --report-dir results/figure17/rerun_reports/area \
  --out results/rerun/figure17_area.csv

python experiments/figure17/scripts/build_figure17_summary.py \
  --component results/rerun/figure17_component.csv \
  --area results/rerun/figure17_area.csv \
  --out-json results/rerun/figure17_summary.json
```

Expected Figure 17 point: 75 LUT, 67 FF, one DSP; rounded LUT reduction range
29.2%-38.0%. Table 2 external rows are literature values; see
[results/RESULTS.md](results/RESULTS.md#external-references).

## 5. Tables 3/4 and Figure 18

### 5.1 Fast report-based reproduction

Re-parse the packaged existing reports:

```bash
python experiments/table3_figure18/scripts/extract_ooc_reports.py \
  --source-root results/table3_figure18/evidence/table3 \
  --table3-only \
  --output-dir results/rerun/table3

python experiments/table3_figure18/scripts/verify_figure18_evidence.py
python -m unittest discover \
  -s experiments/table3_figure18/tests -v
```

Expected markers include `TABLE3_EVIDENCE_PASS` and
`FIGURE18_EVIDENCE_PASS timing=106 selected_utilization=7`.

### 5.2 Full Table 3 implementation regeneration

The complete GEMV implementation inputs for all 14 Table 3 rows are under
`experiments/table3_figure18/baseline_implementations/`. They include the HLS
GEMV kernels, packed-processing-element RTL, C black-box models, JSON metadata,
HLS and U55C configurations, and synthesis/implementation scripts.

Use AMD/Xilinx Vivado, Vitis, and Vitis HLS 2023.2. Configure the toolchain and
U55C platform:

```bash
export XILINX_VITIS_SETTINGS=<VITIS_2023_2>/settings64.sh
export XILINX_VIVADO_SETTINGS=<VIVADO_2023_2>/settings64.sh
export XRT_SETUP=<XRT_INSTALL>/setup.sh
export PLATFORM=<U55C_PLATFORM>.xpfm
BASE=experiments/table3_figure18/baseline_implementations
```

Regenerate the six W4A4 rows:

```bash
bash "$BASE/WP521/ooc_implement.sh"
bash "$BASE/DeepBurning/ooc_implement.sh"
bash "$BASE/DSP-Packing/ooc_implement1.sh"
bash "$BASE/DuoQ/ooc_implement1.sh"
bash "$BASE/UDP-general/INT4_INT4/ooc_implement.sh"
bash "$BASE/Ultra-DSP-MAX/W4A4/ooc_implement.sh"
```

Regenerate the four additional UDP rows and all five Ultra-DSP precision rows:

```bash
for precision in INT5_INT5 INT3_INT4 INT4_INT5 INT3_INT5; do
  bash "$BASE/UDP-general/${precision}/ooc_implement.sh"
done

bash "$BASE/Ultra-DSP-MAX/run_all_ooc.sh"
```

The exact paper-row mapping is documented in
`experiments/table3_figure18/baseline_implementations/README.md`. After the OOC
runs finish, parse the new report tree without changing canonical outputs:

```bash
python experiments/table3_figure18/scripts/extract_ooc_reports.py \
  --source-root "$BASE" \
  --output-dir results/rerun/table3_figure18
```

The resulting Table 3 CSV contains the 14 resource, power, throughput,
energy-efficiency, and timing rows. Throughput is computed from DSP count,
packing count, two operations per MAC, and the 200 MHz frequency.

Re-parse the packaged 64x64 depth-3 OOC point without launching Vivado:

```bash
python experiments/overlap_depth_sweep/scripts/extract_depth3_ooc.py
```

Expected marker: `DEPTH3_OOC_PASS PE=4096 Fmax_MHz=216.267 DSP=4096`.

For a reviewer-owned external report tree:

```bash
python experiments/table3_figure18/scripts/extract_ooc_reports.py \
  --source-root <REPORT_ROOT> \
  --output-dir results/rerun/table3_figure18
```

Table 4 UDP comparison values are literature values from the UDP paper; only
the Ultra-DSP row is recomputed from packaged evidence.

## 6. Table 5 and Figure 13

### 6.1 FPGA analytical model

```bash
python scripts/compute_fpga_gemv_model.py \
  --self-test \
  --output results/rerun/figure13/fpga_gemv_batch_model.csv

python scripts/compute_fpga_gemv_model.py \
  --dsp-count 8192 \
  --output results/rerun/figure13/fpga_gemv_batch_model_8192dsp.csv
```

The model uses 460 decimal GB/s, fixed 45 W, 200 MHz, nine packed MACs per DSP
per cycle, and `latency=max(memory_time, compute_time)`. Batch-1 latency is
0.018236 ms.

Assemble measured and analytical rows:

```bash
python scripts/assemble_batch_comparison.py \
  --output results/rerun/figure13/batch_comparison_canonical.csv
```

### 6.2 Xeon archived report parsing

```bash
python experiments/table5_cpu_xeon/scripts/summarize_table5_xeon.py
```

Expected marker: `XEON_TABLE5_PASS`.

### 6.3 i7 and GPU optional reruns

The exact i7 command and required RAPL monitor are documented in
`experiments/table5_cpu_i7/README.md`. Write its `--out-dir` to
`results/rerun/table5_cpu_i7`.

For the GPU formal measurement:

```bash
bash scripts/run_gpu_int4_full.sh results/rerun/table5_gpu
```

The GPU command writes latency, power samples, and summary files to the selected
output directory.

## 7. Tables 6 and 7, and Figure 12

Licensed model weights and evaluation datasets are not included. The package
uses narrow symmetric OSTQuant. Training learns rotation and smoothing
parameters; GPTQ is a later weight reconstruction step and does not replace
OSTQuant.

### 7.1 Table 6

Read the complete environment, training, evaluation, and staging procedure in
`experiments/table6_table7_accuracy/experiments/table6_overpacking/README.md`.
The formal row set contains BF16, Ultra-DSP, DSP-Packing, and DB-MixQ for two
models. WP521 is excluded from Table 6.

After an optional run, parse and compare without overwriting canonical files:

```bash
python experiments/table6_table7_accuracy/experiments/table6_overpacking/scripts/parse_table6.py \
  --run-dir <RESULT_DIR> \
  --out results/rerun/table6/table6_fresh_summary.json \
  --out-md results/rerun/table6/table6_fresh_summary.md

python experiments/table6_table7_accuracy/experiments/table6_overpacking/scripts/compare_table6.py \
  --fresh results/rerun/table6/table6_fresh_summary.json \
  --expected experiments/table6_table7_accuracy/experiments/table6_overpacking/expected/table6_expected_from_existing_artifacts.json \
  --out results/rerun/table6/table6_tolerance.json \
  --out-md results/rerun/table6/table6_tolerance.md
```

### 7.2 Table 7

The formal package records paper anchors and method instructions, not a fresh
formal run. Follow
`experiments/table6_table7_accuracy/experiments/mixed_precision_ultradsp/README.md`
for W3-W5 and A3-A5 runs.

Aggregate an optional rerun with:

```bash
python experiments/table6_table7_accuracy/experiments/mixed_precision_ultradsp/scripts/build_table7_geomean.py \
  --out-json results/rerun/table7/table7_optional_recomputed_summary.json \
  --out-csv results/rerun/table7/table7_optional_recomputed_summary.csv \
  --out-md results/rerun/table7/table7_optional_recomputed_table.md
```

### 7.3 Figure 12

```bash
python experiments/figure12_pareto/scripts/generate_pareto.py \
  --out-dir results/rerun/figure12
```

Accuracy provenance is frozen in `results/figure12/accuracy_provenance.csv`;
throughput uses the Table 3 equation and a 4096-DSP budget model.

## 8. Table 8

```bash
python experiments/table8_fp_mx/scripts/fp_layout_solver.py \
  --out results/rerun/table8/fp_layouts.csv

python experiments/table8_fp_mx/scripts/audit_correction_luts.py \
  --layouts results/table8/fp_layouts.csv \
  --report-dir results/table8/vivado_resource \
  --out results/rerun/table8/fp_correction_lut_audit.csv
```

The declared AE scope is FP/MX mantissa-magnitude packing and correction.

## 9. Figures 14 and 15

Rebuild only the simulator outputs from the packaged csynth reports:

```bash
python experiments/figures14_15/scripts/simulate_figures.py \
  --report-dir results/figures14_15/evidence/raw_hls_reports \
  --result-dir results/rerun/figures14_15

python experiments/figures14_15/tests/test_simulator.py
```

Expected geomeans are 2.31x/2.22x for Figure 14 speed/energy efficiency and
1.88x/1.88x for Figure 15. Raw model and calibrated fields remain separate.

## 10. Figure 16, Figure 19, and Figure 20

```bash
mkdir -p results/rerun/ilp_notebooks
jupyter nbconvert --to notebook --execute \
  experiments/ilp_solver/pareto.ipynb \
  --output results/rerun/ilp_notebooks/pareto.executed.ipynb
jupyter nbconvert --to notebook --execute \
  experiments/ilp_solver/parallelism.ipynb \
  --output results/rerun/ilp_notebooks/parallelism.executed.ipynb
jupyter nbconvert --to notebook --execute \
  experiments/ilp_solver/efficiency.ipynb \
  --output results/rerun/ilp_notebooks/efficiency.executed.ipynb
```

Canonical executed notebooks are in `results/ilp_notebooks/`.

## 11. Phase Adaptivity

Parse and verify the 12 existing reports:

```bash
python experiments/phase_adaptivity/scripts/extract_phase_adaptivity.py
python experiments/phase_adaptivity/scripts/verify_report_manifest.py
```

Expected markers are `PHASE_ADAPTIVITY_PASS` and
`PHASE_REPORT_MANIFEST_PASS reports=12`. P-only and D-only meet 200 MHz. The
Hybrid functional pass and its full-design timing miss are both intentionally
reported.

## 12. Final Package Check

```bash
python scripts/verify_artifact.py
```

Run the artifact checker once more from the repository root after updating
code, result files, or documentation.
