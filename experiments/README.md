# Experiment Index

This directory contains experiment code, configurations, and input definitions.
Canonical outputs are never stored beside the code: every CSV, JSON, log,
report, executed notebook, and generated figure is filed under `results/`.
Commands below are run from the repository root unless noted otherwise.

| Directory | Paper mapping | Function | Canonical results | Reproduction entry point |
|---|---|---|---|---|
| `baseline_kernels/` | Table 5, Figure 13 | CPU/GPU reference kernels and measurement drivers | `results/table5_*`, `results/figure13/` | `scripts/run_gpu_int4_full.sh` or the per-platform README |
| `exactness/` | Table 2 support | Random signed-integer exactness sweep over 49 W/A pairs | `results/exactness/` | `python experiments/exactness/scripts/measure_absolute_precision.py --groups 100 --samples-per-group 10000 --seed 1305 --out-dir results/rerun/exactness` |
| `component_ablation/` | Table 2, Figure 17 | Packing-stage resource and correctness analysis | `results/component_ablation/`, `results/figure17/` | See `REPRODUCE.md`, Section 4 |
| `area_ablation/` | Figure 17 | Cross-precision Vivado resource sweep | `results/area_ablation/`, `results/figure17/` | See `REPRODUCE.md`, Section 4 |
| `overlap_depth_sweep/` | Figure 18 support | Overlap-depth constraints and the 64x64 OOC point | `results/overlap_depth_sweep/` | Run the audit/parser commands in its README |
| `batch_sweep/` | Table 5, Figure 13 | CPU/GPU batch data assembly and FPGA analytical model notes | `results/figure13/` | `python scripts/compute_fpga_gemv_model.py --self-test` |
| `figure12_pareto/` | Figure 12 | Accuracy-throughput Pareto construction | `results/figure12/` | `python experiments/figure12_pareto/scripts/generate_pareto.py --out-dir results/rerun/figure12` |
| `ilp_solver/` | Figures 16, 19, 20 | Source notebooks for layout, parallelism, and efficiency | `results/ilp_notebooks/` | Execute with `jupyter nbconvert` as shown in `REPRODUCE.md` |
| `figure17/` | Figure 17 | Vivado 2023.2 synthesis driver and summary builder | `results/figure17/` | `vivado -mode batch -source experiments/figure17/scripts/run_figure17_synth.tcl` |
| `table3_figure18/` | Tables 3, 4; Figure 18 | Complete Table 3 GEMV implementations plus parsers and tests for routed reports | `results/table3_figure18/` | Follow its README for full OOC regeneration, or run `python experiments/table3_figure18/scripts/verify_figure18_evidence.py` |
| `table5_cpu_i7/` | Table 5 | Local i7 MKL/RAPL measurement code | `results/table5_cpu_i7/` | See the directory README; reruns must target `results/rerun/table5_cpu_i7/` |
| `table5_cpu_xeon/` | Table 5 | Archived Xeon report parser | `results/table5_cpu_xeon/` | `python experiments/table5_cpu_xeon/scripts/summarize_table5_xeon.py` |
| `table6_table7_accuracy/` | Tables 6, 7 | Symmetric OSTQuant training/evaluation and aggregation | `results/table6/`, `results/table7/` | Follow the two experiment READMEs under its `experiments/` directory |
| `table8_fp_mx/` | Table 8 | FP/MX mantissa-magnitude packing backend | `results/table8/` | See `REPRODUCE.md`, Section 8 |
| `phase_adaptivity/` | Phase-adaptivity ablation | Parser and manifest verifier for P-only, D-only, and Hybrid reports | `results/phase_adaptivity/` | Run both scripts in the directory README |

The authoritative end-to-end commands and environment requirements are in
`REPRODUCE.md`. Optional reruns must use `results/rerun/<experiment>/` so they
do not overwrite the packaged evidence.
