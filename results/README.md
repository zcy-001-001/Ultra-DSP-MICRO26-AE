# Result and Evidence Index

This is the only top-level result store in the artifact. Results are grouped by
paper item or supporting experiment, not by the machine on which they were
produced. Each result keeps its evidence class: measured, parsed from an
existing report, analytical, calibrated, paper-anchored, or literature-derived.

| Directory | Contents | Main paper mapping |
|---|---|---|
| `exactness/` | Full 49-pair arithmetic summaries | Table 2 support |
| `rtl/` | Vivado 2023.2 six-case functional summary | Table 2 support |
| `component_ablation/` | Component CSVs, figures, and Vivado reports | Table 2, Figure 17 |
| `area_ablation/` | Cross-precision CSVs, plots, and Vivado reports | Figure 17 |
| `overlap_depth_sweep/` | Constraint audits, self-checks, and 64x64 OOC reports | Figure 18 support |
| `table3_figure18/` | Table 3 summaries plus 42 Table 3 and 113 Figure 18 routed reports | Tables 3/4, Figure 18 |
| `table5_cpu_i7/` | i7 latency/power samples and summaries | Table 5 |
| `table5_cpu_xeon/` | Xeon archived measurements and selected-row summary | Table 5 |
| `table5_gpu/` | Formal INT4 GPU measurement and method note | Table 5 |
| `figure13/` | CPU/GPU inputs, analytical FPGA sweeps, power evidence, and combined plot data | Figure 13 |
| `figure12/` | Accuracy provenance, Pareto CSVs, and figures | Figure 12 |
| `figures14_15/` | Thirteen HLS reports, model/calibration CSVs, and regenerated plots | Figures 14/15 |
| `ilp_notebooks/` | Executed notebooks with embedded plot outputs | Figures 16/19/20 |
| `figure17/` | Parsed summaries and 56 Vivado synthesis reports | Figure 17 |
| `table6/` | Symmetric OSTQuant environment, checkpoint, training, log, summary, and tolerance evidence | Table 6 |
| `table7/` | Paper values, method summaries, and optional development logs | Table 7 |
| `table8/` | FP/MX layouts, correctness, Vivado/xsim reports, CSV summaries, and plots | Table 8 |
| `phase_adaptivity/` | Twelve archived reports, dual-hash manifest, and parsed summary | Phase-adaptivity ablation |
| `long_context/` | Long-context CSVs and plots | Figure 15 support |

[`RESULTS.md`](RESULTS.md) gives the concise numerical paper-item summary.
[`REPRODUCE.md`](../REPRODUCE.md) explains how to verify the canonical files
and how to place an optional rerun under `results/rerun/` without changing the
packaged evidence.
