# Ultra-DSP Paper Results

This document summarizes the numerical results associated with the tables and
figures in **Ultra-DSP: A Universal Lossless DSP Overpacking Framework for
Low-Bit LLM Inference**. Detailed CSV, JSON, report, notebook, and figure files
are stored in the neighboring result directories.

## 📁 Result Map

| Paper item | Result directory | Contents |
|---|---|---|
| Table 2 | `component_ablation/`, `area_ablation/`, `ilp_notebooks/` | Component, resource, and layout analysis |
| Table 3 | `table3_figure18/` | OOC implementation summaries and routed reports |
| Table 4 | `table3_figure18/` | Ultra-DSP implementation data and cited UDP comparison values |
| Table 5 | `table5_cpu_i7/`, `table5_cpu_xeon/`, `table5_gpu/`, `figure13/` | CPU, GPU, and FPGA latency and power data |
| Table 6 | `table6/` | Llama-2-7B and Llama-3-8B accuracy results |
| Table 7 | `table7/` | Mixed-precision cross-model geomeans |
| Table 8 | `table8/` | FP/MX packing layouts and correction resources |
| Figure 12 | `figure12/` | Accuracy-throughput Pareto data and plots |
| Figure 13 | `figure13/` | Batch-size latency, power, and efficiency comparison |
| Figures 14 and 15 | `figures14_15/` | End-to-end speed and energy-efficiency data |
| Figure 16 | `ilp_notebooks/pareto.executed.ipynb` | Executed Pareto-layout notebook |
| Figure 17 | `figure17/` | Cross-precision LUT, FF, and DSP results |
| Figure 18 | `table3_figure18/`, `overlap_depth_sweep/` | PE-scaling implementation data |
| Figures 19 and 20 | `ilp_notebooks/` | Parallelism and computational-efficiency notebooks |

## 📊 Table 3: FPGA Implementation

The W4A4 rows are:

| Method | LUT | FF | DSP | Power (W) | Throughput (GOPS) |
|---|---:|---:|---:|---:|---:|
| WP521 | 12,049 | 28,587 | 4,096 | 4.573 | 6,553.6 |
| DB-MixQ | 99,271 | 205,650 | 4,096 | 4.895 | 9,830.4 |
| DSP-Packing | 106,736 | 54,448 | 4,096 | 4.979 | 9,830.4 |
| DuoQ | 98,337 | 158,930 | 4,096 | 5.445 | 6,553.6 |
| UDP | 148,831 | 80,010 | 4,096 | 5.084 | 9,830.4 |
| Ultra-DSP | 244,939 | 259,987 | 4,096 | 6.218 | 14,745.6 |

The complete fourteen-row table is available in
`table3_figure18/table3_ooc_summary.csv`. Throughput is computed as
`DSP count × packing count × 2 × frequency`.

## 📊 Table 5 and Figure 13: Platform Comparison

| Platform | Matrix shape | Latency (ms) | Power (W) |
|---|---:|---:|---:|
| Intel Xeon Gold 6544Y | 2048 × 2048 | 0.1290 | 212.709 |
| Intel Xeon Gold 6544Y | 4096 × 4096 | 0.3688 | 228.648 |
| NVIDIA RTX 6000 Ada | 2048 × 2048 | approximately 0.011 | See `table5_gpu/` |
| NVIDIA RTX 6000 Ada | 4096 × 4096 | approximately 0.019 | See `table5_gpu/` |
| Ultra-DSP FPGA W4A4 | 4096 × 4096, batch 1 | 0.018 | 45 |

The full batch sweep and plotting data are in `figure13/`. CPU and GPU result
files retain their measured samples, while the FPGA rows use the analytical
latency model described in the experiment scripts.

## 📊 Table 6: Model Accuracy

| Model | Method | Average accuracy |
|---|---|---:|
| Llama-2-7B | Baseline (BF16) | 68.47 |
| Llama-2-7B | Ultra-DSP | 64.67 |
| Llama-2-7B | DSP-Packing | 31.87 |
| Llama-2-7B | DB-MixQ | 32.76 |
| Llama-3-8B | Baseline (BF16) | 70.61 |
| Llama-3-8B | Ultra-DSP | 65.65 |
| Llama-3-8B | DSP-Packing | 33.41 |
| Llama-3-8B | DB-MixQ | 33.00 |

The corresponding summaries and training/evaluation records are under
`table6/`. The workflow uses symmetric OSTQuant, followed by the configured
weight-reconstruction stage.

## 📊 Table 7: Mixed-Precision Accuracy

| Format | Cross-model geomean |
|---|---:|
| W5A5 | 68.86 |
| W4A5 | 68.11 |
| W4A4 | 66.78 |
| W5A4 | 65.92 |
| W3A5 | 63.81 |
| W3A4 | 60.78 |
| W5A3 | 45.94 |
| W4A3 | 44.40 |

Supporting data and aggregation files are stored in `table7/`.

## 📈 Figures 12, 16, 19, and 20: Layout Search

Figure 12 combines normalized accuracy with the 4,096-DSP throughput model.
The accuracy inputs used by the Pareto generator are:

| Method | Llama-2-7B average | Relative accuracy (%) |
|---|---:|---:|
| WP521 | 31.80 | 48.21 |
| DB-MixQ | 32.85 | 49.80 |
| DSP-Packing | 31.97 | 48.47 |
| DuoQ | 65.96 | 100.00 |
| UDP | 65.96 | 100.00 |
| Ultra-DSP | 65.96 | 100.00 |

Figures 16, 19, and 20 are embedded in the executed notebooks under
`ilp_notebooks/`.

## 📈 Figures 14 and 15: End-to-End Evaluation

| Figure | Metric | Paper value |
|---|---|---:|
| Figure 14 | Geomean normalized speedup | 2.31× |
| Figure 14 | Geomean energy efficiency | 2.22× |
| Figure 15 | Geomean normalized speedup | 1.88× |
| Figure 15 | Geomean energy efficiency | 1.88× |

The plot data, generated figures, and HLS source reports are under
`figures14_15/`.

## 📈 Figure 17: Processing-Element Resources

The final W4A4 processing-element point uses 75 LUTs, 67 FFs, and one DSP. The
rounded LUT reduction across the evaluated precision pairs ranges from 29.2%
to 38.0%. The complete cross-precision data and generated plots are under
`figure17/`.

## 📈 Figure 18: PE Scaling

The selected PE counts are 256, 1,024, 2,304, 4,096, 6,400, and 8,100. The
4,096-PE implementation reaches an estimated 216.267 MHz. Utilization,
frequency, and report-derived summaries are stored under `table3_figure18/` and
`overlap_depth_sweep/`.

## 📚 External References

- Table 2 comparison values for TransFRU are taken from *TransFRU: Efficient
  Deployment of Transformers on FPGA with Full Resource Utilization*.
- Table 4 comparison values for UDP are taken from *UDP: A Universal DSP
  Packing Framework for Low-bitwidth MAC Acceleration on FPGAs*.
