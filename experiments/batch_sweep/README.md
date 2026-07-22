# Table 5 and Figure 13 Batch Comparison

This directory contains the CPU/GPU batch-sweep drivers and the documentation
for the FPGA analytical model. All packaged outputs are under
`results/figure13/`; this experiment directory contains code and method only.

## Evidence policy

- CPU and GPU rows are measured inputs or explicit paper anchors. Their raw
  provenance is retained in `results/figure13/cpu/` and
  `results/figure13/gpu/`.
- FPGA rows are `ANALYTICAL_MODEL`, not board measurements.
- The canonical FPGA model fixes 460 decimal GB/s, 45 W, 200 MHz, 4096 or 8192
  DSPs, and nine packed MACs per DSP per cycle.
- The 45 W value is the evaluation convention, rounded from the archived
  44.780 W full-design Vivado estimate in `results/figure13/evidence/`.

For `[B,4096] x [4096,4096]` W4A4 GEMV:

```text
weight_read_time = 4096 * 4096 * 4 / (460e9 * 8)
compute_time(B) = B * 4096 * 4096 / (DSPs * 9) / 200e6
latency(B) = max(weight_read_time, compute_time(B))
throughput(B) = 2 * B * 4096 * 4096 / latency(B)
energy(B) = 45 W * latency(B)
```

Memory and compute are assumed fully pipelined. The batch-1 latency is
`0.018236 ms`, reported as `0.018 ms` after paper rounding.

## Directory functions

| Path | Function |
|---|---|
| `benchmark_batch_streaming_gpu.py` | Streaming-weight GPU batch sweep. |
| `benchmark_batch_streaming_cpu.py` | MKL/RAPL CPU batch sweep. |
| `plot_gpu_fpga_batch.py` | GPU and FPGA comparison plot. |
| `plot_gpu_cpu_fpga_batch.py` | CPU, GPU, and FPGA comparison plot. |
| `fpga_model/README.md` | Auditable analytical-model definition. |

## Reproduction

Regenerate the analytical rows and combined table from the repository root:

```bash
python scripts/compute_fpga_gemv_model.py --self-test \
  --output results/rerun/figure13/fpga_gemv_batch_model.csv
python scripts/compute_fpga_gemv_model.py --dsp-count 8192 \
  --output results/rerun/figure13/fpga_gemv_batch_model_8192dsp.csv
python scripts/assemble_batch_comparison.py \
  --output results/rerun/figure13/batch_comparison_canonical.csv
```

Optional CPU/GPU reruns require the environments described in
`experiments/baseline_kernels/README.md`. Write their outputs beneath
`results/rerun/figure13/`; do not overwrite the canonical evidence.

The canonical combined CSV has 20 rows: five CPU, five GPU, five 4096-DSP FPGA,
and five 8192-DSP FPGA rows. Its `evidence_class` column keeps measured, paper
anchor, and analytical rows distinct.
