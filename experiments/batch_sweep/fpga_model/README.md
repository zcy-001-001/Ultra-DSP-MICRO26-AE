# FPGA GEMV Analytical Model (Table 5 and Figure 13)

This directory contains the AE recomputation of the FPGA points used in the
CPU/GPU/FPGA comparison. These FPGA rows are derived values, not measurements:

`evidence_class = ANALYTICAL_MODEL`

## Directory contents

| File | Function |
|---|---|
| `../../../results/figure13/fpga_gemv_batch_model.csv` | Auditable model inputs, intermediate timing terms, and final latency/throughput/energy results. |
| `../../../results/figure13/fpga_gemv_batch_model_8192dsp.csv` | The same model for 8192 DSPs, retaining the author-specified fixed 45 W FPGA power. |
| `../../../results/figure13/batch_comparison_canonical.csv` | CPU/GPU paper anchors and reruns plus both FPGA models, with provenance retained per row. |
| `README.md` | Model definition, paper mapping, reproduction steps, and interpretation limits. |

The implementation is `../../../scripts/compute_fpga_gemv_model.py`. Measured
CPU and GPU inputs remain under `../../../results/figure13/cpu/` and
`../../../results/figure13/gpu/`; they
are not silently merged into this analytical-only CSV.

## Fixed AE configuration

The default workload is `[B,4096] x [4096,4096]`, with
`B in {1,4,16,64,256}`. The model fixes:

| Parameter | Value |
|---|---:|
| Weight / activation format | W4A4 |
| External memory bandwidth | 460 GB/s (decimal, `460 * 10^9` bytes/s) |
| DSP count | 4096 |
| Packed MACs per DSP per cycle | 9 |
| Clock frequency | 200 MHz (5 ns per cycle) |
| Power | 45 W |

For each batch size:

```text
weight_bits = 4096 * 4096 * 4
memory_time = weight_bits / (460 * 10^9 * 8)
MACs(B) = B * 4096 * 4096
compute_time(B) = MACs(B) / (4096 * 9) * 5 ns
latency(B) = max(memory_time, compute_time(B))
throughput(B) = 2 * MACs(B) / latency(B)
energy(B) = 45 W * latency(B)
```

Memory and compute are assumed to be fully pipelined, so their times overlap
and the larger term determines total latency. One MAC is counted as two
operations when reporting TOPS.

## Paper mapping

| Paper item | CSV rows | Interpretation |
|---|---|---|
| Table 5 | `batch_size=1` | The FPGA W4A4 latency, power, throughput, and energy-efficiency point. |
| Figure 13 | all batch rows | FPGA batch-size trend using the same overlap model. |

Every row carries its mapping in `paper_mapping` and is labeled
`ANALYTICAL_MODEL` in `evidence_class`.

The companion baseline tree was audited at
`<REMOTE_WORKSPACE>/A-MICRO-CPU-GPU-Analysis`. Its GPU and CPU streaming
drivers use the same `K=N=4096` default and the same batch set
`1,4,16,64,256`. The paper-anchored CSVs in that tree contain one row for each
of those batch sizes. This audit establishes shape and batch alignment only;
it does not change the measured CPU/GPU evidence into analytical evidence.

## Reproduction

From the artifact root:

```bash
python scripts/compute_fpga_gemv_model.py --self-test
python scripts/compute_fpga_gemv_model.py --dsp-count 8192 --power-w 45 \
  --output results/rerun/figure13/fpga_gemv_batch_model_8192dsp.csv
python scripts/assemble_batch_comparison.py \
  --output results/rerun/figure13/batch_comparison_canonical.csv
```

Expected terminal markers:

```text
SELF_TEST=PASS
ROWS=5
```

The first command regenerates `results/figure13/fpga_gemv_batch_model.csv` by
default. The script uses only the Python standard library. To avoid overwriting
the canonical AE result, provide an explicit rerun path:

```bash
python scripts/compute_fpga_gemv_model.py \
  --batches 1,4,16,64,256 \
  --output results/rerun/figure13/comparison.csv
```

The canonical combined CSV contains 20 rows: five CPU, five GPU, five
4096-DSP FPGA, and five 8192-DSP FPGA rows. CPU/GPU rows are labeled
`MEASURED` or `PAPER_ANCHOR`; FPGA rows are labeled `ANALYTICAL_MODEL`.

## Audit note on bandwidth and rounding

With the author-confirmed decimal definition of 460 GB/s, the batch-1 memory
time is `0.01823610435 ms`; the batch-1 compute time is `0.00227555556 ms`, so
the reported latency is memory-bound. The paper reports this as `0.018 ms`
after rounding. The `0.182 ms` value in an informal note was a decimal-point
typo and is not used by the model.

The earlier 410 GB/s draft produced `0.02046001951 ms` and is retained only in
the version history. It was superseded by the author-confirmed 460 GB/s
Table 5/Figure 13 convention.

## Scope and limitations

- The 45 W value is fixed by the evaluation methodology; it is not measured by
  this script.
- The fixed 45 W input applies to both the 4096-DSP and 8192-DSP analytical
  arrays in the canonical AE outputs.
- The model assumes ideal overlap and does not include launch, control, NoC,
  activation-transfer, or output-write overhead.
- CPU and GPU rows are hardware measurements and have different evidence
  provenance. Any combined plot or table must preserve that distinction.
- The script exposes `--bandwidth-gbps` for sensitivity analysis, but the
  canonical Table 5/Figure 13 CSVs use 460 decimal GB/s.
