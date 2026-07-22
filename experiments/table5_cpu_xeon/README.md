# Xeon Table 5 archived measurement

This directory preserves the historical Xeon measurement used by Table 5. No
new CPU rerun is required. The selected run uses Intel MKL
`gemm_s8u8s32`, a 1 GiB streaming weight bank, NUMA first-touch, a 3-second
warmup, a 10-second measurement window, and Linux RAPL.

## Folder functions

| Path | Function |
|---|---|
| `../../results/table5_cpu_xeon/int8_gemv_cpu_energy.csv` | Six-shape raw latency and two-socket RAPL summary from the archived run. |
| `../../results/table5_cpu_xeon/int8_gemv_cpu_energy.json` | JSON copy of the same raw records. |
| `../../results/table5_cpu_xeon/table5_xeon_summary.csv` | The two Table 5 shapes with package-0 and dual-socket scopes kept separate. |
| `scripts/benchmark_gemv_cpu.py` | Original benchmark entry point. |
| `scripts/summarize_table5_xeon.py` | Deterministically extracts and checks the two Table 5 rows. |
| `AUTHOR_REFERENCE_README.md` | Sanitized historical run notes. |

## Table 5 selection

The paper uses the single-socket `package0_w` field. The source host has two
CPU packages, so `power_total_w` and the raw `energy_mj` field are retained as
supplemental dual-socket measurements and are not substituted for the Table 5
energy scope.

| Shape | Latency | Package 0 | Dual-socket total | Package-0 energy | Paper anchor |
|---|---:|---:|---:|---:|---|
| 1x2048x2048 | 0.1290 ms | 212.709 W | 421.919 W | 27.439461 mJ | 0.129 ms / 213 W / 27.439 mJ |
| 1x4096x4096 | 0.3688 ms | 228.648 W | 459.080 W | 84.325382 mJ | 0.369 ms / 229 W / 84.325 mJ |

The archived measurements therefore reproduce both Table 5 Xeon rows after
the paper's displayed rounding. The resident-weight microbenchmark is a
different workload and is intentionally not used here.

## Reproduction

Re-extract the paper-facing rows from the archived report:

```bash
python experiments/table5_cpu_xeon/scripts/summarize_table5_xeon.py
```

Expected marker:

```text
XEON_TABLE5_PASS shapes=2 source=archived_streaming_weight_MKL_RAPL
```
