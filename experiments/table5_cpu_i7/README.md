# Table 5 local i7 reproduction

This folder contains the formal local Intel Core i7-13700 measurement used for
the Table 5 CPU rows.  It follows the reference CPU baseline method: Intel MKL
`gemm_s8u8s32`, resident weights, batch 1, a 10-second idle RAPL baseline, a
5-second warmup, a 30-second sustained measurement, and 1-second RAPL samples.

## Results

| Shape | Measured latency (ms) | Paper (ms) | Difference | Measured PKG (W) | Paper (W) | Difference | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| 1x2048x2048 | 0.1439 | 0.173 | -16.821% | 71.641 | 72 | -0.499% | PASS |
| 1x4096x4096 | 0.4575 | 0.574 | -20.296% | 71.809 | 75 | -4.255% | PASS |

Energy computed as raw package power times median latency is 10.309 mJ and
32.853 mJ, respectively.  These differ from the paper anchors by -17.116% and
-23.331%.  Per the AE acceptance policy, the results are considered a
successful reproduction because the implementation, shapes, timing windows,
and power scope match and the values are close to the paper.  Measured values
and paper anchors remain separate columns; paper values never replace raw
measurements.

The idle package mean in this run was 73.05 W, slightly above both loaded
means.  Therefore the clamped idle-subtracted package field is zero and is not
used for the paper-facing power column.  The canonical power is the raw mean of
`RAPL_Package0_PKG` inside each 30-second measurement window, matching the
reference script.  The Windows DRAM RAPL counter returned zero and is retained
as a platform limitation rather than imputed.

## Folder contents

| Path | Function |
|---|---|
| `scripts/measure_table5_i7.py` | Formal timing, RAPL sampling, comparison, and CSV/JSON writer. |
| `scripts/mkl_int8_backend.py` | Portable resident-weight MKL INT8 backend. |
| `scripts/rapl_monitor.py` | Windows multi-domain `typeperf` RAPL reader. |
| `../../results/table5_cpu_i7/i7_table5_summary.csv` | Compact measured/paper comparison. |
| `../../results/table5_cpu_i7/i7_table5_summary.json` | Method, environment, hashes, and per-domain statistics. |
| `../../results/table5_cpu_i7/i7_table5_latency_samples.csv` | Every timed GEMV iteration. |
| `../../results/table5_cpu_i7/i7_table5_power_samples.csv` | Timestamped PKG/PP0/DRAM readings and window flags. |

## Reproduction

Requirements: Windows, an Intel CPU exposing Windows Energy Meter counters,
Python 3.12 or compatible, NumPy, and Intel MKL.  Activate the intended Conda
environment.  If MKL cannot be discovered automatically, set `MKL_RT_PATH` to
the local `mkl_rt` library.

Run from the package root:

```powershell
python experiments/table5_cpu_i7/scripts/measure_table5_i7.py `
  --backend experiments/table5_cpu_i7/scripts/mkl_int8_backend.py `
  --rapl-monitor experiments/table5_cpu_i7/scripts/rapl_monitor.py `
  --out-dir results/rerun/table5_cpu_i7 `
  --idle-sec 10 --warmup-sec 5 --measure-sec 30 --sample-interval 1
```

Expected evidence volume is at least 50,000 latency iterations and 25 valid
package samples per shape.  Small numerical differences are expected from
thread scheduling, thermal state, memory state, and background activity.
