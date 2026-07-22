# Figure 17 Vivado 2023.2 Results

Figure 17 was reproduced with Vivado 2023.2 for `xcu55c-fsvh2892-2L-e`
using `AreaOptimized_high` and `control_set_opt_threshold=1`. This is a
single-PE out-of-context synthesis experiment, not a routed full system.

## W4A4 component stages

| Stage | Prefill/decode packing | LUT | FF | DSP |
|---|---:|---:|---:|---:|
| Normal signed | 4/2 | 11 | 0 | 1 |
| Sign-magnitude | 6/5 | 40 | 27 | 1 |
| Overpacking without correction | 8/6 | 49 | 33 | 1 |
| Lossless full correction | 8/6 | 82 | 90 | 1 |
| ILP/resource optimized Ultra-DSP | 9/7 | 75 | 67 | 1 |

The final point reproduces the paper's `75 LUT / 67 FF / 1 DSP` result.

## Cross-precision LUT reduction

| Format | Base LUT | Final LUT | Reduction |
|---|---:|---:|---:|
| W3A4 | 137 | 97 | 29.197% |
| W3A5 | 131 | 82 | 37.405% |
| W4A4 | 121 | 75 | 38.017% |
| W4A5 | 125 | 82 | 34.400% |
| W5A4 | 133 | 89 | 33.083% |
| W5A5 | 119 | 74 | 37.815% |

The `29.197%-38.017%` range reproduces the paper's rounded
`29.2%-38.0%` statement. Status: `PASS`.

## Evidence files

| Path | Function |
|---|---|
| `component_vivado2023_2_summary.csv` | Parsed W4A4 component-stage resources. |
| `area_vivado2023_2_summary.csv` | Parsed six-format, four-stage area sweep. |
| `figure17_summary.json` | Claim-oriented machine-readable summary. |
| `reports/component/` | Eight utilization/timing reports for four component stages. |
| `reports/area/` | Forty-eight utilization/timing reports for 24 area cases. |

Hostnames and private paths are redacted; tool, device, strategy, utilization,
and timing information is preserved.

## Optional rerun

Run from the repository root on a machine with Vivado 2023.2:

```bash
vivado -mode batch -source experiments/figure17/scripts/run_figure17_synth.tcl

python experiments/component_ablation/scripts/collect_vivado_utilization.py \
  --report-dir results/rerun/figure17/reports/component \
  --out results/rerun/figure17/component_rerun.csv

python experiments/area_ablation/scripts/collect_vivado_utilization.py \
  --report-dir results/rerun/figure17/reports/area \
  --out results/rerun/figure17/area_rerun.csv

python experiments/figure17/scripts/build_figure17_summary.py \
  --component results/rerun/figure17/component_rerun.csv \
  --area results/rerun/figure17/area_rerun.csv \
  --out-json results/rerun/figure17/figure17_rerun.json
```

Expected markers are `FIGURE17_VIVADO_SYNTH_PASS component=4 area=24` and
`FIGURE17_SUMMARY_PASS`.
