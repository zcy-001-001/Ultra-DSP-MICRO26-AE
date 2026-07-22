# Cross-Precision Area Ablation

This experiment provides the Figure 17 cross-precision resource sweep for six
W/A formats. It contains the RTL manifest, generated RTL, Vivado batch drivers,
report parser, and plotting code. Canonical reports and figures are under
`results/area_ablation/`; the paper-facing Figure 17 summary is under
`results/figure17/`.

## Directory functions

| Path | Function |
|---|---|
| `manifest.csv` | Input design matrix. |
| `rtl/optimization_stages/` | Four optimization stages for each precision. |
| `rtl/lsb_depth_sweep/` | LSB correction-depth variants. |
| `scripts/generate_area_ablation.py` | Regenerates RTL definitions. |
| `scripts/synth_all.tcl` | Vivado batch synthesis driver. |
| `scripts/collect_vivado_utilization.py` | Parses LUT/FF/DSP reports. |
| `scripts/plot_area_ablation.py` | Generates summary plots. |

## Reproduction

From the repository root:

```bash
python experiments/area_ablation/scripts/generate_area_ablation.py
vivado -mode batch -source experiments/area_ablation/scripts/synth_all.tcl
python experiments/area_ablation/scripts/collect_vivado_utilization.py \
  --report-dir results/rerun/area_ablation/vivado_resource \
  --out results/rerun/area_ablation/vivado_resource_summary.csv
```

The canonical package already contains the Vivado 2023.2 reports. A rerun is
optional and must write below `results/rerun/area_ablation/`.
