# W4A4 Component Ablation

This experiment separates the W4A4 design into normal signed packing,
sign-magnitude conversion, lossless overpacking/correction, ILP layout, and
resource optimization stages.

| Stage | Prefill/decode packing | LUT | FF | DSP |
|---|---:|---:|---:|---:|
| Normal signed | 4/2 | 11 | 0 | 1 |
| Sign-magnitude | 6/5 | 40 | 27 | 1 |
| Lossless overpacking | 8/6 | 82 | 90 | 1 |
| ILP layout | 9/7 | 121 | 31 | 1 |
| Resource-optimized Ultra-DSP | 9/7 | 75 | 67 | 1 |

The V0 source is
`verilog/resource_targets/w4a4_sf_v0_normal_signed_p2d2_single_dsp.v`.
Input definitions are under `data/`; code and synthesis drivers are under
`scripts/`. Canonical CSVs, PNG/PDF figures, and Vivado reports are under
`results/component_ablation/`.

## Reproduction

Deterministic checks and summaries from the repository root:

```bash
python experiments/component_ablation/scripts/check_v0_normal_signed.py
python experiments/component_ablation/scripts/measure_accuracy.py \
  --groups 20 --samples-per-group 10000 \
  --out results/rerun/component_ablation/component_accuracy.csv \
  --group-out results/rerun/component_ablation/component_accuracy_groups.csv
python experiments/component_ablation/scripts/packing_analysis.py \
  --out results/rerun/component_ablation/packing_summary.csv
```

An optional Vivado 2023.2 rerun uses
`experiments/component_ablation/scripts/synth_signmag_first_vivado.tcl`; direct
all reports and parsed summaries to `results/rerun/component_ablation/`.
