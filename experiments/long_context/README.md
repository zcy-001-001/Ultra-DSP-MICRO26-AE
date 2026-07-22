# Long-Context Plot Support

This directory contains plotting scripts for the 8K prefill/decode splits used
to support Figure 15. The scripts reuse the packaged stage-level model and emit
normalized speedup and energy-efficiency tables and figures.

Canonical outputs are under `results/long_context/`:

- `long_context_8k_points.csv`
- `long_context_8k_normalized_speedup.csv`
- `long_context_8k_normalized_energy_efficiency.csv`
- PNG/PDF figures for both metrics

## Reproduction

From the repository root:

```bash
python experiments/long_context/scripts/plot_long_context.py \
  --out-dir results/rerun/long_context
```

If the optional publication font is unavailable, the plotter falls back to the
local serif stack. This changes typography only, not the numerical CSV data.
