# Table 8 FP/MX Packing-Core Evaluation

This experiment evaluates the decoded mantissa/magnitude multiplication backend
of FP4, FP8, FP10-E4M5, and MXFP4 formats. It does not claim a complete floating-
point pipeline: sign, exponent/scale handling, normalization, rounding,
saturation, accumulation, and repacking remain outside this packed DSP core.

## Scope and packaged result

| Magnitude width | Ultra-DSP products per DSP per cycle |
|---:|---:|
| 1 | 36 |
| 2 | 18 |
| 3 | 10 |
| 4 | 6 |
| 5 | 4 |

All 32 design rows have zero Python-reference errors, covering 34,904
exhaustive pair cases and 1,409,024 random output checks. Vivado reports show
one DSP for each scalar or packed DSP design and zero DSPs for the pure-LUT
parallel baseline.

`NoPackingScalar` is a one-product-per-DSP baseline. `NoDSP_LUTParallel` is the
same-throughput pure-LUT baseline. The main result reports LUT/DSP; FF fields are
retained in the full CSV only for auditability.

## Directory functions

| Path | Function |
|---|---|
| `scripts/fp_layout_solver.py` | Searches legal packed layouts. |
| `scripts/generate_fp_pe_rtl.py` | Generates scalar, non-overlap, Ultra-DSP, and LUT-parallel RTL. |
| `scripts/verilog_self_check.py` | Runs Python and xsim correctness checks. |
| `scripts/synth_fp_pe.tcl` | Synthesizes all manifest entries with Vivado. |
| `scripts/collect_fp_resources.py` | Parses utilization reports and computes per-product metrics. |
| `scripts/audit_correction_luts.py` | Checks correction LUTs against the solver layout. |
| `scripts/plot_fp_pe_area.py` | Generates the area-efficiency plot. |
| `rtl/`, `tb/`, `manifest.csv` | Generated design sources, testbenches, and input manifest. |
| `../../results/table8/` | Canonical CSVs, xsim logs, Vivado reports, and PNG/PDF figures. |

## Reproduction

Deterministic analysis from the repository root:

```bash
python experiments/table8_fp_mx/scripts/fp_layout_solver.py \
  --out results/rerun/table8/fp_layouts.csv
python experiments/table8_fp_mx/scripts/audit_correction_luts.py \
  --layouts results/table8/fp_layouts.csv \
  --report-dir results/table8/vivado_resource \
  --out results/rerun/table8/fp_correction_lut_audit.csv
```

An optional full Vivado/xsim rerun requires Vivado 2023.2. Direct all generated
logs, reports, CSVs, and figures to `results/rerun/table8/`; the complete command
sequence is in `REPRODUCE.md`, Section 8.
