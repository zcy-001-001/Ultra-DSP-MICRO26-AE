# Figures 14 and 15: End-to-End Model

This directory reconstructs the paper's end-to-end plots while keeping three
evidence layers separate:

1. **Raw HLS evidence**: frozen Vitis HLS C-synthesis reports and their parsed
   operator latencies.
2. **Uncalibrated model**: an Amdahl-style theoretical replacement of only the
   matrix operations, using each packing scheme's multiplications per DSP.
3. **Calibrated result**: the same model calibrated to the paper plotting
   anchors. These values reproduce the published figures but are not presented
   as new measurements.

## Directory functions

| Relative path | Function |
|---|---|
| `../../results/figures14_15/evidence/raw_hls_reports/` | Frozen report subset used for the operator split. |
| `hls_source_minimal/` | Minimal HLS source/config closure and 2023.2 report-generation scripts. |
| `inputs/method_parameters.csv` | Packing factors and paper calibration anchors. |
| `requirements.txt` | Tested Python package versions. |
| `scripts/simulate_figures.py` | Report parser, model, calibration, CSV writer, and plotter. |
| `tests/test_simulator.py` | Numerical, schema, and privacy regression checks. |
| `../../results/figures14_15/hls_report_inventory.csv` | Tool version, latency, and SHA-256 inventory. |
| `../../results/figures14_15/hls_remote_provenance.csv` | Relative upstream report mapping and source-to-packaged SHA-256 equality proof. |
| `../../results/figures14_15/raw_hls_operator_breakdown.csv` | Per-layer matrix/non-matrix breakdown. |
| `../../results/figures14_15/stage_model_and_calibration.csv` | Raw-model and calibrated stage values. |
| `../../results/figures14_15/figure14_canonical.csv` | Canonical short-context plot data. |
| `../../results/figures14_15/figure15_canonical.csv` | Canonical 8K simulator plot data. |
| `../../results/figures14_15/figure14_reproduced.*` | Reproduced Figure 14 in PNG/PDF. |
| `../../results/figures14_15/figure15_reproduced.*` | Reproduced Figure 15 in PNG/PDF. |

## Reproduction

From the repository root:

```bash
python experiments/figures14_15/scripts/simulate_figures.py \
  --report-dir results/figures14_15/evidence/raw_hls_reports \
  --result-dir results/rerun/figures14_15
python experiments/figures14_15/tests/test_simulator.py
```

On Windows, the wrapper in the experiment directory runs both commands:

```powershell
.\scripts\reproduce.ps1
```

The packaged run was tested with Python 3.12.7. Install the tested packages
with `python -m pip install -r requirements.txt`; Python 3.10 or newer is
expected to work. The frozen
reports were produced for an Alveo U55C. Their embedded metadata identifies
Vitis HLS 2024.1 and a 5 ns target for the transformer kernel. This is
historical evidence, not a claim that the reports were regenerated under the
AE server's Vivado/Vitis 2023.2 environment.

The 13 packaged reports were also checked against the author-designated
existing HLS tree. Every raw remote SHA-256 equals the packaged SHA-256; see
`results/figures14_15/hls_remote_provenance.csv`. Because the packaged files are byte-identical,
they were not copied again and no synthesis was launched. Only source-relative
paths below `<HLS_SOURCE_ROOT>` are recorded.

## Regenerating the HLS reports with Vitis 2023.2

On a Linux machine with the U55C platform installed, source the installation's
Vitis 2023.2 `settings64.sh` first. Then, from this directory:

```bash
cd hls_source_minimal
bash run_csynth.sh
bash collect_selected_reports.sh
cd ..
python scripts/simulate_figures.py \
  --report-dir hls_source_minimal/regenerated_reports_2023_2 \
  --result-dir ../../../results/rerun/figures14_15/vitis_2023_2
```

`run_csynth.sh` rejects other Vitis versions and writes only under
`hls_source_minimal/build_2023_2/`. The collection step keeps the regenerated
2023.2 reports separate from the frozen historical 2024.1 evidence. Do not
overwrite `results/figures14_15/evidence/raw_hls_reports/`; comparing the two inventories preserves
tool-version provenance. See `hls_source_minimal/SOURCE_INVENTORY.md` for the
source closure, exclusions, and licensing note.

## Model and assumptions

The HLS scheduler contains seven matrix calls per transformer layer: four
4096-output projections and three FFN projections. The matrix loop report gives
a 24-cycle pipeline overhead. The script combines these with the worst-case
reported latencies for attention, normalization/quantization, SwiGLU, and two
residual updates.

For the uncalibrated prefill model:

```text
speedup = 1 / ((1 - matrix_fraction) + matrix_fraction / packing_ratio)
```

Decode is treated as memory-bound, so increasing packing density does not lower
latency in this model. Sequence points use the paper plotting script's
stage-first interpolation:

```text
sequence_ratio = (P * prefill_ratio + D * decode_ratio) / (P + D)
```

The top-level transformer report has undefined latency because sequence
position is runtime-dependent. Therefore, the raw HLS breakdown is a
sequential analytical composition of subreports; it is not a board-measured
end-to-end latency. The initial embedding and final classifier reports are
retained in the inventory but are not amortized into every transformer layer.

The raw HLS-derived matrix fraction differs from the paper-calibrated prefill
fraction. Both remain visible in the CSVs. The calibration fit and per-method
power ratios are derived from `inputs/method_parameters.csv`; no calibrated
field is labeled as measured. Figure 15 extends the calibrated stage model to
five 8K P/D splits and is simulator-only, consistent with the paper.
