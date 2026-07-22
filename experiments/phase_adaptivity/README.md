# Phase-adaptivity evidence

This directory packages the author-designated P-only, D-only, and Hybrid
64x64 reports. The implementation target is Alveo U55C with Vivado 2023.2 and
4096 logical PEs.

## Folder functions

| Path | Function |
|---|---|
| `../../results/phase_adaptivity/reports/p_only/` | P-only 3x3 post-route utilization, timing, power, kernel utilization, and xsim evidence. |
| `../../results/phase_adaptivity/reports/d_only/` | D-only 1x7 reports with the same evidence scope. |
| `../../results/phase_adaptivity/reports/hybrid/` | Hybrid 3x3/1x7 full-board utilization, timing, kernel utilization, and full-design power provenance. |
| `../../results/phase_adaptivity/phase_adaptivity_summary.csv` | Machine-readable extraction of the three configurations. |
| `../../results/phase_adaptivity/report_manifest.csv` | Raw metadata/SHA-256 paired with each sanitized public report SHA-256. |
| `../../results/phase_adaptivity/REPORTS_MANIFEST.sha256` | Public-copy SHA-256 manifest. |
| `scripts/extract_phase_adaptivity.py` | Re-extracts the summary and verifies functional markers. |
| `AUTHOR_REFERENCE_README.md` | Sanitized author notes for the archived run. |

## Results

| Configuration | Full LUT | Full FF | DSP | WNS at 200 MHz | Power | Functional |
|---|---:|---:|---:|---:|---:|---|
| P-only 3x3 | 454,815 | 447,403 | 4,100 | +0.003 ns | 36.353 W | PASS |
| D-only 1x7 | 365,304 | 389,022 | 4,100 | +0.003 ns | 36.018 W | PASS |
| Hybrid 3x3/1x7 | 519,486 | 475,288 | 4,100 | -0.421 ns | 44.780 W | PASS |

The 4,100 full-design DSP count contains 4,096 kernel DSPs plus four platform
DSPs. Kernel-only reports preserve the phase-specific LUT/FF breakdown. The
Hybrid full-board 200 MHz timing report is intentionally disclosed as not
meeting timing; it is not relabeled as a pass. The same 4096-PE Hybrid kernel
has a separate positive-WNS 210 MHz OOC implementation under
`results/overlap_depth_sweep/depth3_64x64_ooc_210MHz/`, which is the
source of the 216.267 MHz Figure 18 point.

All three power values are Vivado 2023.2 vector-less estimates with `Low`
confidence, not board measurements. The 44.780 W report belongs to the original
PD/Hybrid full design. It provides provenance for rounding the analytical FPGA
input to 45 W; it must not be presented as an independent P-only or D-only run.
The P-only and D-only values come from their own routed designs.

The eleven reports that were already packaged matched freshly sanitized remote
copies byte-for-byte. The Hybrid full-design power report was the only missing
file and is now included. `report_manifest.csv` retains the original raw hash,
raw byte count, and remote modification epoch without exposing a remote path,
account, or node name; the public hash covers the sanitized copy.

## Reproduction

```bash
python experiments/phase_adaptivity/scripts/extract_phase_adaptivity.py
python experiments/phase_adaptivity/scripts/verify_report_manifest.py
```

Expected marker:

```text
PHASE_ADAPTIVITY_PASS configs=3 functional=3 timing_met=2 timing_disclosed_not_met=1 hybrid_power=full_design_provenance
```
