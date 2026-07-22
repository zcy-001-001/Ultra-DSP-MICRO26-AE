# Overlap-Depth Sweep

This supporting experiment studies the packing/resource tradeoff when the
allowed pointwise overlap depth increases. The paper's W4A4 design uses the
depth-3 `3x3` prefill and `1x7` decode layout. Deeper layouts are supplemental
and are not substituted for the paper point.

| Depth | Prefill layout | Prefill products | Decode layout | Main limitation |
|---:|---|---:|---|---|
| 3 | 3x3 | 9 | 1x7 | Paper configuration; 64x64 OOC closes at 210 MHz |
| 4 | 4x3 | 12 | 1x7 | Higher LUT/FF cost; tested routed targets miss timing |
| 5 | 5x4 | 20 | 1x7 | 64x64 design exceeds U55C LUT capacity at placement |
| 6 | 5x4 | 20 | 1x7 | No additional throughput over depth 5 |

## Directory functions

| Path | Function |
|---|---|
| `scripts/generate_overlap_depth_sweep.py` | Generates legal layouts and RTL. |
| `scripts/audit_overlap_constraints.py` | Audits uniform and optional exhaustive constraints. |
| `scripts/verilog_self_check.py` | Single-PE xsim checks. |
| `scripts/depth64_core_self_check.py` | 64x64 transformed-core xsim checks. |
| `scripts/extract_depth3_ooc.py` | Parses the archived depth-3 64x64 routed point. |
| `scripts/collect_depth64_ooc_results.py` | Summarizes deeper 64x64 attempts. |
| `rtl/` | Generated W4A4 depth variants. |
| `../../results/overlap_depth_sweep/` | Canonical CSVs, logs, self-checks, and Vivado reports. |

## Reproduction

Read-only checks from the repository root:

```bash
python experiments/overlap_depth_sweep/scripts/audit_overlap_constraints.py \
  --out results/rerun/overlap_depth_sweep/constraint_audit.csv
python experiments/overlap_depth_sweep/scripts/extract_depth3_ooc.py
```

Expected depth-3 marker:

```text
DEPTH3_OOC_PASS PE=4096 Fmax_MHz=216.267 DSP=4096
```

Optional xsim/Vivado runs must write under
`results/rerun/overlap_depth_sweep/`. Existing routed evidence is already
packaged and need not be regenerated.
