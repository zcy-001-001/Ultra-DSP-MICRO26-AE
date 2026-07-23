# Existing OOC report extraction

This directory contains the complete GEMV implementation inputs and the
post-implementation evidence used by Table 3 and Figure 18. Reviewers can
either re-parse the packaged routed reports or regenerate the Table 3 OOC
projects from the published HLS, RTL, configuration, and shell-script sources.

## Directory contents

| Path | Purpose |
|---|---|
| `baseline_implementations/` | Complete Table 3 GEMV sources and 200 MHz OOC/full-build scripts for WP521, DB-MixQ, DSP-Packing, DuoQ, UDP, and Ultra-DSP. |
| `scripts/extract_ooc_reports.py` | Read existing routed reports, extract scalar metrics, calculate throughput, and emit sanitized CSV files. |
| `tests/test_extract_ooc_reports.py` | Unit tests for utilization, power, timing, and path-safety parsing. |
| `../../results/table3_figure18/evidence/table3/` | Forty-two sanitized routed reports plus raw-to-public SHA-256 provenance for all 14 Table 3 rows. |
| `../../results/table3_figure18/table3_ooc_summary.csv` | Canonical Table 3 resource, power, throughput, energy-efficiency, and timing rows. |
| `../../results/table3_figure18/figure18_post_implementation_summary.csv` | One timing-clean point per PE size; the six paper points are explicitly flagged. |
| `../../results/table3_figure18/figure18_frequency_sweep.csv` | All readable routed frequency attempts used by the selection rule. |
| `../../results/table3_figure18/evidence/figure18/` | All 106 routed timing attempts, seven selected utilization reports, and raw-to-public SHA-256 provenance. |

## Table 3 method

All rows use an array budget of 4,096 DSPs and a 200 MHz operating frequency.
Throughput is calculated as:

```text
throughput_GOPS = DSP_count * packing_count * 2 * frequency_MHz / 1000
```

The packing counts are 4/6/6/4/6/9 for the W4A4 WP521, DB-MixQ,
DSP-Packing, DuoQ, UDP, and Ultra-DSP rows. The mixed-format UDP and Ultra-DSP
packing counts are recorded directly in the CSV. `power_w` is the Vivado
`Total On-Chip Power` estimate in the routed OOC report, not a board power
measurement.

DSP-Packing and DuoQ use their second archived OOC result because those reports
match the resource rows frozen in the paper. UDP and Ultra-DSP use the general
and maximum-packing directories, respectively. Only sanitized relative report
paths are retained.

The W4A4 Ultra-DSP anchor is 244.939 kLUT, 259.987 kFF, 6.218 W,
14,745.6 GOPS, and 2,371.44 GOPS/W.

## Table 3 implementation sources

The implementation-to-row mapping and every build command are documented in
[`baseline_implementations/README.md`](baseline_implementations/README.md).
The published source set contains all 14 Table 3 rows:

- W4A4: WP521, DB-MixQ, DSP-Packing, DuoQ, UDP, and Ultra-DSP.
- W5A5, W4A3, W4A5, and W5A3: UDP and Ultra-DSP.

The baseline folders retain the source names used by the original builds.
`DeepBurning/` corresponds to the DB-MixQ row, `UDP-general/` contains the five
UDP precision pairs, and `Ultra-DSP-MAX/` contains the five maximum-packing
Ultra-DSP precision pairs. The source package includes `GEMV.cpp`, `GEMV.h`,
processing-element RTL, C black-box models, black-box JSON metadata, HLS/link
configurations, and the synthesis/implementation scripts.

Configure Vivado/Vitis 2023.2 and the U55C platform before a full rerun:

```bash
export XILINX_VITIS_SETTINGS=<VITIS_2023_2>/settings64.sh
export XILINX_VIVADO_SETTINGS=<VIVADO_2023_2>/settings64.sh
export XRT_SETUP=<XRT_INSTALL>/setup.sh
export PLATFORM=<U55C_PLATFORM>.xpfm
```

For example, regenerate the six W4A4 OOC implementations from the repository
root:

```bash
BASE=experiments/table3_figure18/baseline_implementations
bash "$BASE/WP521/ooc_implement.sh"
bash "$BASE/DeepBurning/ooc_implement.sh"
bash "$BASE/DSP-Packing/ooc_implement1.sh"
bash "$BASE/DuoQ/ooc_implement1.sh"
bash "$BASE/UDP-general/INT4_INT4/ooc_implement.sh"
bash "$BASE/Ultra-DSP-MAX/W4A4/ooc_implement.sh"
```

Use the commands in the baseline implementation README to generate the other
eight precision/method rows. Fresh OOC reports can then be parsed directly:

```bash
python experiments/table3_figure18/scripts/extract_ooc_reports.py \
  --source-root experiments/table3_figure18/baseline_implementations \
  --output-dir results/rerun/table3_figure18
```

## Figure 18 method

For every symmetric array from 8x8 through 90x90, the script reads both
archived OOC frequency-sweep variants when present. It selects the highest
routed target frequency with non-negative WNS. The estimated post-route maximum
frequency is:

```text
estimated_fmax_MHz = 1000 / (1000 / target_frequency_MHz - WNS_ns)
```

The paper points contain 256, 1,024, 2,304, 4,096, 6,400, and 8,100 PEs. The
64-PE point is retained as supplemental evidence and marked false in
`selected_for_paper_figure`.

`expected_pe_dsp` is the logical one-DSP-per-PE count used on the x-axis.
`routed_report_dsp` preserves the actual routed report value, which may include
small interface overheads.

## Reproduction

From the repository root, use either a local report tree:

```text
python experiments/table3_figure18/scripts/extract_ooc_reports.py \
  --source-root <REPORT_ROOT> --output-dir results/rerun/table3_figure18
```

or a report tree reachable through OpenSSH:

```text
python experiments/table3_figure18/scripts/extract_ooc_reports.py \
  --ssh-host <REMOTE_HOST> --source-root <REPORT_ROOT> \
  --output-dir results/rerun/table3_figure18
```

Then run the parser tests and a read-only numerical check:

```text
python -m unittest discover -s experiments/table3_figure18/tests -v
python experiments/table3_figure18/scripts/extract_ooc_reports.py \
  --ssh-host <REMOTE_HOST> --source-root <REPORT_ROOT> --check
```

No command above launches Vivado or changes the source report tree.

The packaged evidence can also be verified without access to the original
report tree:

```text
python experiments/table3_figure18/scripts/extract_ooc_reports.py \
  --source-root results/table3_figure18/evidence/table3 \
  --table3-only --output-dir results/rerun/table3
python experiments/table3_figure18/scripts/verify_figure18_evidence.py
python experiments/table3_figure18/scripts/refresh_table3_public_manifest.py
python -m unittest discover -s experiments/table3_figure18/tests -v
```
