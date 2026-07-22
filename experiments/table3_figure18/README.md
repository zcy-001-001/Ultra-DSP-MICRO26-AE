# Existing OOC report extraction

This directory freezes the existing post-implementation evidence used by
Table 3 and Figure 18. It does not rerun synthesis or implementation.

## Directory contents

| Path | Purpose |
|---|---|
| `scripts/extract_ooc_reports.py` | Read existing routed reports, extract scalar metrics, calculate throughput, and emit sanitized CSV files. |
| `tests/test_extract_ooc_reports.py` | Unit tests for utilization, power, timing, and path-safety parsing. |
| `../../results/table3_figure18/evidence/table3/` | Forty-two sanitized routed reports plus raw-to-public SHA-256 provenance for all 14 Table 3 rows. |
| `../../results/table3_figure18/table3_ooc_summary.csv` | Canonical Table 3 resource, power, throughput, energy-efficiency, and timing rows. |
| `../../results/table3_figure18/figure18_post_implementation_summary.csv` | One timing-clean point per PE size; the six paper points are explicitly flagged. |
| `../../results/table3_figure18/figure18_frequency_sweep.csv` | All readable routed frequency attempts used by the selection rule. |
| `../../results/table3_figure18/evidence/figure18/` | All 106 routed timing attempts, seven selected utilization reports, and raw-to-public SHA-256 provenance. |
| `TODO.md` | Remaining interpretation notes that do not block extraction. |

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
private report root:

```text
python experiments/table3_figure18/scripts/extract_ooc_reports.py \
  --source-root results/table3_figure18/evidence/table3 \
  --table3-only --output-dir results/rerun/table3
python experiments/table3_figure18/scripts/verify_figure18_evidence.py
python experiments/table3_figure18/scripts/refresh_table3_public_manifest.py
python -m unittest discover -s experiments/table3_figure18/tests -v
```
