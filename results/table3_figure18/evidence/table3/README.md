# Table 3 routed OOC evidence

This directory contains the minimum routed-report evidence needed to rebuild
the 14 rows of Table 3. It archives existing Vivado reports only; no synthesis
or implementation is run here.

## Directory contents

| Path | Purpose |
|---|---|
| `Baseline/` | Forty-two sanitized reports, preserving the relative paths used by `results/table3_figure18/table3_ooc_summary.csv`. |
| `manifest.json` | Machine-readable mapping from each public report to its source-relative path, raw source hash, public-copy hash, byte counts, and Vivado version. |

Each Table 3 row has exactly three reports:

- `gemv_kernel_utilization_routed.rpt` for LUT, FF, and DSP usage;
- `bd_0_wrapper_power_routed.rpt` for routed estimated power;
- `bd_0_wrapper_timing_summary_routed.rpt` for WNS and timing closure.

The public copies replace the machine-specific Vivado `Host` header with
`<REDACTED_HOST>`. Source roots, usernames, internal hostnames, and private
absolute paths are not stored in this directory. The original remote SHA256
is retained in `manifest.json` so the sanitized copy remains auditable.

## Operand-swapped source mapping

The paper labels formats as weight bits followed by activation bits. Some
archived Ultra-DSP directories use the opposite operand order because integer
multiplication is commutative:

| Paper row | Archived source directory | Reason |
|---|---|---|
| `W4A3 / Ultra-DSP` | `Baseline/Ultra-DSP-MAX/W3A4/` | The packed multiplier layout is reused with the operands exchanged. |
| `W5A3 / Ultra-DSP` | `Baseline/Ultra-DSP-MAX/W3A5/` | The packed multiplier layout is reused with the operands exchanged. |

UDP follows the same archived naming convention through `INT3_INT4` and
`INT3_INT5`. The Table 3 labels remain `W4A3` and `W5A3`.

## Reproduction

From the repository root, parse the packaged evidence into a new output
directory without contacting a private build host:

```text
python experiments/table3_figure18/scripts/extract_ooc_reports.py \
  --source-root results/table3_figure18/evidence/table3 \
  --table3-only \
  --output-dir results/rerun/table3
```

Then run the parser unit tests:

```text
python -m unittest discover -s experiments/table3_figure18/tests -v
```

If a later package-wide privacy pass changes only public placeholders, refresh
the public-copy hash/size fields without changing raw remote provenance:

```text
python experiments/table3_figure18/scripts/refresh_table3_public_manifest.py
```

The regenerated Table 3 CSV should contain 14 rows. Throughput is derived from
the DSP count, packing count, and 200 MHz frequency; it is not parsed from the
reports. Power is the Vivado routed estimate rather than board telemetry.
