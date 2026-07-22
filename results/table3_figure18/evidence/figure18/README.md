# Figure 18 routed-report evidence

This folder archives the existing Vivado 2023.2 reports used by the Figure 18
frequency sweep. No implementation was rerun while creating this evidence.

## Contents

- `reports/`: sanitized copies preserving paths relative to the private report
  root. It contains all 106 timing reports named by
  `results/table3_figure18/figure18_frequency_sweep.csv` and the seven utilization reports for
  the selected 8x8 through 90x90 points.
- `raw_public_manifest.csv`: evidence kind, source-relative path, raw size and
  modification epoch, raw SHA-256, public size and SHA-256, and sanitizer
  replacement count.
- `PUBLIC_MANIFEST.sha256`: hashes of the sanitized public report files.

The raw SHA-256 values identify the source reports without redistributing
private host metadata. Public copies replace host names and private absolute
paths with placeholders. Paths in both manifests are relative; the private
report root is intentionally omitted.

The 8x8 point is retained as supplemental sweep evidence. The paper plot uses
the six selected points from 16x16 through 90x90. At 90x90, the report contains
8,160 routed DSPs for 8,100 logical PEs; this interface overhead is preserved
rather than normalized away.

## Verification

From the artifact root:

```bash
python scripts/sanitize_paths.py \
  results/table3_figure18/evidence/figure18 --check
python experiments/table3_figure18/scripts/verify_figure18_evidence.py
```

Expected report counts are 106 timing reports and seven selected utilization
reports. The evidence verifier also recomputes every public SHA-256 and parses
the reports through the same timing/utilization functions used by the OOC
extractor.
