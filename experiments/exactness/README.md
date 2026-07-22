# Integer Exactness Sweep

This experiment checks result-level arithmetic exactness over all 49 weight and
activation precision pairs from W2A2 through W8A8. The code models sign/magnitude
decomposition, unsigned DSP packing, overlap correction, and product extraction.

## Contents

| Path | Function |
|---|---|
| `scripts/measure_absolute_precision.py` | Runs the seeded random sweep and writes group and summary tables. |
| `../../results/exactness/absolute_precision_summary.csv` | Canonical long-form result table. |
| `../../results/exactness/absolute_precision_groups.csv` | Per-group audit records. |
| `../../results/exactness/absolute_precision_5x2_table.md` | Human-readable summary table. |

## Method and result

- Inputs use signed symmetric integer ranges.
- Seed: 1305.
- Canonical run: 100 groups and 10,000 random pairs per group, or one million
  pairs for every precision combination.
- Pass criterion: Ultra-DSP `EP=0`, `MSE=0`, `error_count=0`, and maximum
  absolute error zero for all 49 rows.

The packaged result passes all 49 rows. DSP-Packing and DeepBurning W4A4 rows
are retained as non-lossless comparison points.

## Reproduction

From the repository root:

```bash
python experiments/exactness/scripts/measure_absolute_precision.py \
  --groups 100 --samples-per-group 10000 --seed 1305 \
  --out-dir results/rerun/exactness
```

For a quick code-path smoke test, reduce `--groups` and
`--samples-per-group`, but do not compare the reduced run with the canonical
evidence volume.
