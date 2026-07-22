# Table 7: mixed-precision Ultra-DSP accuracy

This directory documents how to run the author-specified narrow-symmetric
OSTQuant workflow for Table 7. The formal AE result is the paper anchor in
`results/table7/table7_paper_anchor.*`; this package does not claim a new Table 7 run.

Previously collected evaluation logs and summaries remain under `results/table7/` as
non-canonical development evidence. They are not used as the formal Table 7
source and should not be cited as a new AE measurement.

## Folder functions

| Path | Function |
|---|---|
| `scripts/run_one_precision_ultradsp.sh` | Train and evaluate one model/precision pair. |
| `scripts/run_mixed_precision_ultradsp_host.sh` | Schedule model/precision pairs on an available multi-GPU host. |
| `scripts/parse_mixed_precision_ultradsp.py` | Parse per-run lm-eval logs into model-level task averages. |
| `scripts/build_table7_geomean.py` | Combine seven mixed-precision formats with the W4A4 Table 6 Ultra-DSP rows. |
| `results/table7/table7_paper_anchor.*` | Formal Table 7 values copied from the paper. |
| `results/rerun/table7/table7_optional_recomputed_*` | Suggested output names for an optional evaluator rerun. |

## Quantization convention

The workflow uses the paper's narrow symmetric variant:

```text
--narrow_symmetric=True
--a_asym False
--k_asym False
--v_asym False
```

The signed integer ranges are `[-3,3]`, `[-7,7]`, and `[-15,15]` for 3, 4,
and 5 bits. Set `w_bits` to W and set `a_bits`, `down_bits`, `k_bits`, and
`v_bits` to A. Ultra-DSP evaluation uses `--linear_int_mode=exact`.

## Reproduction procedure

The commands below are instructions only; they were not run while preparing
this AE update. Prepare the external model weights, datasets, and Python
environment, then run from `experiments/table6_table7_accuracy/`.
environment, then run one model per available GPU host:

```bash
cd <REPO_ROOT>
source <ENV_ROOT>/bin/activate

MODEL_KEYS=llama2_7b \
RUN_DIR=<RUN_ROOT>/mixed_precision_ultradsp \
experiments/mixed_precision_ultradsp/scripts/run_mixed_precision_ultradsp_host.sh

MODEL_KEYS=llama3_8b \
RUN_DIR=<RUN_ROOT>/mixed_precision_ultradsp \
experiments/mixed_precision_ultradsp/scripts/run_mixed_precision_ultradsp_host.sh
```

After both model runs finish, parse and aggregate them into explicitly
optional output files:

```bash
python experiments/mixed_precision_ultradsp/scripts/parse_mixed_precision_ultradsp.py \
  --run-dir <RUN_ROOT>/mixed_precision_ultradsp \
  --out-json <RUN_ROOT>/mixed_precision_ultradsp_summary.json \
  --out-md <RUN_ROOT>/mixed_precision_ultradsp_table.md

python experiments/mixed_precision_ultradsp/scripts/build_table7_geomean.py \
  --mixed-summary <RUN_ROOT>/mixed_precision_ultradsp_summary.json
```

Expected aggregation marker:

```text
TABLE7_GEOMEAN_PASS formats=8
```

For each format, the script first computes the arithmetic mean of ARC-e,
HellaSwag, PIQA, and OBQA for each model, then reports
`sqrt(Avg_Llama2 * Avg_Llama3)`. W4A4 is joined from the two Table 6
Ultra-DSP rows. Any rerun must retain its raw logs and environment record and
must remain separate from `table7_paper_anchor.*`.
