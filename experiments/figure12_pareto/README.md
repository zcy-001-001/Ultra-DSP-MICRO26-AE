# Figure 12 Accuracy-Throughput Pareto

This experiment combines archived accuracy provenance with the fixed FPGA
resource-budget model to regenerate Figure 12. Accuracy is
`RECOMPUTED_FROM_LOGS`; throughput is `ANALYTICAL_MODEL`.

WP521 is a Figure 12 comparison point only. It is not a Table 6 row. The Table 6
formal row set remains BF16, Ultra-DSP, DSP-Packing, and DB-MixQ for two models.

## Fixed method

```text
U55C resources: 9024 DSP, 1,304,000 LUT, 2,607,000 FF
Fixed DSP budget: 4096
LUT/FF budget sweep: 20%, 25%, 30%, 35%, 40%, 45%
Frequency: 200 MHz
throughput = effective_DSPs * packing * 2 * frequency
```

`scripts/generate_pareto.py` reads the packaged accuracy provenance and emits
the budget points and 2x3 figure. Canonical outputs are:

- `results/figure12/accuracy_provenance.csv`
- `results/figure12/pareto_budget_points.csv`
- `results/figure12/pareto_budget_2x3.png`
- `results/figure12/pareto_budget_2x3.pdf`

## Reproduction

From the repository root:

```bash
python experiments/figure12_pareto/scripts/generate_pareto.py \
  --out-dir results/rerun/figure12
```

The generated rows must preserve the shared lossless-accuracy explanation for
Ultra-DSP, DuoQ, and UDP, while keeping WP521 provenance separate from Table 6.
