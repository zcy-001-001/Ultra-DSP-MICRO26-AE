# Table 6 fresh-run tolerance comparison

Run ID: `table6-symmetric-a800-20260720-v2`  
Per-task tolerance: 3.00 percentage points  
Average tolerance: 2.00 percentage points
Strict status: **OUT_OF_TOLERANCE**  
Final AE status: **PASS_WITH_AUTHOR_ACCEPTED_VARIANCE**

| Model | Method | Max task delta | Average delta | Strict status | AE status |
|---|---|---:|---:|---|---|
| llama2_7b | Baseline(BF16) | 0.63 | +0.08 | PASS | PASS |
| llama2_7b | Ultra-DSP | 2.00 | -1.29 | PASS | PASS |
| llama2_7b | DSP-Packing | 0.38 | -0.10 | PASS | PASS |
| llama2_7b | DB-MixQ | 0.40 | -0.09 | PASS | PASS |
| llama3_8b | Baseline(BF16) | 0.20 | -0.09 | PASS | PASS |
| llama3_8b | Ultra-DSP | 3.24 | -1.96 | OUT_OF_TOLERANCE | AUTHOR_ACCEPTED_VARIANCE |
| llama3_8b | DSP-Packing | 1.00 | -0.31 | PASS | PASS |
| llama3_8b | DB-MixQ | 0.80 | +0.28 | PASS | PASS |

## Trend checks

- PASS: `llama2_7b` - BF16 > Ultra-DSP > lossy packing baselines
- PASS: `llama3_8b` - BF16 > Ultra-DSP > lossy packing baselines

Overall status: **PASS_WITH_AUTHOR_ACCEPTED_VARIANCE**
