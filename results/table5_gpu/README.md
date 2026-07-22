# RTX 6000 Ada formal INT4 GEMV rerun

- Status: `MEASURED`
- GPU: NVIDIA RTX 6000 Ada Generation
- Backend: CUTLASS INT4, SM 8.9
- Correctness verification: enabled
- Warmup: 100 iterations
- Measurement window: 10 seconds per shape
- Streaming target: 1024 MB
- Power samples: 1,898-1,919 per paper-relevant shape

The formal run completed all six shapes. For the two Table 5 shapes:

| Shape | Rerun latency | Paper latency | Rerun power | Paper power | Rerun energy | Paper energy |
|---|---:|---:|---:|---:|---:|---:|
| 1x2048x2048 | 0.0110 ms | 0.011 ms | 142.54 W | 175 W | 1.5662 mJ | 1.924 mJ |
| 1x4096x4096 | 0.0191 ms | 0.019 ms | 197.65 W | 236 W | 3.7819 mJ | 4.510 mJ |

Latency reproduces closely (0% and +0.53%), and all six shapes pass correctness
verification. Average power is lower by 18.55% and 16.25%, respectively. Per
the AE acceptance rule, this is treated as an environment/sampling difference,
so the GPU baseline is accepted as reproduced at the method, correctness,
latency, and order-of-magnitude level. The measured columns remain raw; the
paper columns are calibration anchors and are never substituted into measured
CSV/JSON fields.

Machine-readable outputs:

- `int4_gemv_energy.csv`
- `int4_gemv_energy.json`
- `gemv_energy_summary.csv`
- `gemv_energy_summary.json`
