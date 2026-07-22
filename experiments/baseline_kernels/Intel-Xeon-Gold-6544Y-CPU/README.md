# Intel Xeon Gold 6544Y CPU Baseline

This directory contains the 1x Intel Xeon Gold 6544Y CPU baseline for
INT8 LLM GEMM and batch-1 GEMV. It uses Intel MKL `gemm_s8u8s32`; on Sapphire
Rapids, MKL dispatches to AMX-INT8 when available.

## Workload

| Item | Value |
|---|---|
| CPU | 1x Intel Xeon Gold 6544Y |
| Backend | Intel MKL `gemm_s8u8s32` |
| ISA | AMX-INT8 / AVX-512 VNNI |
| Data type | INT8 x INT8, symmetric zero point |
| Model shapes | LLaMA-2-7B linear layers |
| Prefill sequence lengths | 512, 1024, 1536 |
| Decode batch size | 1 |
| Power source | Linux RAPL sysfs package counter |

## Run

```bash
conda activate base
python3 benchmark_int8_gemm.py --warmup 50 --iterations 200 --decode-steps "512,1024,1536"
python3 energy_efficiency_test.py --power-mode rapl --measure-sec 30 --decode-steps "512,1024,1536"
```

Or run all default measurements:

```bash
bash run_all.sh
```

RAPL requires read access to package energy counters. Ask the system
administrator to enable read access if the counters are not readable.

## Key Results

Reference `1 x 4096 x 4096 GEMV` row:

| Platform | Time (ms) | Power (W) | Energy (mJ) | Speedup | Energy Eff. |
|---|---:|---:|---:|---:|---:|
| MKL (Xeon Gold 6544Y) | 0.369 | 229 | 84.325 | 1.56x | 0.51x |

## Output

Benchmark runs write CSV/JSON outputs under `results/`:

| File | Purpose |
|---|---|
| `gemm_benchmark.*` | Latency/TOPS without power |
| `energy_efficiency.*` | Prefill and single-step decode energy metrics |
| `decode_sequence_benchmark.*` | Multi-step decode latency |
| `decode_sequence_energy.*` | Multi-step decode energy metrics |

## Files

| File | Purpose |
|---|---|
| `benchmark_int8_gemm.py` | Latency/TOPS benchmark for prefill and decode shapes |
| `energy_efficiency_test.py` | RAPL-based TOPS/W and mJ/token measurement |
| `run_all.sh` | Runs the default benchmark sequence |
| `README.md` | This file |
