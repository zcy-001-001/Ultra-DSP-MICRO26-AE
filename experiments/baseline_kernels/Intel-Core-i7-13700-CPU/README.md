# Intel Core i7-13700 CPU Baseline

This directory contains the Intel Core i7-13700 CPU baseline for INT8 LLM GEMM
and batch-1 GEMV. It uses Intel MKL `gemm_s8u8s32` through `ctypes` and measures
energy with the Windows RAPL performance counter.

## Workload

| Item | Value |
|---|---|
| CPU | Intel Core i7-13700 |
| Backend | Intel MKL `gemm_s8u8s32` |
| Data type | INT8 x INT8, symmetric zero point |
| Model shapes | LLaMA-2-7B linear layers |
| Prefill sequence lengths | 512, 1024, 1536 |
| Decode batch size | 1 |
| Power source | Windows `typeperf` RAPL counter |

## Run

Install NumPy and Intel MKL. If MKL is not on the dynamic loader path, point
`MKL_RT_PATH` to the runtime library:

```powershell
$env:MKL_RT_PATH="<mkl-runtime-library>"
```

Run latency/TOPS:

```bash
python benchmark_int8_gemm.py --warmup 50 --iterations 200 --decode-steps "512,1024,1536"
```

Run energy efficiency:

```bash
python energy_efficiency_test.py --power-mode rapl --measure-sec 30 --decode-steps "512,1024,1536"
```

Outputs are written to `results/` when the benchmark is run.

## Key Results

Reference `1 x 4096 x 4096 GEMV` row:

| Platform | Time (ms) | Power (W) | Energy (mJ) | Speedup | Energy Eff. |
|---|---:|---:|---:|---:|---:|
| MKL (i7-13700) | 0.574 | 75 | 42.850 | 1.00x | 1.00x |

## Files

| File | Purpose |
|---|---|
| `benchmark_int8_gemm.py` | Latency/TOPS benchmark for prefill and decode shapes |
| `energy_efficiency_test.py` | TOPS/W and mJ/token measurement with Windows RAPL |
| `README.md` | This file |
