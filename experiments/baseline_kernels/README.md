# CPU/GPU Baseline Artifact

This artifact contains the CPU and GPU baseline benchmarks used for the
reference CPU/GPU comparison:

- `Intel-Core-i7-13700-CPU/`: Intel Core i7-13700 INT8 MKL baseline on Windows.
- `Intel-Xeon-Gold-6544Y-CPU/`: 1x Intel Xeon Gold 6544Y INT8 MKL baseline on Linux.
- `NVIDIA-RTX-6000-Ada-Generation-GPU/`: NVIDIA RTX 6000 Ada Generation GPU CUTLASS GEMV baseline.
- `cutlass/`: CUTLASS headers required to build the GPU extension.

## Environment Summary

| Component | Target | Main dependency | Power source |
|---|---|---|---|
| `Intel-Core-i7-13700-CPU` | Intel Core i7-13700 | Intel MKL `gemm_s8u8s32` | Windows RAPL counter |
| `Intel-Xeon-Gold-6544Y-CPU` | 1x Intel Xeon Gold 6544Y | Intel MKL `gemm_s8u8s32` | Linux RAPL sysfs |
| `NVIDIA-RTX-6000-Ada-Generation-GPU` | NVIDIA RTX 6000 Ada Generation GPU | PyTorch CUDA extension + CUTLASS | NVML |

## Quick Start

Run the CPU baselines from their own directories:

```bash
cd Intel-Core-i7-13700-CPU
python benchmark_int8_gemm.py --warmup 50 --iterations 200 --decode-steps "512,1024,1536"
python energy_efficiency_test.py --measure-sec 30 --decode-steps "512,1024,1536"
```

```bash
cd Intel-Xeon-Gold-6544Y-CPU
python3 benchmark_int8_gemm.py --warmup 50 --iterations 200 --decode-steps "512,1024,1536"
python3 energy_efficiency_test.py --power-mode rapl --measure-sec 30 --decode-steps "512,1024,1536"
```

Run the GPU GEMV baseline:

```bash
cd NVIDIA-RTX-6000-Ada-Generation-GPU
python benchmark_gemv.py --gpu 0 --dtype all --verify --warmup 10 --measure-sec 0.2 --streaming-mb 256 --out-dir results_verify
python benchmark_gemv.py --gpu 0 --dtype all --warmup 100 --measure-sec 10 --streaming-mb 1024
```

## Key Results

Reference `1 x 4096 x 4096 GEMV` comparison:

| Platform | Time (ms) | Power (W) | Energy (mJ) | Speedup | Energy Eff. |
|---|---:|---:|---:|---:|---:|
| MKL (i7-13700) | 0.574 | 75 | 42.850 | 1.00x | 1.00x |
| MKL (Xeon Gold 6544Y) | 0.369 | 229 | 84.325 | 1.56x | 0.51x |
| CUTLASS (RTX 6000 Ada) | 0.019 | 236 | 4.510 | 30.07x | 9.50x |
