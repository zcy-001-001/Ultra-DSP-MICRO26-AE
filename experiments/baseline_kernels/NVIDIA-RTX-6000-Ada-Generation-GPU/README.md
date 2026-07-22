# NVIDIA RTX 6000 Ada Generation GPU Baseline

This directory contains the GPU batch-1 GEMV baseline used for the reference
CPU/GPU comparison. The benchmark targets NVIDIA RTX 6000 Ada Generation GPU
and measures INT8 and INT4 GEMV latency, energy, and TOPS/W with a memory-aware
streaming working set.

## Hardware And Software

| Item | Value |
|---|---|
| GPU | NVIDIA RTX 6000 Ada Generation |
| Architecture | Ada Lovelace, SM 8.9 |
| Power limit | 300 W |
| Python | Python 3 with PyTorch CUDA |
| CUDA build target | `sm_89` |
| Power API | NVML via `pynvml` |
| CUTLASS | Bundled headers in `../cutlass` |

## Workloads

All configurations are batch-1 GEMV:

| Name | Shape |
|---|---|
| `GEMV_1024x1024` | `[1,1024]x[1024,1024]` |
| `GEMV_2048x2048` | `[1,2048]x[2048,2048]` |
| `GEMV_4096x4096` | `[1,4096]x[4096,4096]` |
| `GEMV_4096x12288` | `[1,4096]x[4096,12288]` |
| `GEMV_4096x16384` | `[1,4096]x[4096,16384]` |
| `GEMV_8192x8192` | `[1,8192]x[8192,8192]` |

## Method

- `cutlass_int8_gemv.cu` implements INT8 x INT8 -> INT32 GEMV.
- `cutlass_int4_gemv.cu` implements packed INT4 x INT4 -> INT32 GEMV.
- `benchmark_gemv.py` rotates through a large bank of fake weights so the
  measurement includes GPU memory traffic rather than repeatedly reusing one
  cache-hot weight tensor.
- Default streaming target is `1024 MB` per shape, capped at 512 tensors.
- Power is sampled by a helper process through NVML every 5 ms.
- `--verify` compares each CUTLASS output against an `int32` PyTorch reference.

## Run

```bash
pip install torch pynvml
python benchmark_gemv.py --gpu 0 --dtype all --verify --warmup 10 --measure-sec 0.2 --streaming-mb 256 --out-dir results_verify
python benchmark_gemv.py --gpu 0 --dtype all --warmup 100 --measure-sec 10 --streaming-mb 1024
```

To run one datatype:

```bash
python benchmark_gemv.py --gpu 0 --dtype int8 --warmup 100 --measure-sec 10 --streaming-mb 1024
python benchmark_gemv.py --gpu 0 --dtype int4 --warmup 100 --measure-sec 10 --streaming-mb 1024
```

If using an external CUTLASS checkout, set:

```bash
export CUTLASS_ROOT=<cutlass-checkout>
```

## Key Results

Reference `1 x 4096 x 4096 GEMV` row:

| Platform | Time (ms) | Power (W) | Energy (mJ) | Speedup | Energy Eff. |
|---|---:|---:|---:|---:|---:|
| CUTLASS (RTX 6000 Ada) | 0.019 | 236 | 4.510 | 30.07x | 9.50x |

## Files

| File | Purpose |
|---|---|
| `benchmark_gemv.py` | Main runner for INT8/INT4 GEMV measurement |
| `benchmark_utils.py` | Latency, TOPS/W, energy, and result I/O helpers |
| `build_ext.py` | JIT build for CUTLASS CUDA extensions |
| `config.py` | GEMV shape definitions |
| `cutlass_int8_gemv.cu` | INT8 CUTLASS GEMV extension |
| `cutlass_int4_gemv.cu` | INT4 CUTLASS GEMV extension |
| `power_monitor.py` | NVML helper-process power sampler |
