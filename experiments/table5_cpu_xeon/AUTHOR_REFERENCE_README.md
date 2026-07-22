# CPU GEMV Memory-Aware Benchmark — Intel Xeon Gold 6544Y

这个目录是针对 `A-MICRO-CPU-GPU-Analysis/intel-6544Y-CPU` 基线做的 **INT8 GEMV 专项测试**。目标不是只测 MKL 内核本身的纯计算速度，而是更贴近 LLaMA decode 阶段的真实行为，把 **权重加载、工作集大小、NUMA first-touch、虚拟内存页映射带来的 memory-bound 影响** 一起考虑进去。

当前版本只实现了 **INT8 CPU 路径**，因为原始 `intel-6544Y-CPU` 基线也是走 Intel MKL 的 INT8 接口；CPU 上没有新增 INT4 内核。

## 1. 这些文件分别是什么

| 文件 | 作用 |
|------|------|
| `benchmark_gemv_cpu.py` | 主入口。遍历所有 GEMV shape，执行 warmup / 正式测量，输出 latency、power、energy、TOPS/W。 |
| `mkl_int8.py` | MKL `gemm_s8u8s32_` 的 ctypes 封装，同时实现 memory-aware 的 streaming weight bank 和 NUMA first-touch 分配。 |
| `power_monitor.py` | 读取 Linux RAPL sysfs，分别统计 `package-0`、`package-1` 和总功耗。 |
| `benchmark_utils.py` | 通用统计逻辑，负责 sustained loop、平均 latency、energy / TOPS/W 计算，以及 CSV / JSON 落盘。 |
| `config.py` | 本次要测的 GEMV shape 配置。 |
| `results/` | 正式测量结果。 |
| `results_verify/` | 短时验证结果，用来确认实现与参考结果一致。 |

## 2. 相比原 `intel-6544Y-CPU` 基线，改了什么

原目录里的 decode/GEMV 测法，核心还是：

1. 建一个固定的 `A` 和固定的 `B`
2. 重复调用 MKL
3. 在持续一段时间后统计平均 latency 和 RAPL 功耗

这个方法适合做 **CPU baseline**，但如果我们关心真实的 decode memory-bound 行为，它有两个局限：

1. 同一块权重会被反复复用，cache / page table / NUMA 映射趋于稳定。
2. 测到的更像 “同一 GEMV 内核 steady-state 重复执行” 的代价，而不是 “每步都要从更大权重空间里取下一块数据” 的代价。

所以这个新目录里专门做了几个增强：

1. **Streaming weight bank**
   不只保留一个 `[K, N]` 权重矩阵，而是保留很多个，组成一个大 bank。
2. **固定目标工作集**
   默认把权重 bank 总大小拉到约 `1024 MB`，尽量让测试不被 LLC 热缓存“美化”。
3. **随机访问顺序**
   每次 GEMV 从 bank 中随机抽下一块权重，减少重复访问同一矩阵的局部性。
4. **NUMA first-touch**
   这台机器是双路 6544Y。分配权重 bank 时，代码会在不同 NUMA node 的 CPU 集之间轮流切 affinity，让内存页 first-touch 更均匀地落到两个 socket。
5. **RAPL 按 socket 统计**
   输出 `package-0`、`package-1` 和总功耗，而不是只看单个总数。

## 3. 这里到底测的是什么

每个 case 都是：

- 输入激活：`A`，shape 为 `[1, K]`
- 权重矩阵：`B`，shape 为 `[K, N]`
- 输出：`C`，shape 为 `[1, N]`

也就是标准 GEMV：

`[1, K] x [K, N] -> [1, N]`

本次覆盖的 shape 是：

| 名称 | Shape |
|------|-------|
| `GEMV_1024x1024` | `[1,1024] x [1024,1024]` |
| `GEMV_2048x2048` | `[1,2048] x [2048,2048]` |
| `GEMV_4096x4096` | `[1,4096] x [4096,4096]` |
| `GEMV_4096x12288` | `[1,4096] x [4096,12288]` |
| `GEMV_4096x16384` | `[1,4096] x [4096,16384]` |
| `GEMV_8192x8192` | `[1,8192] x [8192,8192]` |

## 4. 数据是怎么设置的

这里的数据是 **随机伪数据**，不依赖真实 LLaMA checkpoint：

- `A`：随机 `INT8`
- `B`：随机 `INT8`，之后转换成 MKL 需要的 `UINT8` 表示
- `C`：`INT32`

这样做的原因是你关心的是：

1. shape 对不对
2. 访存行为对不对
3. latency / power / energy efficiency 的相对变化是否合理

所以这里重点不是模型数值语义，而是 **算子尺寸和内存访问模式**。

## 5. 关键设置是怎么配的

### 5.1 MKL 和线程

代码复用了原 baseline 的 MKL INT8 接口：

- 后端：`gemm_s8u8s32_`
- 运行时：`libmkl_rt.so`
- 默认线程数：`os.cpu_count()`，在这台机器上是 `64`

另外设置了：

- `MKL_DYNAMIC=FALSE`
- `OMP_PROC_BIND=spread`
- `OMP_PLACES=cores`

含义是：

- 不让 MKL 动态减少线程数
- 让 OpenMP 线程尽量分散到不同核心
- 线程绑定到 core 粒度，减少来回漂移

### 5.2 Streaming working set

主参数是：

- `--streaming-mb`

默认值：

- `1024`

含义：

- 希望总权重 bank 大小接近 1 GB

池子大小按下面这个逻辑自动计算：

`pool_size = ceil(streaming_target_bytes / single_weight_bytes)`

并限制在：

- `min_pool_size = 8`
- `max_pool_size = 1024`

例如：

- `[4096,4096]` 单个权重大约 `16 MB`，所以 `pool_size = 64`
- `[4096,16384]` 单个权重大约 `64 MB`，所以 `pool_size = 16`
- `[1024,1024]` 单个权重只有 `1 MB`，所以 `pool_size = 1024`

### 5.3 NUMA / 虚拟内存设置

这台机器实际环境是：

- CPU: `Intel Xeon Gold 6544Y`
- `2` 个 socket
- `64` 个逻辑 CPU
- `2` 个 NUMA node
- `/proc/sys/kernel/numa_balancing = 1`

系统里没有 `numactl`，所以这里没有走外部命令绑核/绑内存，而是在 Python 进程里做了两件事：

1. 读取 `/sys/devices/system/node/node*/cpulist`
2. 在创建每一块权重矩阵前，轮流把进程 affinity 切到不同 node 的 CPU 集上

这相当于在构造 weight bank 时做 **交替 first-touch**，尽量让页分布不要全部落到单个 socket。

如果你想对照“关闭这部分”的效果，可以加：

```bash
--disable-numa-first-touch
```

### 5.4 功耗和能效

功耗通过 Linux RAPL sysfs 读取：

- `/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj`
- `/sys/class/powercap/intel-rapl/intel-rapl:1/energy_uj`

输出指标包括：

- `latency_ms`
- `power_total_w`
- `package0_w`
- `package1_w`
- `energy_mj`
- `tops_per_w`

其中：

- `energy_mj = power_total_w * latency_sec * 1000`
- `TOPS = 2 * M * K * N / latency / 1e12`

## 6. 运行方式

### 6.1 环境

```bash
conda activate base
cd <REMOTE_HOME>/A-MICRO-CPU-GPU-Analysis/GEMV-CPU
```

### 6.2 快速校验

```bash
python3 benchmark_gemv_cpu.py \
  --warmup-sec 0.2 \
  --measure-sec 0.2 \
  --streaming-mb 256 \
  --verify \
  --out-dir results_verify
```

### 6.3 正式测量

```bash
python3 benchmark_gemv_cpu.py \
  --warmup-sec 3 \
  --measure-sec 10 \
  --streaming-mb 1024 \
  --out-dir results
```

## 7. 正式测试结果

测试设置：

- 时间：`2026-03-28`
- 线程数：`64`
- warmup：`3 s`
- measurement：`10 s`
- streaming working set target：`1024 MB`
- NUMA first-touch：开启
- power source：RAPL sysfs

| Name | Shape | Latency (ms) | Total Power (W) | Socket0 (W) | Socket1 (W) | Energy (mJ) | TOPS/W | Single Weight (MB) | Working Set (MB) | Pool |
|------|-------|--------------|-----------------|-------------|-------------|-------------|--------|--------------------|------------------|------|
| `GEMV_1024x1024` | `[1,1024]x[1024,1024]` | 0.0909 | 409.646 | 207.828 | 201.818 | 37.2485 | 0.000056 | 1.0 | 1024.0 | 1024 |
| `GEMV_2048x2048` | `[1,2048]x[2048,2048]` | 0.1290 | 421.919 | 212.709 | 209.211 | 54.4429 | 0.000154 | 4.0 | 1024.0 | 256 |
| `GEMV_4096x4096` | `[1,4096]x[4096,4096]` | 0.3688 | 459.080 | 228.648 | 230.432 | 169.3157 | 0.000198 | 16.0 | 1024.0 | 64 |
| `GEMV_4096x12288` | `[1,4096]x[4096,12288]` | 0.7244 | 501.624 | 249.827 | 251.797 | 363.3927 | 0.000277 | 48.0 | 1056.0 | 22 |
| `GEMV_4096x16384` | `[1,4096]x[4096,16384]` | 0.9841 | 525.814 | 259.770 | 266.045 | 517.4422 | 0.000259 | 64.0 | 1024.0 | 16 |
| `GEMV_8192x8192` | `[1,8192]x[8192,8192]` | 1.0792 | 482.806 | 239.609 | 243.196 | 521.0366 | 0.000258 | 64.0 | 1024.0 | 16 |

结果文件：

- `results/int8_gemv_cpu_energy.csv`
- `results/int8_gemv_cpu_energy.json`

验证文件：

- `results_verify/int8_gemv_cpu_energy.csv`
- `results_verify/int8_gemv_cpu_energy.json`

## 8. 怎么理解这些结果

有几个现象是正常的：

1. **shape 变大后 latency 明显上升**
   这说明现在的测试不再只是“算子本体热点循环”，而是已经把更多权重加载成本带进来了。
2. **功耗随着更大的权重和更重的流式访问而上升**
   尤其是 `4096x12288` 和 `4096x16384`，两路 socket 都更接近高负载。
3. **`4096x16384` 和 `8192x8192` 的单个权重大小都约 64 MB**
   两者 working set 都约 1 GB，所以 latency 和能耗接近是合理的；这时瓶颈已经不只是算术量，也和权重流式读取强相关。
4. **小 GEMV 的 TOPS/W 非常低**
   这不是 bug，而是 CPU 在 batch=1 decode 下的典型特征：算子小、线程多、算力利用率低，但双路 package 仍然维持较高功耗。

## 9. 和原 `intel-6544Y-CPU` decode 数据的关系

如果你对照原目录 README，会发现旧的 decode latency 更小，比如 `[1,4096]x[4096,4096]` 原来大约 `0.060 ms`，而这里是 `0.3688 ms`。这正是因为这里有意把 memory-bound 因素加进来了：

1. 不再重复打同一块权重
2. 不再让权重长期停留在更有利的 cache / page 状态
3. 尽量逼近“更大的权重空间中流式取下一层/下一块矩阵”的 decode 场景

所以这个目录的数据更适合回答：

- GEMV 在 CPU 上到底有多 memory-bound
- shape 变化后 latency / power / energy efficiency 为什么会一起变化
- 双路系统里 NUMA 和权重工作集大小会如何影响 decode 阶段

## 10. 一句话总结

`intel-6544Y-CPU` 是原始 CPU baseline；`GEMV-CPU` 是在它的 MKL INT8 路径上，专门为 **memory-aware GEMV** 重新搭的一套测试。它不是只测算力峰值，而是把 **load 权重、工作集、NUMA 和 sustained power** 一起纳入了统计。
