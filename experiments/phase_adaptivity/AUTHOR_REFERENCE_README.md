# ReviewD Phase Adaptivity 64x64 200MHz 实验记录

本文档是新增记录文件，未覆盖原有 README。实验目录：

`<REMOTE_WORKSPACE>/A-MICRO26-DSP-Packing`

## 目录说明

- `W4A4_P3x3_only_64_64/`：ReviewD 的 P-only 对照设计。PE 只保留 3x3 P layout；prefilling 阶段可使用 3x3 的 9 个乘法位置，decoding 阶段模式信号保留但只能有效使用 3 个 `a1` lane。
- `W4A4_D1x7_only_64_64/`：ReviewD 的 D-only 对照设计。PE 只保留 1x7 D layout；prefilling 和 decoding 阶段都使用 1x7 的 7 个位置。
- 两个目录都从 `W4A4_PD_64_64/` 派生，保留了 `*_PD_reference.*` 作为原始 PD 版本参考；原始 baseline 未修改。
- 新增/保留功能测试：`src/tb_reviewd_functional.cpp` 和 `src/tb_reviewd_rtl_single_pe.v`。

## 设计约束

- 阵列规模：64 x 64，4096 DSP。
- 目标频率：200 MHz。
- HLS/implementation 并行度配置：`config/hls.cfg` 和 `hardware_implement_200.sh` 使用 48 jobs。
- Blackbox 面积模型：
  - P-only 3x3：59 LUT / 60 FF per PE，总计 241664 LUT / 245760 FF / 4096 DSP。
  - D-only 1x7：42 LUT / 46 FF per PE，总计 172032 LUT / 188416 FF / 4096 DSP。
- Power 口径：Vivado 2023.2 post-route `report_power` vector-less 估计；两种设计使用同一方式从 routed DCP 生成。报告 confidence 为 Low，因为没有 SAIF/VCD activity file。

## 主要结果

| 设计 | xclbin | WNS (ns) | TNS (ns) | WHS (ns) | Total Power (W) | FPGA Power (W) | HBM Power (W) | Dynamic (W) | Static (W) | `gemv_kernel_1` Power (W) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P-only 3x3 | 68 MB | 0.003 | 0.000 | 0.009 | 36.353 | 25.256 | 11.097 | 32.167 | 4.186 | 4.141 |
| D-only 1x7 | 69 MB | 0.003 | 0.000 | 0.009 | 36.018 | 24.922 | 11.096 | 31.842 | 4.176 | 3.750 |

D-only 相对 P-only：

- Total On-Chip Power 低 0.335 W，约 0.92%。
- `gemv_kernel_1` 层级 power 低 0.391 W，约 9.44%。
- HBM/platform power 基本相同，差异主要来自 kernel logic。

## 资源结果

| 设计 | Kernel LUT | Kernel LUTAsMem | Kernel REG | Kernel BRAM | Kernel URAM | Kernel DSP |
|---|---:|---:|---:|---:|---:|---:|
| P-only 3x3 | 323194 | 33877 | 263431 | 24 | 0 | 4096 |
| D-only 1x7 | 233637 | 29781 | 205052 | 23 | 0 | 4096 |

Full design utilization（包含平台）：

| 设计 | CLB LUTs | CLB Registers | Block RAM Tile | DSPs |
|---|---:|---:|---:|---:|
| P-only 3x3 | 454815 | 447403 | 226.5 | 4100 |
| D-only 1x7 | 365304 | 389022 | 225.5 | 4100 |

## 关键报告路径

P-only：

- `W4A4_P3x3_only_64_64/implement_200/GEMV.xclbin`
- `W4A4_P3x3_only_64_64/implement_200/logs/implement.stdout.log`
- `W4A4_P3x3_only_64_64/implement_200/reports/post_route_power.rpt`
- `W4A4_P3x3_only_64_64/implement_200/reports/post_route_timing_summary.rpt`
- `W4A4_P3x3_only_64_64/_x/reports/link/imp/impl_1_kernel_util_routed.rpt`

D-only：

- `W4A4_D1x7_only_64_64/implement_200/GEMV.xclbin`
- `W4A4_D1x7_only_64_64/implement_200/logs/implement.stdout.log`
- `W4A4_D1x7_only_64_64/implement_200/reports/post_route_power.rpt`
- `W4A4_D1x7_only_64_64/implement_200/reports/post_route_timing_summary.rpt`
- `W4A4_D1x7_only_64_64/_x/reports/link/imp/impl_1_kernel_util_routed.rpt`

## Reproduction

在任一实验子目录下执行：

```bash
source ./env.sh
```

功能测试：

```bash
INC=<VITIS_HLS_ROOT>/include
g++ -std=c++17 -O2 -Wno-unknown-pragmas -I${INC} -Isrc \
  src/GEMV.cpp src/Hybrid_INT4_INT4_PD.cpp src/tb_reviewd_functional.cpp \
  -o reviewd_functional_check_rebuilt
./reviewd_functional_check_rebuilt
```

HLS synthesis：

```bash
bash csynth_200.sh
```

Implementation：

```bash
AUTO_CSYNTH=0 VIVADO_SYN_JOBS=48 VIVADO_IMPL_JOBS=48 bash hardware_implement_200.sh
```

Post-route power/timing/utilization 报告：

```bash
mkdir -p implement_200/reports
cat > implement_200/reports/post_route_power_only.tcl <<'TCL'
open_checkpoint _x/link/vivado/vpl/prj/prj.runs/impl_1/level0_wrapper_routed.dcp
report_power -file implement_200/reports/post_route_power.rpt
report_timing_summary -file implement_200/reports/post_route_timing_summary.rpt
report_utilization -file implement_200/reports/post_route_utilization.rpt
exit
TCL
vivado -mode batch -nojournal -nolog -notrace \
  -source implement_200/reports/post_route_power_only.tcl
```

## 已完成验证

- P-only C functional rebuild/run：`ReviewD P-only functional check passed`
- D-only C functional rebuild/run：`ReviewD D-only functional check passed`
- P-only implementation：`GEMV.xclbin` 生成成功，v++ 总耗时 3h35m31s。
- D-only implementation：`GEMV.xclbin` 生成成功，v++ 总耗时 3h07m47s。
- 两个设计 post-route timing 均满足 200 MHz 约束。

当前非交互环境中 `iverilog/vvp` 不在 PATH，因此打包后的最终验证使用 C-level functional test 从源码重新编译运行。RTL single-PE smoke testbench 已保留在 `src/tb_reviewd_rtl_single_pe.v`。
