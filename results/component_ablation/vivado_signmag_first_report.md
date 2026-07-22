# Vivado 资源报告：W4A4 组件消融

综合设置：
- Vivado 2024.1
- Part：`xcu55c-fsvh2892-2L-e`
- Directive：`AreaOptimized_high`
- 额外参数：`-control_set_opt_threshold 1`
- 计数口径：每一行是一套同时支持 P/D 的共享 PE，不把 P-only 和 D-only 两套硬件相加。

## 最终使用的 Active Rows

| Step | Top | LUT | FF | DSP | 说明 |
|---|---|---:|---:|---:|---|
| V0 | `w4a4_sf_v0_normal_signed_p2d2_single_dsp` | 11 | 0 | 1 | 二补码 Normal Signed Packing；P=4 effective，D=2；P 端把两个 activation 先相加后送入 B。 |
| V1 | `w4a4_sf_v1_signmag_nonoverlap_p6d5_single_dsp` | 40 | 27 | 1 | Online sign-magnitude，非重叠，P=6，D=5。 |
| V2 | `w4a4_sf_v2_overlap_no_correction_p8d6_single_dsp` | 49 | 33 | 1 | 手调 raw overlap，P=8，D=6，不加 correction。 |
| V3 | `w4a4_sf_v3_full_correction_p8d6_single_dsp` | 82 | 90 | 1 | 与 V2 相同 layout，加 full correction。 |
| V4 | `W4A4_PD_End` | 75 | 67 | 1 | 最终 Ultra-DSP，使用 ILP Layout。 |

## V0 修正说明

旧版本把 `WP521` 当作 Normal Signed Packing，但 `WP521` 会先取 magnitude 并单独处理 sign，本质上已经包含 sign/magnitude 分离，不适合作为 V0 baseline。

新的 V0 使用本目录 RTL：

`experiments/component_ablation/verilog/resource_targets/w4a4_sf_v0_normal_signed_p2d2_single_dsp.v`

其端口逻辑是：
- D 阶段：`B=a1`，`A=w1@0`，`D=w2@8`，得到两个 signed product。
- P 阶段：`B=a1+a2`，`A/D` 仍放两个 signed weights；物理输出是两个 fused lanes，但等效 MAC 数是 4，因此图中 P packing 计为 4。
- 资源统计：`a1+a2` 的 fabric pre-adder 写在当前 V0 RTL 内，因此 Vivado 得到的 11 LUT 已包含这个预加逻辑。
- 为避免低 lane 负数符号扩展污染高 lane，高 lane 使用 `lane1 = P[15:8] + P[7]` 的组合修正。

## 历史报告

`results\vivado_signmag_first\` 里仍保留一些历史报告，但不作为最终 active 表格使用：
- `WP521`：旧的 sign/magnitude 风格 baseline，已移出 active rows。
- `W4A4_sf_v0_normal_signed_p4d2`：旧 WP521 wrapper，已移出 active rows。
- `w4a4_sf_v2_overlap_no_correction_p10d7_single_dsp` 和 `w4a4_sf_v3_full_correction_density_first_p10d7_single_dsp`：过密中间 layout，不用于最终手调 ablation。
- `w4a4_v1_signmag_nonoverlap`、`w4a4_v2_overlap_no_correction`、`w4a4_v3_full_correction_density_first`：behavioral multiply 版本会被 Vivado 推成多 DSP，不用于最终资源口径。
