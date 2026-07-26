set_param general.maxThreads 8

set script_dir [file dirname [file normalize [info script]]]
set root_dir [file dirname $script_dir]
set report_dir [file join $script_dir results simd_chain_lut6_2_pair_v4_ooc]
file mkdir $report_dir

read_verilog -sv [file join $root_dir src W4A4_P.v]
read_verilog -sv [file join $root_dir src W4A4_P_wrapper.v]

synth_design -top W4A4_P -part xcu55c-fsvh2892-2L-e \
    -mode out_of_context -directive sdx_optimization_effort_high
create_clock -name ap_clk -period 5.000 [get_ports ap_clk]

write_checkpoint -force [file join $report_dir W4A4_P_simd_chain_lut6_2_pair_synth.dcp]
report_utilization \
    -file [file join $report_dir utilization_synth.rpt]
report_utilization -hierarchical -hierarchical_depth 6 \
    -file [file join $report_dir utilization_hierarchical_synth.rpt]
report_timing_summary -delay_type max -max_paths 20 \
    -file [file join $report_dir timing_summary_synth.rpt]

opt_design -directive Explore
source [file join $root_dir ooc_slr_balance.tcl]
apply_w4a4_slr_balance

write_checkpoint -force [file join $report_dir W4A4_P_simd_chain_lut6_2_pair_opt.dcp]
report_utilization \
    -file [file join $report_dir utilization_opt.rpt]
report_utilization -hierarchical -hierarchical_depth 6 \
    -file [file join $report_dir utilization_hierarchical_opt.rpt]
report_utilization -slr \
    -file [file join $report_dir utilization_slr_opt.rpt]
report_timing_summary -delay_type max -max_paths 20 \
    -file [file join $report_dir timing_summary_opt.rpt]
report_timing -delay_type max -max_paths 20 -sort_by group \
    -file [file join $report_dir timing_paths_opt.rpt]
report_drc \
    -file [file join $report_dir drc_opt.rpt]
report_w4a4_slr_balance \
    [file join $report_dir slr_balance_opt.rpt]
