set_param general.maxThreads 8

set script_dir [file dirname [file normalize [info script]]]
set report_dir [file join $script_dir results pair_add_lut6_2]
file mkdir $report_dir

read_verilog -sv [file join $script_dir test_pair_add_variants.v]
synth_design -top pair_add_lut6_2 \
    -part xcu55c-fsvh2892-2L-e -mode out_of_context \
    -flatten_hierarchy none
opt_design

report_utilization \
    -file [file join $report_dir utilization_opt.rpt]
report_drc \
    -file [file join $report_dir drc_opt.rpt]
write_checkpoint -force \
    [file join $report_dir pair_add_lut6_2_opt.dcp]
