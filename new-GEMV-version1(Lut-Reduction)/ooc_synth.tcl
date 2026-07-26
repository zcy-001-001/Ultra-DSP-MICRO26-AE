if {$argc < 3 || $argc > 4} {
    error "usage: ooc_synth.tcl <rtl_dir> <report_dir> <xdc_file> ?synth|opt|full?"
}

set rtl_dir [file normalize [lindex $argv 0]]
set report_dir [file normalize [lindex $argv 1]]
set xdc_file [file normalize [lindex $argv 2]]
set stop_stage [expr {$argc == 4 ? [lindex $argv 3] : "opt"}]
set part_name xcu55c-fsvh2892-2L-e

if {$stop_stage ni {synth opt full}} {
    error "invalid OOC stop stage '$stop_stage': expected synth, opt, or full"
}

if {![file isdirectory $rtl_dir]} {
    error "HLS RTL directory does not exist: $rtl_dir"
}
if {![file exists $xdc_file]} {
    error "OOC clock constraint does not exist: $xdc_file"
}

file mkdir $report_dir
create_project -in_memory -part $part_name
set_property target_language Verilog [current_project]
set_property XPM_LIBRARIES {XPM_MEMORY XPM_FIFO} [current_project]

set rtl_files [lsort [glob -nocomplain -directory $rtl_dir *.v]]
if {![llength $rtl_files]} {
    error "No generated Verilog files found in: $rtl_dir"
}

read_verilog $rtl_files
read_xdc $xdc_file

puts "OOC_IMPLEMENT: synth_design starts"
synth_design -top gemv_kernel -part $part_name -mode out_of_context \
    -directive sdx_optimization_effort_high

write_checkpoint -force [file join $report_dir gemv_kernel_ooc_synth.dcp]
report_utilization -file [file join $report_dir utilization_synth.rpt]
report_utilization -hierarchical -hierarchical_depth 5 \
    -file [file join $report_dir utilization_hierarchical_synth.rpt]
report_timing_summary -delay_type max -max_paths 10 \
    -file [file join $report_dir timing_summary_synth.rpt]

if {$stop_stage eq "synth"} {
    puts "OOC_IMPLEMENT: completed through synth_design"
    exit
}

puts "OOC_IMPLEMENT: opt_design starts"
opt_design -directive Explore

set slr_balance_tcl \
    [file join [file dirname [info script]] ooc_slr_balance.tcl]
if {![file exists $slr_balance_tcl]} {
    error "SLR balance script does not exist: $slr_balance_tcl"
}
source $slr_balance_tcl
apply_w4a4_slr_balance
report_w4a4_slr_balance \
    [file join $report_dir slr_balance_opt.rpt]
write_checkpoint -force [file join $report_dir gemv_kernel_ooc_opt.dcp]

if {$stop_stage eq "opt"} {
    report_utilization -file [file join $report_dir utilization_opt.rpt]
    report_utilization -hierarchical -hierarchical_depth 5 \
        -file [file join $report_dir utilization_hierarchical_opt.rpt]
    report_timing_summary -delay_type max -max_paths 10 \
        -file [file join $report_dir timing_summary_opt.rpt]
    puts "OOC_IMPLEMENT: completed through opt_design; place/route skipped"
    exit
}

puts "OOC_IMPLEMENT: place_design starts"
place_design -directive SSI_SpreadLogic_high
write_checkpoint -force \
    [file join $report_dir gemv_kernel_ooc_placed_raw.dcp]
report_utilization \
    -file [file join $report_dir utilization_placed_raw.rpt]
report_timing_summary -delay_type min_max -max_paths 10 \
    -file [file join $report_dir timing_summary_placed_raw.rpt]

phys_opt_design -directive AggressiveExplore
write_checkpoint -force [file join $report_dir gemv_kernel_ooc_placed.dcp]
report_utilization -file [file join $report_dir utilization_placed.rpt]
report_timing_summary -delay_type min_max -max_paths 10 \
    -file [file join $report_dir timing_summary_placed.rpt]

puts "OOC_IMPLEMENT: route_design starts"
route_design -directive AggressiveExplore
write_checkpoint -force \
    [file join $report_dir gemv_kernel_ooc_routed_raw.dcp]
report_route_status \
    -file [file join $report_dir route_status_raw.rpt]
set routed_fully \
    [report_route_status -boolean_check ROUTED_FULLY -ignore_cache]
set errors_in_routes \
    [report_route_status -boolean_check ERRORS_IN_ROUTES -ignore_cache]
if {!$routed_fully || $errors_in_routes} {
    error "Route acceptance failed: routed_fully=$routed_fully errors_in_routes=$errors_in_routes"
}
report_timing_summary -delay_type min_max -max_paths 10 \
    -file [file join $report_dir timing_summary_routed_raw.rpt]

set setup_paths [get_timing_paths -quiet -delay_type max -max_paths 1 -nworst 1]
if {[llength $setup_paths] > 0} {
    set routed_wns [get_property SLACK [lindex $setup_paths 0]]
    puts "OOC_IMPLEMENT: routed WNS is ${routed_wns} ns; timing closure is informational"
}

write_checkpoint -force [file join $report_dir gemv_kernel_ooc_implemented.dcp]
report_utilization -file [file join $report_dir utilization_implemented.rpt]
report_utilization -hierarchical -hierarchical_depth 5 \
    -file [file join $report_dir utilization_hierarchical_implemented.rpt]
report_timing_summary -delay_type min_max -max_paths 10 \
    -file [file join $report_dir timing_summary_implemented.rpt]
report_clock_utilization -file [file join $report_dir clock_utilization_implemented.rpt]
report_route_status -file [file join $report_dir route_status.rpt]
report_drc -file [file join $report_dir drc.rpt]
set blocking_drc [get_drc_violations -quiet -filter \
    {SEVERITY == "Error" || SEVERITY == "Critical Warning"}]
if {[llength $blocking_drc] != 0} {
    error "Blocking DRC violations remain: [llength $blocking_drc]"
}
check_timing -verbose -file [file join $report_dir check_timing.rpt]

puts "OOC_IMPLEMENT: completed through route_design"
exit
