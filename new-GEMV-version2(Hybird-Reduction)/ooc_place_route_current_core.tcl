if {$argc < 2 || $argc > 3} {
    error "usage: ooc_place_route_current_core.tcl <opt_dcp> <output_dir> ?audit|place|full?"
}

set opt_dcp [file normalize [lindex $argv 0]]
set output_dir [file normalize [lindex $argv 1]]
set stop_stage [expr {$argc == 3 ? [lindex $argv 2] : "full"}]

if {$stop_stage ni {audit place full}} {
    error "invalid stop stage '$stop_stage': expected audit, place, or full"
}
if {![file exists $opt_dcp]} {
    error "optimized W4A4 core checkpoint does not exist: $opt_dcp"
}

file mkdir $output_dir
puts "W4A4_CORE_RESUME: opening $opt_dcp"
open_checkpoint $opt_dcp

set clocks [get_clocks -quiet ap_clk]
if {[llength $clocks] != 1} {
    error "expected one ap_clk clock, found [llength $clocks]"
}
set clock_period [get_property PERIOD [lindex $clocks 0]]
if {abs(double($clock_period) - 5.0) > 0.001} {
    error "expected ap_clk period 5.000 ns, found $clock_period ns"
}

set expected_pblock_cells [dict create \
    pblock_w4a4_slr0 29490 \
    pblock_w4a4_slr1 26861 \
    pblock_w4a4_slr2 29490]
set audit_file [file join $output_dir checkpoint_audit.rpt]
set audit_handle [open $audit_file w]
puts $audit_handle "W4A4 6416-DSP compute-core checkpoint audit"
puts $audit_handle "checkpoint=$opt_dcp"
puts $audit_handle "part=[get_property PART [current_project]]"
puts $audit_handle "clock=ap_clk period_ns=$clock_period"

dict for {pblock_name expected_cells} $expected_pblock_cells {
    set pblock [get_pblocks -quiet $pblock_name]
    if {[llength $pblock] != 1} {
        close $audit_handle
        error "expected one $pblock_name, found [llength $pblock]"
    }
    set direct_cells [llength [get_cells -quiet -of_objects $pblock]]
    if {$direct_cells != $expected_cells} {
        close $audit_handle
        error "$pblock_name contains $direct_cells direct cells; expected $expected_cells"
    }
    puts $audit_handle \
        "$pblock_name range=[get_property GRID_RANGES $pblock] direct_cells=$direct_cells is_soft=[get_property IS_SOFT $pblock]"
    puts "W4A4_CORE_RESUME: verified $pblock_name cells=$direct_cells"
}
close $audit_handle

report_timing_summary -delay_type max -max_paths 10 \
    -file [file join $output_dir timing_summary_opt_restored.rpt]
if {$stop_stage eq "audit"} {
    puts "W4A4_CORE_RESUME: checkpoint audit completed"
    exit
}

puts "W4A4_CORE_RESUME: place_design starts"
place_design -directive SSI_SpreadLogic_high
write_checkpoint -force \
    [file join $output_dir W4A4_P_6416dsp_placed_raw.dcp]
report_utilization \
    -file [file join $output_dir utilization_placed_raw.rpt]
report_timing_summary -delay_type min_max -max_paths 20 \
    -file [file join $output_dir timing_summary_placed_raw.rpt]
report_design_analysis -congestion \
    -file [file join $output_dir congestion_placed_raw.rpt]

puts "W4A4_CORE_RESUME: pre-route phys_opt_design starts"
phys_opt_design -directive AggressiveExplore
write_checkpoint -force \
    [file join $output_dir W4A4_P_6416dsp_placed.dcp]
report_utilization \
    -file [file join $output_dir utilization_placed.rpt]
report_timing_summary -delay_type min_max -max_paths 20 \
    -file [file join $output_dir timing_summary_placed.rpt]
report_design_analysis -congestion \
    -file [file join $output_dir congestion_placed.rpt]

if {$stop_stage eq "place"} {
    puts "W4A4_CORE_RESUME: completed through place_design"
    exit
}

puts "W4A4_CORE_RESUME: route_design starts"
route_design -directive AggressiveExplore
write_checkpoint -force \
    [file join $output_dir W4A4_P_6416dsp_implemented.dcp]
report_utilization \
    -file [file join $output_dir utilization_implemented.rpt]
report_utilization -hierarchical -hierarchical_depth 6 \
    -file [file join $output_dir utilization_hierarchical_implemented.rpt]
report_utilization -slr \
    -file [file join $output_dir utilization_slr_implemented.rpt]
report_timing_summary -delay_type min_max -max_paths 20 \
    -file [file join $output_dir timing_summary_implemented.rpt]
report_clock_utilization \
    -file [file join $output_dir clock_utilization_implemented.rpt]
report_route_status \
    -file [file join $output_dir route_status.rpt]
set routed_fully \
    [report_route_status -boolean_check ROUTED_FULLY -ignore_cache]
set errors_in_routes \
    [report_route_status -boolean_check ERRORS_IN_ROUTES -ignore_cache]
if {!$routed_fully || $errors_in_routes} {
    error "Route acceptance failed: routed_fully=$routed_fully errors_in_routes=$errors_in_routes"
}
report_design_analysis -congestion \
    -file [file join $output_dir congestion_implemented.rpt]
report_drc \
    -file [file join $output_dir drc.rpt]
set blocking_drc [get_drc_violations -quiet -filter \
    {SEVERITY == "Error" || SEVERITY == "Critical Warning"}]
if {[llength $blocking_drc] != 0} {
    error "Blocking DRC violations remain: [llength $blocking_drc]"
}
check_timing -verbose \
    -file [file join $output_dir check_timing.rpt]

set setup_paths [get_timing_paths -quiet -delay_type max -max_paths 1 -nworst 1]
if {[llength $setup_paths] > 0} {
    puts "W4A4_CORE_RESUME: routed WNS is [get_property SLACK [lindex $setup_paths 0]] ns"
}
puts "W4A4_CORE_RESUME: completed through route_design"
exit
