set script_dir [file dirname [file normalize [info script]]]
set ae_root [file normalize [file join $script_dir ".." ".." ".."]]
set part_name "xcu55c-fsvh2892-2L-e"

if {[info exists ::env(FIGURE17_OUT_DIR)]} {
    set out_root [file normalize $::env(FIGURE17_OUT_DIR)]
} else {
    set out_root [file join $ae_root "results" "figure17" "rerun_reports"]
}
file mkdir [file join $out_root "component"]
file mkdir [file join $out_root "area"]

set component_root [file join $ae_root "experiments" "component_ablation" "verilog" "resource_targets"]
set component_common [file join $component_root "w4a4_dsp48e2_m_sub_c.v"]
set component_designs [list \
    [list "w4a4_sf_v0_normal_signed_p2d2_single_dsp" \
        [list [file join $component_root "w4a4_sf_v0_normal_signed_p2d2_single_dsp.v"]]] \
    [list "w4a4_sf_v1_signmag_nonoverlap_p6d5_single_dsp" \
        [list [file join $component_root "w4a4_sf_v1_signmag_nonoverlap_p6d5_single_dsp.v"] $component_common]] \
    [list "w4a4_sf_v2_overlap_no_correction_p8d6_single_dsp" \
        [list [file join $component_root "w4a4_sf_v2_overlap_no_correction_p8d6_single_dsp.v"] $component_common]] \
    [list "w4a4_sf_v3_full_correction_p8d6_single_dsp" \
        [list [file join $component_root "w4a4_sf_v3_full_correction_p8d6_single_dsp.v"] $component_common]] \
]

foreach design $component_designs {
    set top [lindex $design 0]
    set files [lindex $design 1]
    puts "=== COMPONENT $top ==="
    create_project -in_memory -part $part_name
    foreach rtl_file $files { read_verilog $rtl_file }
    synth_design -top $top -part $part_name -directive AreaOptimized_high -control_set_opt_threshold 1
    report_utilization -file [file join $out_root "component" "${top}_utilization_synth.rpt"]
    report_timing_summary -file [file join $out_root "component" "${top}_timing_synth.rpt"]
    close_project
}

set area_root [file join $ae_root "experiments" "area_ablation" "rtl" "optimization_stages"]
set area_files [lsort [glob -nocomplain [file join $area_root "*.v"]]]
if {[llength $area_files] != 24} {
    puts "ERROR: expected 24 Figure 17 optimization-stage RTL files, found [llength $area_files]"
    exit 2
}

foreach rtl_file $area_files {
    set top [file rootname [file tail $rtl_file]]
    puts "=== AREA $top ==="
    create_project -in_memory -part $part_name
    read_verilog $rtl_file
    synth_design -top $top -part $part_name -directive AreaOptimized_high -control_set_opt_threshold 1
    report_utilization -file [file join $out_root "area" "${top}_utilization_synth.rpt"]
    report_timing_summary -file [file join $out_root "area" "${top}_timing_synth.rpt"]
    close_project
}

puts "FIGURE17_VIVADO_SYNTH_PASS component=[llength $component_designs] area=[llength $area_files]"
exit 0
