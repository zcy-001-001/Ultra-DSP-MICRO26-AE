set script_dir [file dirname [file normalize [info script]]]
set root_dir [file normalize [file join $script_dir ".."]]
set package_dir [file normalize [file join $root_dir ".." ".."]]
set out_dir [file join $package_dir "results" "rerun" "component_ablation" "vivado_signmag_first"]
file mkdir $out_dir

# Match Ultra-DSP-main/Area_Test_Ablation_Test/Ablation Experiment.
set part_name "xcu55c-fsvh2892-2L-e"
set synth_directive "AreaOptimized_high"
set control_set_threshold 1

set jobs {
    {w4a4_sf_v0_normal_signed_p2d2_single_dsp {
        verilog/resource_targets/w4a4_sf_v0_normal_signed_p2d2_single_dsp.v
    }}
    {w4a4_sf_v1_signmag_nonoverlap_p6d5_single_dsp {
        verilog/resource_targets/w4a4_sf_v1_signmag_nonoverlap_p6d5_single_dsp.v
        verilog/resource_targets/w4a4_dsp48e2_m_sub_c.v
    }}
    {w4a4_sf_v2_overlap_no_correction_p8d6_single_dsp {
        verilog/resource_targets/w4a4_sf_v2_overlap_no_correction_p8d6_single_dsp.v
        verilog/resource_targets/w4a4_dsp48e2_m_sub_c.v
    }}
    {w4a4_sf_v3_full_correction_p8d6_single_dsp {
        verilog/resource_targets/w4a4_sf_v3_full_correction_p8d6_single_dsp.v
        verilog/resource_targets/w4a4_dsp48e2_m_sub_c.v
    }}
}

# The historical W4A4_PD_End job used a source outside the released closure.
# It is intentionally not rerun here; its archived result remains available in
# results/component_ablation/. Figure 17 uses the released optimized RTL path.

foreach job $jobs {
    set top [lindex $job 0]
    set sources [lindex $job 1]
    set util_report [file join $out_dir "${top}_utilization_synth.rpt"]
    set timing_report [file join $out_dir "${top}_timing_synth.rpt"]

    if {[file exists $util_report] && ![info exists ::env(FORCE_RESYNTH)]} {
        puts "=== Skipping $top; existing report found at $util_report ==="
        continue
    }

    puts "=== Synthesizing $top ==="
    create_project -in_memory -part $part_name
    set_property target_language Verilog [current_project]
    foreach source_raw $sources {
        set source_file [file normalize [file join $root_dir "$source_raw"]]
        read_verilog [list $source_file]
    }
    synth_design -top $top -part $part_name -directive $synth_directive -control_set_opt_threshold $control_set_threshold
    report_utilization -file $util_report
    report_timing_summary -file $timing_report
    close_project
}
