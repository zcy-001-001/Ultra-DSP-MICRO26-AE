set script_dir [file dirname [file normalize [info script]]]
set root_dir [file normalize [file join $script_dir ".."]]
set package_root [file normalize [file join $root_dir ".." ".."]]

set part "xcu55c-fsvh2892-2L-e"
set out_dir [file join $package_root "results" "overlap_depth_sweep" "vivado_resource"]
set summary_csv [file join $package_root "results" "overlap_depth_sweep" "vivado_resource_lut_ff.csv"]

if {[llength $argv] >= 1} {
    set part [lindex $argv 0]
}
if {[llength $argv] >= 2} {
    set out_dir [lindex $argv 1]
}
if {[llength $argv] >= 3} {
    set summary_csv [lindex $argv 2]
}

proc parse_util_metric {report_path label} {
    if {![file exists $report_path]} {
        return ""
    }
    set fp [open $report_path r]
    set text [read $fp]
    close $fp
    set pattern [format {\|[[:space:]]*%s\*?[[:space:]]*\|[[:space:]]*([0-9,]+)} $label]
    if {[regexp $pattern $text -> value]} {
        regsub -all "," $value "" value
        return $value
    }
    return ""
}

set rtl_dir [file join $root_dir "rtl"]
set rtl_files [lsort [glob -nocomplain [file join $rtl_dir "w4a4_overlap_depth*_hybrid.v"]]]

if {[llength $rtl_files] == 0} {
    puts "ERROR: no generated RTL files found. Run scripts/generate_overlap_depth_sweep.py first."
    exit 1
}

file mkdir $out_dir
puts "Part: $part"
puts "Output: $out_dir"
puts "Summary CSV: $summary_csv"
puts "Design count: [llength $rtl_files]"

set summary_fp [open $summary_csv w]
puts $summary_fp "top_module,part,synth_directive,control_set_opt_threshold,clb_luts,lut_as_logic,lut_as_memory,ff,clb_registers,dsp,utilization_report,timing_report"

foreach rtl_file $rtl_files {
    set top [file rootname [file tail $rtl_file]]
    set util_report [file join $out_dir "${top}_utilization_synth.rpt"]
    set timing_report [file join $out_dir "${top}_timing_synth.rpt"]
    puts "=== Synthesizing $top ==="
    create_project -in_memory -part $part
    set_property target_language Verilog [current_project]
    read_verilog $rtl_file
    # Match area-ablation and compent-ablation:
    #   xcu55c-fsvh2892-2L-e + AreaOptimized_high + control_set_opt_threshold 1
    synth_design -top $top -part $part -directive AreaOptimized_high -control_set_opt_threshold 1
    report_utilization -file $util_report
    report_timing_summary -file $timing_report

    set clb_luts [parse_util_metric $util_report "CLB LUTs"]
    set lut_as_logic [parse_util_metric $util_report "LUT as Logic"]
    set lut_as_memory [parse_util_metric $util_report "LUT as Memory"]
    set clb_registers [parse_util_metric $util_report "CLB Registers"]
    set dsp [parse_util_metric $util_report "DSPs"]
    puts $summary_fp "$top,$part,AreaOptimized_high,1,$clb_luts,$lut_as_logic,$lut_as_memory,$clb_registers,$clb_registers,$dsp,$util_report,$timing_report"
    flush $summary_fp
    close_project
}

close $summary_fp
puts "Wrote $summary_csv"
