set script_dir [file dirname [file normalize [info script]]]
set root_dir [file normalize [file join $script_dir ".."]]
set package_root [file normalize [file join $root_dir ".." ".."]]

if {[llength $argv] < 2} {
    puts "Usage: vivado -mode batch -source scripts/synth_one.tcl -tclargs <top_module> <rtl_file> ?part? ?out_dir? ?summary_csv?"
    puts "Example: vivado -mode batch -source scripts/synth_one.tcl -tclargs w4a4_overlap_depth3_hybrid rtl/w4a4_overlap_depth3_hybrid.v"
    exit 1
}

set top [lindex $argv 0]
set rtl_file [lindex $argv 1]
set part "xcu55c-fsvh2892-2L-e"
set out_dir [file join $package_root "results" "overlap_depth_sweep" "vivado_resource"]
set summary_csv [file join $package_root "results" "overlap_depth_sweep" "vivado_resource_lut_ff.csv"]

if {[llength $argv] >= 3} {
    set part [lindex $argv 2]
}
if {[llength $argv] >= 4} {
    set out_dir [lindex $argv 3]
}
if {[llength $argv] >= 5} {
    set summary_csv [lindex $argv 4]
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

if {[file pathtype $rtl_file] eq "relative"} {
    set root_relative [file normalize [file join $root_dir $rtl_file]]
    set cwd_relative [file normalize [file join [pwd] $rtl_file]]
    if {[file exists $root_relative]} {
        set rtl_file $root_relative
    } else {
        set rtl_file $cwd_relative
    }
} else {
    set rtl_file [file normalize $rtl_file]
}

if {![file exists $rtl_file]} {
    puts "ERROR: RTL file not found: $rtl_file"
    exit 2
}

file mkdir $out_dir

puts "=== Synthesizing $top ==="
puts "RTL:  $rtl_file"
puts "Part: $part"
puts "Output: $out_dir"
puts "Summary CSV: $summary_csv"

create_project -in_memory -part $part
set_property target_language Verilog [current_project]
read_verilog $rtl_file
# Match area-ablation and compent-ablation:
#   xcu55c-fsvh2892-2L-e + AreaOptimized_high + control_set_opt_threshold 1
synth_design -top $top -part $part -directive AreaOptimized_high -control_set_opt_threshold 1
set util_report [file join $out_dir "${top}_utilization_synth.rpt"]
set timing_report [file join $out_dir "${top}_timing_synth.rpt"]
report_utilization -file $util_report
report_timing_summary -file $timing_report

set clb_luts [parse_util_metric $util_report "CLB LUTs"]
set lut_as_logic [parse_util_metric $util_report "LUT as Logic"]
set lut_as_memory [parse_util_metric $util_report "LUT as Memory"]
set clb_registers [parse_util_metric $util_report "CLB Registers"]
set dsp [parse_util_metric $util_report "DSPs"]
set write_header 0
if {![file exists $summary_csv] || [file size $summary_csv] == 0} {
    set write_header 1
}
set summary_fp [open $summary_csv a]
if {$write_header} {
    puts $summary_fp "top_module,part,synth_directive,control_set_opt_threshold,clb_luts,lut_as_logic,lut_as_memory,ff,clb_registers,dsp,utilization_report,timing_report"
}
puts $summary_fp "$top,$part,AreaOptimized_high,1,$clb_luts,$lut_as_logic,$lut_as_memory,$clb_registers,$clb_registers,$dsp,$util_report,$timing_report"
close $summary_fp
puts "Wrote/updated $summary_csv"
close_project
