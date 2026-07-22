set script_dir [file dirname [file normalize [info script]]]
set root_dir [file normalize [file join $script_dir ".."]]
set package_root [file normalize [file join $root_dir ".." ".."]]

if {[llength $argv] < 2} {
    puts "Usage: vivado -mode batch -source scripts/synth_one.tcl -tclargs <top_module> <rtl_file> ?part? ?out_dir?"
    puts "Example: vivado -mode batch -source scripts/synth_one.tcl -tclargs W4A4_stage_s3_xor_sign rtl/optimization_stages/W4A4_stage_s3_xor_sign.v"
    exit 1
}

set top [lindex $argv 0]
set rtl_file [lindex $argv 1]
set part "xcu55c-fsvh2892-2L-e"
set out_dir [file join $package_root "results" "area_ablation" "vivado_resource"]

if {[llength $argv] >= 3} {
    set part [lindex $argv 2]
}
if {[llength $argv] >= 4} {
    set out_dir [lindex $argv 3]
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

create_project -in_memory -part $part
read_verilog $rtl_file
synth_design -top $top -part $part -directive AreaOptimized_high -control_set_opt_threshold 1
report_utilization -file [file join $out_dir "${top}_utilization_synth.rpt"]
report_timing_summary -file [file join $out_dir "${top}_timing_synth.rpt"]
close_project
