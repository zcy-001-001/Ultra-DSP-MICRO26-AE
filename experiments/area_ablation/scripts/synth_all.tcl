set script_dir [file dirname [file normalize [info script]]]
set root_dir [file normalize [file join $script_dir ".."]]
set package_root [file normalize [file join $root_dir ".." ".."]]

set part "xcu55c-fsvh2892-2L-e"
set out_dir [file join $package_root "results" "area_ablation" "vivado_resource"]

if {[llength $argv] >= 1} {
    set part [lindex $argv 0]
}
if {[llength $argv] >= 2} {
    set out_dir [lindex $argv 1]
}

set rtl_dirs [list \
    [file join $root_dir "rtl" "optimization_stages"] \
    [file join $root_dir "rtl" "lsb_depth_sweep"] \
]

set rtl_files {}
foreach rtl_dir $rtl_dirs {
    foreach rtl_file [glob -nocomplain [file join $rtl_dir "*.v"]] {
        lappend rtl_files $rtl_file
    }
}
set rtl_files [lsort $rtl_files]

if {[llength $rtl_files] == 0} {
    puts "ERROR: no generated RTL files found. Run scripts/generate_area_ablation.py first."
    exit 1
}

file mkdir $out_dir
puts "Part: $part"
puts "Output: $out_dir"
puts "Design count: [llength $rtl_files]"

foreach rtl_file $rtl_files {
    set top [file rootname [file tail $rtl_file]]
    puts "=== Synthesizing $top ==="
    create_project -in_memory -part $part
    read_verilog $rtl_file
    synth_design -top $top -part $part -directive AreaOptimized_high -control_set_opt_threshold 1
    report_utilization -file [file join $out_dir "${top}_utilization_synth.rpt"]
    report_timing_summary -file [file join $out_dir "${top}_timing_synth.rpt"]
    close_project
}
