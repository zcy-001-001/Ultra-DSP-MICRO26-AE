set script_dir [file dirname [file normalize [info script]]]
set root_dir [file normalize [file join $script_dir ".."]]
set package_root [file normalize [file join $root_dir ".." ".."]]

set manifest [file join $root_dir "manifest.csv"]
set out_dir [file join $package_root "results" "table8" "vivado_resource"]
set part "xcu55c-fsvh2892-2L-e"

if {[llength $argv] >= 1} {
    set manifest [lindex $argv 0]
}
if {[llength $argv] >= 2} {
    set out_dir [lindex $argv 1]
}
if {[llength $argv] >= 3} {
    set part [lindex $argv 2]
}

if {[file pathtype $manifest] eq "relative"} {
    set manifest [file normalize [file join $root_dir $manifest]]
}
if {[file pathtype $out_dir] eq "relative"} {
    set out_dir [file normalize [file join $root_dir $out_dir]]
}

if {![file exists $manifest]} {
    puts "ERROR: manifest not found: $manifest"
    exit 2
}

file mkdir $out_dir

set fh [open $manifest r]
set header [gets $fh]
set columns [split $header ","]
set top_idx [lsearch -exact $columns "top_module"]
set rtl_idx [lsearch -exact $columns "rtl_file"]

if {$top_idx < 0 || $rtl_idx < 0} {
    puts "ERROR: manifest must contain top_module and rtl_file columns"
    close $fh
    exit 3
}

while {[gets $fh line] >= 0} {
    if {[string trim $line] eq ""} {
        continue
    }
    set fields [split $line ","]
    set top [lindex $fields $top_idx]
    set rtl_file [lindex $fields $rtl_idx]
    if {[file pathtype $rtl_file] eq "relative"} {
        set rtl_file [file normalize [file join $root_dir $rtl_file]]
    }
    if {![file exists $rtl_file]} {
        puts "ERROR: RTL file not found: $rtl_file"
        close $fh
        exit 4
    }

    puts "=== Synthesizing $top ==="
    puts "RTL:  $rtl_file"
    puts "Part: $part"

    create_project -in_memory -part $part
    read_verilog $rtl_file
    synth_design -top $top -part $part -directive AreaOptimized_high -control_set_opt_threshold 1
    report_utilization -file [file join $out_dir "${top}_utilization_synth.rpt"]
    report_timing_summary -file [file join $out_dir "${top}_timing_synth.rpt"]
    close_project
}

close $fh
puts "Completed FP PE synthesis reports under $out_dir"
