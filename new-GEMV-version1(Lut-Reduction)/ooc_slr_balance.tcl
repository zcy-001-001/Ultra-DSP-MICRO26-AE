# Keep every multiplier column and its nine reduction trees in one SLR. The
# middle SLR receives two fewer columns because the AXI/dataflow wrapper tends
# to be placed there as well.
proc w4a4_slr_for_column {column} {
    if {$column < 22} {
        return SLR0
    }
    if {$column < 42} {
        return SLR1
    }
    return SLR2
}

proc w4a4_append_slr_cell {array_name column cell} {
    upvar 1 $array_name slr_cells
    set slr [w4a4_slr_for_column $column]
    lappend slr_cells($slr) $cell
}

proc apply_w4a4_slr_balance {} {
    array set slr_cells {SLR0 {} SLR1 {} SLR2 {}}

    set core_re \
        {.*core_col_gen\[([0-9]+)\]\.core_row_gen\[[0-9]+\]\.dsp_core$}
    set core_cells [get_cells -quiet -hier -regexp $core_re]
    if {[llength $core_cells] != 4096} {
        error "Expected 4096 multiplier cores, found [llength $core_cells]"
    }
    foreach cell $core_cells {
        if {![regexp $core_re $cell -> column]} {
            error "Unable to recover a column index from $cell"
        }
        w4a4_append_slr_cell slr_cells $column $cell
    }

    # The first explicit LUT/CARRY8 pair level remains a flat generated array.
    # Its node index is tree_index * 32 + row_pair.
    set level32_re {.*level32_gen\[([0-9]+)\]\.[^/]+$}
    foreach cell [get_cells -quiet -hier -regexp $level32_re] {
        if {![regexp $level32_re $cell -> node_index]} {
            error "Unable to recover a level32 node index from $cell"
        }
        set tree_index [expr {$node_index / 32}]
        set column [expr {$tree_index % 64}]
        w4a4_append_slr_cell slr_cells $column $cell
    }

    # Each of the 576 pure-LUT trees has its own preserved hierarchy. The tree
    # index is channel * 64 + column, so every registered reduction level is
    # constrained to the same SLR as its multiplier column.
    set pure_tree_re {.*pure_tree_gen\[([0-9]+)\]\.tree$}
    set pure_tree_cells [get_cells -quiet -hier -regexp $pure_tree_re]
    if {[llength $pure_tree_cells] != 576} {
        error "Expected 576 independent pure-LUT trees, found [llength $pure_tree_cells]"
    }
    foreach cell $pure_tree_cells {
        if {![regexp $pure_tree_re $cell -> tree_index]} {
            error "Unable to recover a pure-LUT tree index from $cell"
        }
        set column [expr {$tree_index % 64}]
        w4a4_append_slr_cell slr_cells $column $cell
    }

    # The weight sign pipeline is outside the multiplier hierarchy. Assign its
    # registers with the corresponding PE column so the first LUT stage does
    # not leave an SLR and then cross back to its own reduction tree.
    # Vivado can collapse the two-cycle sign pipeline into one SRL16E named
    # weight_signs_r1_reg[N]_srl2. Match both that form and the uncollapsed FF
    # names produced by the standalone array synthesis.
    set weight_sign_re \
        {.*weight_signs_r[12]_reg\[([0-9]+)\](_srl[0-9]+)?$}
    set weight_sign_cells \
        [get_cells -quiet -hier -regexp $weight_sign_re]
    set weight_sign_count [llength $weight_sign_cells]
    if {$weight_sign_count ni {12288 24576}} {
        error "Expected 12288 SRLs or 24576 FFs in the weight-sign pipeline, found $weight_sign_count"
    }
    foreach cell $weight_sign_cells {
        if {![regexp $weight_sign_re $cell -> sign_index]} {
            error "Unable to recover a weight-sign index from $cell"
        }
        set pe_index [expr {$sign_index % 4096}]
        set column [expr {$pe_index / 64}]
        w4a4_append_slr_cell slr_cells $column $cell
    }

    # Activation signs and decoders have one deliberately preserved copy per
    # SLR. Map each bank to the same SLR as the columns it drives.
    set activation_sign_re \
        {.*activation_signs_r[12]_reg\[([0-9]+)\](_srl[0-9]+)?$}
    set activation_sign_cells \
        [get_cells -quiet -hier -regexp $activation_sign_re]
    set activation_sign_count [llength $activation_sign_cells]
    if {$activation_sign_count ni {0 576 1152}} {
        error "Expected 0 optimized-away cells, 576 SRLs, or 1152 activation-sign FFs, found $activation_sign_count"
    }
    if {$activation_sign_count == 0} {
        puts "OOC_SLR_BALANCE: activation-sign FFs were absorbed by full-kernel synthesis"
    }
    foreach cell $activation_sign_cells {
        if {![regexp $activation_sign_re $cell -> sign_index]} {
            error "Unable to recover an activation-sign index from $cell"
        }
        set slr_index [expr {$sign_index / 192}]
        lappend slr_cells(SLR${slr_index}) $cell
    }

    set activation_decode_re \
        {.*act_slr_gen\[([0-9]+)\]\.act_row_gen\[[0-9]+\]\.decode_a[123]$}
    set activation_decode_cells \
        [get_cells -quiet -hier -regexp $activation_decode_re]
    if {[llength $activation_decode_cells] != 576} {
        error "Expected 576 SLR-local activation decoders, found [llength $activation_decode_cells]"
    }
    foreach cell $activation_decode_cells {
        if {![regexp $activation_decode_re $cell -> slr_index]} {
            error "Unable to recover an activation-decoder SLR from $cell"
        }
        lappend slr_cells(SLR${slr_index}) $cell
    }

    foreach slr {SLR0 SLR1 SLR2} {
        if {![llength $slr_cells($slr)]} {
            error "No W4A4 cells were selected for $slr"
        }
        set pblock_name "pblock_w4a4_[string tolower $slr]"
        if {[llength [get_pblocks -quiet $pblock_name]]} {
            error "Pblock already exists: $pblock_name"
        }
        create_pblock $pblock_name
        resize_pblock [get_pblocks $pblock_name] -add $slr
        add_cells_to_pblock [get_pblocks $pblock_name] $slr_cells($slr)
        puts "OOC_SLR_BALANCE: $slr pblock contains [llength $slr_cells($slr)] selected cells"
    }
}

proc report_w4a4_slr_balance {report_file} {
    set report_handle [open $report_file w]
    puts $report_handle "W4A4 OOC SLR constraint summary"

    foreach pblock [lsort [get_pblocks -quiet pblock_w4a4_*]] {
        puts $report_handle [format \
            "%s range=%s direct_cells=%d" \
            $pblock \
            [get_property GRID_RANGES $pblock] \
            [llength [get_cells -quiet -of_objects $pblock]]]
    }

    set weight_sign_re \
        {.*weight_signs_r[12]_reg\[([0-9]+)\](_srl[0-9]+)?$}
    puts $report_handle [format \
        "weight_sign_pipeline_cells=%d" \
        [llength [get_cells -quiet -hier -regexp $weight_sign_re]]]
    puts $report_handle [format \
        "cells_with_LOC=%d" \
        [llength [get_cells -quiet -hier -filter {LOC != {}}]]]
    close $report_handle
}
