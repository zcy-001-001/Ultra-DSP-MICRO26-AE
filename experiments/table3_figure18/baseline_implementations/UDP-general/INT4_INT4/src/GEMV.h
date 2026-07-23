#ifndef GEMV_H
#define GEMV_H

#include "ap_int.h"

#define DSP_ARRAY_ROWS 128
#define DSP_ARRAY_COLS 32
#define NUM_DSP (DSP_ARRAY_ROWS * DSP_ARRAY_COLS)

#define NUM_WEIGHT_AXI_PORTS 32
#define PACKED_WEIGHTS_PER_PE 3
#define ACTIVATION_PORTS_PER_PE 2
#define LOGICAL_OUTPUT_COLS (DSP_ARRAY_COLS * PACKED_WEIGHTS_PER_PE)
#define LOGICAL_COLS_PER_WEIGHT_PORT \
    (((LOGICAL_OUTPUT_COLS) + (NUM_WEIGHT_AXI_PORTS) - 1) / (NUM_WEIGHT_AXI_PORTS))

#define ACT_COUNT (ACTIVATION_PORTS_PER_PE * DSP_ARRAY_ROWS)
#define OUTPUT_COUNT (ACTIVATION_PORTS_PER_PE * LOGICAL_OUTPUT_COLS)
#define WEIGHT_PORT_DEPTH (LOGICAL_COLS_PER_WEIGHT_PORT * DSP_ARRAY_ROWS)

// act_packed layout:
//   [a0(rows), a1(rows)]
//
// output layout:
//   [sum_a0(logical_cols), sum_a1(logical_cols)]
//
// weight_colN layout:
//   up to LOGICAL_COLS_PER_WEIGHT_PORT logical columns.
//   Each logical column stores DSP_ARRAY_ROWS signed int4 weights in row-major order.
//   logical_col -> physical_col = logical_col / 3, lane = logical_col % 3.
extern "C" void gemv_kernel(
    ap_int<32> *output,
    const ap_int<4> *act_packed,
    const ap_int<4> *weight_col0,
    const ap_int<4> *weight_col1,
    const ap_int<4> *weight_col2,
    const ap_int<4> *weight_col3,
    const ap_int<4> *weight_col4,
    const ap_int<4> *weight_col5,
    const ap_int<4> *weight_col6,
    const ap_int<4> *weight_col7,
    const ap_int<4> *weight_col8,
    const ap_int<4> *weight_col9,
    const ap_int<4> *weight_col10,
    const ap_int<4> *weight_col11,
    const ap_int<4> *weight_col12,
    const ap_int<4> *weight_col13,
    const ap_int<4> *weight_col14,
    const ap_int<4> *weight_col15,
    const ap_int<4> *weight_col16,
    const ap_int<4> *weight_col17,
    const ap_int<4> *weight_col18,
    const ap_int<4> *weight_col19,
    const ap_int<4> *weight_col20,
    const ap_int<4> *weight_col21,
    const ap_int<4> *weight_col22,
    const ap_int<4> *weight_col23,
    const ap_int<4> *weight_col24,
    const ap_int<4> *weight_col25,
    const ap_int<4> *weight_col26,
    const ap_int<4> *weight_col27,
    const ap_int<4> *weight_col28,
    const ap_int<4> *weight_col29,
    const ap_int<4> *weight_col30,
    const ap_int<4> *weight_col31
);

#endif // GEMV_H
