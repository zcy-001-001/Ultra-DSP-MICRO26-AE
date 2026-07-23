#ifndef GEMV_H
#define GEMV_H

#include "ap_int.h"

// DSP array configuration
#define DSP_ARRAY_ROWS 128
#define DSP_ARRAY_COLS 32
#define NUM_DSP (DSP_ARRAY_ROWS * DSP_ARRAY_COLS)

// AXI weight-port organization:
// keep the same 32 external AXI ports for every array size and distribute
// logical output columns across them as evenly as possible.
#define NUM_WEIGHT_AXI_PORTS 32
#define LOGICAL_COLS_PER_WEIGHT_PORT \
    (((DSP_ARRAY_COLS) + (NUM_WEIGHT_AXI_PORTS) - 1) / (NUM_WEIGHT_AXI_PORTS))

// Per-PE packed multiply counts (single mode: 3 weights x 2 activations)
#define WEIGHTS_PER_PE 3

// Input/output tensor sizes
// 2 activations per PE -> 2 * ARRAY_ROWS input elements
#define ACT_COUNT (2 * DSP_ARRAY_ROWS)
// 2 activations per PE -> 2 * ARRAY_COLS output elements
#define OUTPUT_COUNT (2 * DSP_ARRAY_COLS)

// Depth of each weight_col* AXI buffer
#define WEIGHT_PORT_DEPTH \
    (LOGICAL_COLS_PER_WEIGHT_PORT * DSP_ARRAY_ROWS * WEIGHTS_PER_PE)

// Kernel interface
// Single mode (no mode signal):
//   act_packed layout  : [a0(rows), a1(rows)]
//   output layout      : [sum_a0(cols), sum_a1(cols)]
//   weight_colN layout : up to LOGICAL_COLS_PER_WEIGHT_PORT logical columns.
//                        Each logical column is stored in row-major order as
//                        rows x (w0,w1,w2). Unused tail slots are ignored.
extern "C" void gemv_kernel(
    ap_int<32> *output,
    const ap_int<5> *act_packed,
    const ap_int<3> *weight_col0,
    const ap_int<3> *weight_col1,
    const ap_int<3> *weight_col2,
    const ap_int<3> *weight_col3,
    const ap_int<3> *weight_col4,
    const ap_int<3> *weight_col5,
    const ap_int<3> *weight_col6,
    const ap_int<3> *weight_col7,
    const ap_int<3> *weight_col8,
    const ap_int<3> *weight_col9,
    const ap_int<3> *weight_col10,
    const ap_int<3> *weight_col11,
    const ap_int<3> *weight_col12,
    const ap_int<3> *weight_col13,
    const ap_int<3> *weight_col14,
    const ap_int<3> *weight_col15,
    const ap_int<3> *weight_col16,
    const ap_int<3> *weight_col17,
    const ap_int<3> *weight_col18,
    const ap_int<3> *weight_col19,
    const ap_int<3> *weight_col20,
    const ap_int<3> *weight_col21,
    const ap_int<3> *weight_col22,
    const ap_int<3> *weight_col23,
    const ap_int<3> *weight_col24,
    const ap_int<3> *weight_col25,
    const ap_int<3> *weight_col26,
    const ap_int<3> *weight_col27,
    const ap_int<3> *weight_col28,
    const ap_int<3> *weight_col29,
    const ap_int<3> *weight_col30,
    const ap_int<3> *weight_col31
);

#endif // GEMV_H
