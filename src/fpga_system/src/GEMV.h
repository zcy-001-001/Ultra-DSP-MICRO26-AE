#ifndef GEMV_H
#define GEMV_H

#include "ap_int.h"

// DSP array configuration
#define DSP_ARRAY_ROWS 64
#define DSP_ARRAY_COLS 64
#define NUM_DSP (DSP_ARRAY_ROWS * DSP_ARRAY_COLS)

// AXI weight-port organization:
// keep the same 32 external AXI ports for every array size and distribute
// logical output columns across them as evenly as possible.
#define NUM_WEIGHT_AXI_PORTS 32
#define LOGICAL_COLS_PER_WEIGHT_PORT \
    (((DSP_ARRAY_COLS) + (NUM_WEIGHT_AXI_PORTS) - 1) / (NUM_WEIGHT_AXI_PORTS))

// Per-PE packed multiply counts
#define PREFILL_WEIGHTS_PER_PE 3
#define DECODING_WEIGHTS_PER_PE 7

// Input/output tensor sizes
#define PREFILL_ACT_COUNT (3 * DSP_ARRAY_ROWS)
#define DECODING_ACT_COUNT DSP_ARRAY_ROWS
#define PREFILL_OUTPUT_COUNT (3 * DSP_ARRAY_COLS)
#define DECODING_OUTPUT_COUNT DSP_ARRAY_COLS
#define MAX_OUTPUT_COUNT PREFILL_OUTPUT_COUNT

// Depth of each weight_col* AXI buffer
#define PREFILL_WEIGHT_PORT_DEPTH \
    (LOGICAL_COLS_PER_WEIGHT_PORT * DSP_ARRAY_ROWS * PREFILL_WEIGHTS_PER_PE)
#define DECODING_WEIGHT_PORT_DEPTH \
    (LOGICAL_COLS_PER_WEIGHT_PORT * DSP_ARRAY_ROWS * DECODING_WEIGHTS_PER_PE)

// Backward-compatible aliases for older host/test code.
#define PREFILL_WEIGHT_PAIR_DEPTH PREFILL_WEIGHT_PORT_DEPTH
#define DECODING_WEIGHT_PAIR_DEPTH DECODING_WEIGHT_PORT_DEPTH

// Kernel interface
// mode = 0: Prefill
//   act_packed layout  : [a1(rows), a2(rows), a3(rows)]
//   output layout      : [sum_a1(cols), sum_a2(cols), sum_a3(cols)]
//   weight_colN layout : up to LOGICAL_COLS_PER_WEIGHT_PORT logical columns.
//                        Each logical column is stored in row-major order as
//                        rows x (w1,w2,w3). Unused tail slots are ignored.
//
// mode = 1: Decoding
//   act_packed layout  : [a1(rows)]
//   output layout      : [sum(cols)] and the remaining output buffer is 0
//   weight_colN layout : up to LOGICAL_COLS_PER_WEIGHT_PORT logical columns.
//                        Each logical column is stored in row-major order as
//                        rows x (w1..w7). Unused tail slots are ignored.
extern "C" void gemv_kernel(
    ap_uint<1> mode,
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
