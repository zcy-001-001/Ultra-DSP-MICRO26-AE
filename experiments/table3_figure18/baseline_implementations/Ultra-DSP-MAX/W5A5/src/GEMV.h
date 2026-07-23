#ifndef GEMV_H
#define GEMV_H

#include "ap_int.h"

#define DSP_ARRAY_ROWS 64
#define DSP_ARRAY_COLS 64
#define NUM_DSP (DSP_ARRAY_ROWS * DSP_ARRAY_COLS)

#define NUM_WEIGHT_AXI_PORTS 32
#define LOGICAL_COLS_PER_WEIGHT_PORT \
    (((DSP_ARRAY_COLS) + (NUM_WEIGHT_AXI_PORTS) - 1) / (NUM_WEIGHT_AXI_PORTS))

#define P_WEIGHTS_PER_PE 2
#define P_ACT_GROUPS 3
#define P_RESULTS_PER_PE (P_WEIGHTS_PER_PE * P_ACT_GROUPS)
#define ACT_COUNT (P_ACT_GROUPS * DSP_ARRAY_ROWS)
#define OUTPUT_COUNT (P_ACT_GROUPS * DSP_ARRAY_COLS)
#define WEIGHT_PORT_DEPTH \
    (LOGICAL_COLS_PER_WEIGHT_PORT * DSP_ARRAY_ROWS * P_WEIGHTS_PER_PE)

// Kernel interface
//   act_packed layout  : [a1(rows), a2(rows), a3(rows)]
//   output layout      : [sum_a1(cols), sum_a2(cols), sum_a3(cols)]
//   weight_colN layout : up to LOGICAL_COLS_PER_WEIGHT_PORT logical columns.
//                        Each logical column is stored in row-major order as
//                        rows x (w1, w2). Weight words keep the raw sign-bit
//                        plus magnitude-bit encoding expected by W5A5_P.v.
extern "C" void gemv_kernel(
    ap_int<32> *output,
    const ap_int<5> *act_packed,
    const ap_int<5> *weight_col0,
    const ap_int<5> *weight_col1,
    const ap_int<5> *weight_col2,
    const ap_int<5> *weight_col3,
    const ap_int<5> *weight_col4,
    const ap_int<5> *weight_col5,
    const ap_int<5> *weight_col6,
    const ap_int<5> *weight_col7,
    const ap_int<5> *weight_col8,
    const ap_int<5> *weight_col9,
    const ap_int<5> *weight_col10,
    const ap_int<5> *weight_col11,
    const ap_int<5> *weight_col12,
    const ap_int<5> *weight_col13,
    const ap_int<5> *weight_col14,
    const ap_int<5> *weight_col15,
    const ap_int<5> *weight_col16,
    const ap_int<5> *weight_col17,
    const ap_int<5> *weight_col18,
    const ap_int<5> *weight_col19,
    const ap_int<5> *weight_col20,
    const ap_int<5> *weight_col21,
    const ap_int<5> *weight_col22,
    const ap_int<5> *weight_col23,
    const ap_int<5> *weight_col24,
    const ap_int<5> *weight_col25,
    const ap_int<5> *weight_col26,
    const ap_int<5> *weight_col27,
    const ap_int<5> *weight_col28,
    const ap_int<5> *weight_col29,
    const ap_int<5> *weight_col30,
    const ap_int<5> *weight_col31);

#endif // GEMV_H
