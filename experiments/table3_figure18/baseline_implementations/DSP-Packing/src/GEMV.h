#ifndef GEMV_H
#define GEMV_H

#include "ap_int.h"

#define DSP_ARRAY_ROWS 128
#define DSP_ARRAY_COLS 32
#define NUM_DSP (DSP_ARRAY_ROWS * DSP_ARRAY_COLS)

#define ACTIVATION_BANKS 3
#define PACKED_OUTPUTS_PER_PE 2
#define NUM_WEIGHT_AXI_PORTS 32
#define LOGICAL_OUTPUT_COLS (DSP_ARRAY_COLS * PACKED_OUTPUTS_PER_PE)
#define LOGICAL_COLS_PER_WEIGHT_PORT     (((LOGICAL_OUTPUT_COLS) + (NUM_WEIGHT_AXI_PORTS) - 1) / (NUM_WEIGHT_AXI_PORTS))

#define ACT_COUNT (ACTIVATION_BANKS * DSP_ARRAY_ROWS)
#define OUTPUT_COUNT LOGICAL_OUTPUT_COLS
#define WEIGHT_PORT_DEPTH (LOGICAL_COLS_PER_WEIGHT_PORT * DSP_ARRAY_ROWS)

extern "C" void gemv_kernel(
    ap_int<32> *output,
    const ap_uint<4> *act_packed,
    const ap_uint<5> *weight_col0,
    const ap_uint<5> *weight_col1,
    const ap_uint<5> *weight_col2,
    const ap_uint<5> *weight_col3,
    const ap_uint<5> *weight_col4,
    const ap_uint<5> *weight_col5,
    const ap_uint<5> *weight_col6,
    const ap_uint<5> *weight_col7,
    const ap_uint<5> *weight_col8,
    const ap_uint<5> *weight_col9,
    const ap_uint<5> *weight_col10,
    const ap_uint<5> *weight_col11,
    const ap_uint<5> *weight_col12,
    const ap_uint<5> *weight_col13,
    const ap_uint<5> *weight_col14,
    const ap_uint<5> *weight_col15,
    const ap_uint<5> *weight_col16,
    const ap_uint<5> *weight_col17,
    const ap_uint<5> *weight_col18,
    const ap_uint<5> *weight_col19,
    const ap_uint<5> *weight_col20,
    const ap_uint<5> *weight_col21,
    const ap_uint<5> *weight_col22,
    const ap_uint<5> *weight_col23,
    const ap_uint<5> *weight_col24,
    const ap_uint<5> *weight_col25,
    const ap_uint<5> *weight_col26,
    const ap_uint<5> *weight_col27,
    const ap_uint<5> *weight_col28,
    const ap_uint<5> *weight_col29,
    const ap_uint<5> *weight_col30,
    const ap_uint<5> *weight_col31
);

#endif
