#include "GEMV.h"
#include "ap_int.h"

void WP521_4096(
    ap_int<4> w0[NUM_DSP],
    ap_int<4> w1[NUM_DSP],
    ap_uint<4> a0[DSP_ARRAY_ROWS],
    ap_uint<4> a1[DSP_ARRAY_ROWS],
    ap_int<8> result_a0w0[NUM_DSP],
    ap_int<8> result_a1w0[NUM_DSP],
    ap_int<8> result_a0w1[NUM_DSP],
    ap_int<8> result_a1w1[NUM_DSP]
);

static void load_activations(
    const ap_uint<4> *act_packed,
    ap_uint<4> a0[DSP_ARRAY_ROWS],
    ap_uint<4> a1[DSP_ARRAY_ROWS]) {
#pragma HLS INLINE

LOAD_ACTS:
    for (int row = 0; row < DSP_ARRAY_ROWS; ++row) {
#pragma HLS PIPELINE II=1
        a0[row] = act_packed[row];
        a1[row] = act_packed[DSP_ARRAY_ROWS + row];
    }
}

static void pack_weight_port(
    const ap_int<4> *weight_port,
    int port_idx,
    ap_int<4> w0[NUM_DSP],
    ap_int<4> w1[NUM_DSP]) {
#pragma HLS INLINE

PACK_LOCAL_COL:
    for (int local_col = 0; local_col < LOGICAL_COLS_PER_WEIGHT_PORT; ++local_col) {
        const int logical_col =
            (port_idx * LOGICAL_COLS_PER_WEIGHT_PORT) + local_col;

        if (logical_col < LOGICAL_OUTPUT_COLS) {
            const int physical_col = logical_col >> 1;
            const int lane = logical_col & 1;

        PACK_ROW:
            for (int row = 0; row < DSP_ARRAY_ROWS; ++row) {
#pragma HLS PIPELINE II=1
                const int pe_index = (physical_col * DSP_ARRAY_ROWS) + row;
                const int base = (local_col * DSP_ARRAY_ROWS) + row;
                const ap_int<4> weight_val = weight_port[base];

                if (lane == 0) {
                    w0[pe_index] = weight_val;
                } else {
                    w1[pe_index] = weight_val;
                }
            }
        }
    }
}

static void accumulate_results(
    ap_int<32> *output,
    ap_int<8> result_a0w0[NUM_DSP],
    ap_int<8> result_a1w0[NUM_DSP],
    ap_int<8> result_a0w1[NUM_DSP],
    ap_int<8> result_a1w1[NUM_DSP]) {
#pragma HLS INLINE

ACC_COL:
    for (int logical_col = 0; logical_col < LOGICAL_OUTPUT_COLS; ++logical_col) {
        const int physical_col = logical_col >> 1;
        const int lane = logical_col & 1;
        ap_int<32> sum = 0;

    ACC_ROW:
        for (int row = 0; row < DSP_ARRAY_ROWS; ++row) {
#pragma HLS PIPELINE II=1
            const int pe_index = (physical_col * DSP_ARRAY_ROWS) + row;

            if (lane == 0) {
                sum += (ap_int<32>)result_a0w0[pe_index] +
                       (ap_int<32>)result_a1w0[pe_index];
            } else {
                sum += (ap_int<32>)result_a0w1[pe_index] +
                       (ap_int<32>)result_a1w1[pe_index];
            }
        }

        output[logical_col] = sum;
    }
}

#define PACK_WEIGHT_PORT(PORT_PTR, PORT_IDX) \
    pack_weight_port((PORT_PTR), (PORT_IDX), w0_arr, w1_arr)

extern "C" void gemv_kernel(
    ap_int<32> *output,
    const ap_uint<4> *act_packed,
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
    const ap_int<4> *weight_col31) {
    #pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem_out
    #pragma HLS INTERFACE m_axi port=act_packed offset=slave bundle=gmem_act
    #pragma HLS INTERFACE m_axi port=weight_col0 offset=slave
    #pragma HLS INTERFACE m_axi port=weight_col1 offset=slave
    #pragma HLS INTERFACE m_axi port=weight_col2 offset=slave
    #pragma HLS INTERFACE m_axi port=weight_col3 offset=slave
    #pragma HLS INTERFACE m_axi port=weight_col4 offset=slave
    #pragma HLS INTERFACE m_axi port=weight_col5 offset=slave
    #pragma HLS INTERFACE m_axi port=weight_col6 offset=slave
    #pragma HLS INTERFACE m_axi port=weight_col7 offset=slave
    #pragma HLS INTERFACE m_axi port=weight_col8 offset=slave
    #pragma HLS INTERFACE m_axi port=weight_col9 offset=slave
    #pragma HLS INTERFACE m_axi port=weight_col10 offset=slave
    #pragma HLS INTERFACE m_axi port=weight_col11 offset=slave
    #pragma HLS INTERFACE m_axi port=weight_col12 offset=slave
    #pragma HLS INTERFACE m_axi port=weight_col13 offset=slave
    #pragma HLS INTERFACE m_axi port=weight_col14 offset=slave
    #pragma HLS INTERFACE m_axi port=weight_col15 offset=slave
    #pragma HLS INTERFACE m_axi port=weight_col16 offset=slave
    #pragma HLS INTERFACE m_axi port=weight_col17 offset=slave
    #pragma HLS INTERFACE m_axi port=weight_col18 offset=slave
    #pragma HLS INTERFACE m_axi port=weight_col19 offset=slave
    #pragma HLS INTERFACE m_axi port=weight_col20 offset=slave
    #pragma HLS INTERFACE m_axi port=weight_col21 offset=slave
    #pragma HLS INTERFACE m_axi port=weight_col22 offset=slave
    #pragma HLS INTERFACE m_axi port=weight_col23 offset=slave
    #pragma HLS INTERFACE m_axi port=weight_col24 offset=slave
    #pragma HLS INTERFACE m_axi port=weight_col25 offset=slave
    #pragma HLS INTERFACE m_axi port=weight_col26 offset=slave
    #pragma HLS INTERFACE m_axi port=weight_col27 offset=slave
    #pragma HLS INTERFACE m_axi port=weight_col28 offset=slave
    #pragma HLS INTERFACE m_axi port=weight_col29 offset=slave
    #pragma HLS INTERFACE m_axi port=weight_col30 offset=slave
    #pragma HLS INTERFACE m_axi port=weight_col31 offset=slave

    static ap_int<4> w0_arr[NUM_DSP];
    static ap_int<4> w1_arr[NUM_DSP];
    static ap_uint<4> a0_arr[DSP_ARRAY_ROWS];
    static ap_uint<4> a1_arr[DSP_ARRAY_ROWS];
    static ap_int<8> result_a0w0_arr[NUM_DSP];
    static ap_int<8> result_a1w0_arr[NUM_DSP];
    static ap_int<8> result_a0w1_arr[NUM_DSP];
    static ap_int<8> result_a1w1_arr[NUM_DSP];

    load_activations(act_packed, a0_arr, a1_arr);

    PACK_WEIGHT_PORT(weight_col0, 0);
    PACK_WEIGHT_PORT(weight_col1, 1);
    PACK_WEIGHT_PORT(weight_col2, 2);
    PACK_WEIGHT_PORT(weight_col3, 3);
    PACK_WEIGHT_PORT(weight_col4, 4);
    PACK_WEIGHT_PORT(weight_col5, 5);
    PACK_WEIGHT_PORT(weight_col6, 6);
    PACK_WEIGHT_PORT(weight_col7, 7);
    PACK_WEIGHT_PORT(weight_col8, 8);
    PACK_WEIGHT_PORT(weight_col9, 9);
    PACK_WEIGHT_PORT(weight_col10, 10);
    PACK_WEIGHT_PORT(weight_col11, 11);
    PACK_WEIGHT_PORT(weight_col12, 12);
    PACK_WEIGHT_PORT(weight_col13, 13);
    PACK_WEIGHT_PORT(weight_col14, 14);
    PACK_WEIGHT_PORT(weight_col15, 15);
    PACK_WEIGHT_PORT(weight_col16, 16);
    PACK_WEIGHT_PORT(weight_col17, 17);
    PACK_WEIGHT_PORT(weight_col18, 18);
    PACK_WEIGHT_PORT(weight_col19, 19);
    PACK_WEIGHT_PORT(weight_col20, 20);
    PACK_WEIGHT_PORT(weight_col21, 21);
    PACK_WEIGHT_PORT(weight_col22, 22);
    PACK_WEIGHT_PORT(weight_col23, 23);
    PACK_WEIGHT_PORT(weight_col24, 24);
    PACK_WEIGHT_PORT(weight_col25, 25);
    PACK_WEIGHT_PORT(weight_col26, 26);
    PACK_WEIGHT_PORT(weight_col27, 27);
    PACK_WEIGHT_PORT(weight_col28, 28);
    PACK_WEIGHT_PORT(weight_col29, 29);
    PACK_WEIGHT_PORT(weight_col30, 30);
    PACK_WEIGHT_PORT(weight_col31, 31);

    WP521_4096(
        w0_arr,
        w1_arr,
        a0_arr,
        a1_arr,
        result_a0w0_arr,
        result_a1w0_arr,
        result_a0w1_arr,
        result_a1w1_arr);

    accumulate_results(
        output,
        result_a0w0_arr,
        result_a1w0_arr,
        result_a0w1_arr,
        result_a1w1_arr);
}

#undef PACK_WEIGHT_PORT
