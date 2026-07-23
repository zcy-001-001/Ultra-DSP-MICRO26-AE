#include "GEMV.h"

#include "ap_int.h"

// W3A4_array black-box function prototype
void W3A4_array(
    ap_int<4> w0[NUM_DSP],
    ap_int<4> w1[NUM_DSP],
    ap_int<4> w2[NUM_DSP],
    ap_int<3> a0[DSP_ARRAY_ROWS],
    ap_int<3> a1[DSP_ARRAY_ROWS],
    ap_int<7> result0[NUM_DSP],
    ap_int<7> result1[NUM_DSP],
    ap_int<7> result2[NUM_DSP],
    ap_int<7> result3[NUM_DSP],
    ap_int<7> result4[NUM_DSP],
    ap_int<7> result5[NUM_DSP]);

static void load_activations(
    const ap_int<3> *act_packed,
    ap_int<3> a0[DSP_ARRAY_ROWS],
    ap_int<3> a1[DSP_ARRAY_ROWS]) {
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
    ap_int<4> w1[NUM_DSP],
    ap_int<4> w2[NUM_DSP]) {
#pragma HLS INLINE

    const int col_stride = DSP_ARRAY_ROWS * WEIGHTS_PER_PE;

PACK_LOCAL_COL:
    for (int local_col = 0; local_col < LOGICAL_COLS_PER_WEIGHT_PORT; ++local_col) {
        const int logical_col =
            (port_idx * LOGICAL_COLS_PER_WEIGHT_PORT) + local_col;

        if (logical_col < DSP_ARRAY_COLS) {
PACK_ROW:
            for (int row = 0; row < DSP_ARRAY_ROWS; ++row) {
#pragma HLS PIPELINE II=1
                const int pe_index = (logical_col * DSP_ARRAY_ROWS) + row;
                const int base =
                    (local_col * col_stride) + (row * WEIGHTS_PER_PE);

                w0[pe_index] = weight_port[base];
                w1[pe_index] = weight_port[base + 1];
                w2[pe_index] = weight_port[base + 2];
            }
        }
    }
}

static void accumulate_results(
    ap_int<32> *output,
    ap_int<7> result0[NUM_DSP],
    ap_int<7> result1[NUM_DSP],
    ap_int<7> result2[NUM_DSP],
    ap_int<7> result3[NUM_DSP],
    ap_int<7> result4[NUM_DSP],
    ap_int<7> result5[NUM_DSP]) {
#pragma HLS INLINE

ACC_COL:
    for (int col = 0; col < DSP_ARRAY_COLS; ++col) {
        ap_int<32> sum_a0 = 0;
        ap_int<32> sum_a1 = 0;

    ACC_ROW:
        for (int row = 0; row < DSP_ARRAY_ROWS; ++row) {
#pragma HLS PIPELINE II=1
            const int pe_index = (col * DSP_ARRAY_ROWS) + row;

            sum_a0 += (ap_int<32>)result0[pe_index] +
                      (ap_int<32>)result1[pe_index] +
                      (ap_int<32>)result2[pe_index];
            sum_a1 += (ap_int<32>)result3[pe_index] +
                      (ap_int<32>)result4[pe_index] +
                      (ap_int<32>)result5[pe_index];
        }

        output[col] = sum_a0;
        output[DSP_ARRAY_COLS + col] = sum_a1;
    }
}

#define PACK_PORT(PORT_PTR, PORT_IDX) \
    pack_weight_port((PORT_PTR), (PORT_IDX), w0_arr, w1_arr, w2_arr)

extern "C" void gemv_kernel(
    ap_int<32> *output,
    const ap_int<3> *act_packed,
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
    static ap_int<4> w2_arr[NUM_DSP];
    static ap_int<3> a0_arr[DSP_ARRAY_ROWS];
    static ap_int<3> a1_arr[DSP_ARRAY_ROWS];

    static ap_int<7> result0_arr[NUM_DSP];
    static ap_int<7> result1_arr[NUM_DSP];
    static ap_int<7> result2_arr[NUM_DSP];
    static ap_int<7> result3_arr[NUM_DSP];
    static ap_int<7> result4_arr[NUM_DSP];
    static ap_int<7> result5_arr[NUM_DSP];

    load_activations(act_packed, a0_arr, a1_arr);

    PACK_PORT(weight_col0, 0);
    PACK_PORT(weight_col1, 1);
    PACK_PORT(weight_col2, 2);
    PACK_PORT(weight_col3, 3);
    PACK_PORT(weight_col4, 4);
    PACK_PORT(weight_col5, 5);
    PACK_PORT(weight_col6, 6);
    PACK_PORT(weight_col7, 7);
    PACK_PORT(weight_col8, 8);
    PACK_PORT(weight_col9, 9);
    PACK_PORT(weight_col10, 10);
    PACK_PORT(weight_col11, 11);
    PACK_PORT(weight_col12, 12);
    PACK_PORT(weight_col13, 13);
    PACK_PORT(weight_col14, 14);
    PACK_PORT(weight_col15, 15);
    PACK_PORT(weight_col16, 16);
    PACK_PORT(weight_col17, 17);
    PACK_PORT(weight_col18, 18);
    PACK_PORT(weight_col19, 19);
    PACK_PORT(weight_col20, 20);
    PACK_PORT(weight_col21, 21);
    PACK_PORT(weight_col22, 22);
    PACK_PORT(weight_col23, 23);
    PACK_PORT(weight_col24, 24);
    PACK_PORT(weight_col25, 25);
    PACK_PORT(weight_col26, 26);
    PACK_PORT(weight_col27, 27);
    PACK_PORT(weight_col28, 28);
    PACK_PORT(weight_col29, 29);
    PACK_PORT(weight_col30, 30);
    PACK_PORT(weight_col31, 31);

    W3A4_array(
        w0_arr,
        w1_arr,
        w2_arr,
        a0_arr,
        a1_arr,
        result0_arr,
        result1_arr,
        result2_arr,
        result3_arr,
        result4_arr,
        result5_arr);

    accumulate_results(
        output,
        result0_arr,
        result1_arr,
        result2_arr,
        result3_arr,
        result4_arr,
        result5_arr);
}

#undef PACK_PORT
