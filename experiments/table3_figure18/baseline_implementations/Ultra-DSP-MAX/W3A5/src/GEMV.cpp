#include "GEMV.h"

#include "ap_int.h"

void W3A5_P(
    ap_int<3> w1[NUM_DSP],
    ap_int<3> w2[NUM_DSP],
    ap_int<3> w3[NUM_DSP],
    ap_int<3> w4[NUM_DSP],
    ap_int<3> w5[NUM_DSP],
    ap_int<5> a1[DSP_ARRAY_ROWS],
    ap_int<5> a2[DSP_ARRAY_ROWS],
    ap_int<7> result1[NUM_DSP],
    ap_int<7> result2[NUM_DSP],
    ap_int<7> result3[NUM_DSP],
    ap_int<7> result4[NUM_DSP],
    ap_int<7> result5[NUM_DSP],
    ap_int<7> result6[NUM_DSP],
    ap_int<7> result7[NUM_DSP],
    ap_int<7> result8[NUM_DSP],
    ap_int<7> result9[NUM_DSP],
    ap_int<7> result10[NUM_DSP],
    ap_uint<1> sign1[NUM_DSP],
    ap_uint<1> sign2[NUM_DSP],
    ap_uint<1> sign3[NUM_DSP],
    ap_uint<1> sign4[NUM_DSP],
    ap_uint<1> sign5[NUM_DSP],
    ap_uint<1> sign6[NUM_DSP],
    ap_uint<1> sign7[NUM_DSP],
    ap_uint<1> sign8[NUM_DSP],
    ap_uint<1> sign9[NUM_DSP],
    ap_uint<1> sign10[NUM_DSP]);

static void load_activations(
    const ap_int<5> *act_packed,
    ap_int<5> a1[DSP_ARRAY_ROWS],
    ap_int<5> a2[DSP_ARRAY_ROWS]) {
#pragma HLS INLINE

LOAD_ACTS:
    for (int row = 0; row < DSP_ARRAY_ROWS; ++row) {
#pragma HLS PIPELINE II=1
        a1[row] = act_packed[row];
        a2[row] = act_packed[DSP_ARRAY_ROWS + row];
    }
}

static void pack_weight_port(
    const ap_int<3> *weight_port,
    int port_idx,
    ap_int<3> w1[NUM_DSP],
    ap_int<3> w2[NUM_DSP],
    ap_int<3> w3[NUM_DSP],
    ap_int<3> w4[NUM_DSP],
    ap_int<3> w5[NUM_DSP]) {
#pragma HLS INLINE

    const int col_stride = DSP_ARRAY_ROWS * P_WEIGHTS_PER_PE;

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
                    (local_col * col_stride) + (row * P_WEIGHTS_PER_PE);

                w1[pe_index] = weight_port[base + 0];
                w2[pe_index] = weight_port[base + 1];
                w3[pe_index] = weight_port[base + 2];
                w4[pe_index] = weight_port[base + 3];
                w5[pe_index] = weight_port[base + 4];
            }
        }
    }
}

static void accumulate_results(
    ap_int<32> *output,
    ap_int<7> result1[NUM_DSP],
    ap_int<7> result2[NUM_DSP],
    ap_int<7> result3[NUM_DSP],
    ap_int<7> result4[NUM_DSP],
    ap_int<7> result5[NUM_DSP],
    ap_int<7> result6[NUM_DSP],
    ap_int<7> result7[NUM_DSP],
    ap_int<7> result8[NUM_DSP],
    ap_int<7> result9[NUM_DSP],
    ap_int<7> result10[NUM_DSP],
    ap_uint<1> sign1[NUM_DSP],
    ap_uint<1> sign2[NUM_DSP],
    ap_uint<1> sign3[NUM_DSP],
    ap_uint<1> sign4[NUM_DSP],
    ap_uint<1> sign5[NUM_DSP],
    ap_uint<1> sign6[NUM_DSP],
    ap_uint<1> sign7[NUM_DSP],
    ap_uint<1> sign8[NUM_DSP],
    ap_uint<1> sign9[NUM_DSP],
    ap_uint<1> sign10[NUM_DSP]) {
#pragma HLS INLINE

ACC_COL:
    for (int col = 0; col < DSP_ARRAY_COLS; ++col) {
        ap_int<32> sum_a1 = 0;
        ap_int<32> sum_a2 = 0;
        ap_int<32> sign_a1 = 0;
        ap_int<32> sign_a2 = 0;

    ACC_ROW:
        for (int row = 0; row < DSP_ARRAY_ROWS; ++row) {
#pragma HLS PIPELINE II=1
            const int pe_index = (col * DSP_ARRAY_ROWS) + row;

            sum_a1 += (ap_int<32>)result1[pe_index] +
                      (ap_int<32>)result2[pe_index] +
                      (ap_int<32>)result3[pe_index] +
                      (ap_int<32>)result4[pe_index] +
                      (ap_int<32>)result5[pe_index];
            sum_a2 += (ap_int<32>)result6[pe_index] +
                      (ap_int<32>)result7[pe_index] +
                      (ap_int<32>)result8[pe_index] +
                      (ap_int<32>)result9[pe_index] +
                      (ap_int<32>)result10[pe_index];

            sign_a1 += (ap_int<32>)sign1[pe_index] +
                       (ap_int<32>)sign2[pe_index] +
                       (ap_int<32>)sign3[pe_index] +
                       (ap_int<32>)sign4[pe_index] +
                       (ap_int<32>)sign5[pe_index];
            sign_a2 += (ap_int<32>)sign6[pe_index] +
                       (ap_int<32>)sign7[pe_index] +
                       (ap_int<32>)sign8[pe_index] +
                       (ap_int<32>)sign9[pe_index] +
                       (ap_int<32>)sign10[pe_index];
        }

        output[col] = sum_a1 + sign_a1;
        output[DSP_ARRAY_COLS + col] = sum_a2 + sign_a2;
    }
}

#define PACK_PORT(PORT_PTR, PORT_IDX) \
    pack_weight_port((PORT_PTR), (PORT_IDX), w1_arr, w2_arr, w3_arr, w4_arr, w5_arr)

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
    const ap_int<3> *weight_col31) {
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

    static ap_int<3> w1_arr[NUM_DSP];
    static ap_int<3> w2_arr[NUM_DSP];
    static ap_int<3> w3_arr[NUM_DSP];
    static ap_int<3> w4_arr[NUM_DSP];
    static ap_int<3> w5_arr[NUM_DSP];
    static ap_int<5> a1_arr[DSP_ARRAY_ROWS];
    static ap_int<5> a2_arr[DSP_ARRAY_ROWS];

    static ap_int<7> result1_arr[NUM_DSP];
    static ap_int<7> result2_arr[NUM_DSP];
    static ap_int<7> result3_arr[NUM_DSP];
    static ap_int<7> result4_arr[NUM_DSP];
    static ap_int<7> result5_arr[NUM_DSP];
    static ap_int<7> result6_arr[NUM_DSP];
    static ap_int<7> result7_arr[NUM_DSP];
    static ap_int<7> result8_arr[NUM_DSP];
    static ap_int<7> result9_arr[NUM_DSP];
    static ap_int<7> result10_arr[NUM_DSP];
    static ap_uint<1> sign1_arr[NUM_DSP];
    static ap_uint<1> sign2_arr[NUM_DSP];
    static ap_uint<1> sign3_arr[NUM_DSP];
    static ap_uint<1> sign4_arr[NUM_DSP];
    static ap_uint<1> sign5_arr[NUM_DSP];
    static ap_uint<1> sign6_arr[NUM_DSP];
    static ap_uint<1> sign7_arr[NUM_DSP];
    static ap_uint<1> sign8_arr[NUM_DSP];
    static ap_uint<1> sign9_arr[NUM_DSP];
    static ap_uint<1> sign10_arr[NUM_DSP];

    load_activations(act_packed, a1_arr, a2_arr);

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

    W3A5_P(
        w1_arr,
        w2_arr,
        w3_arr,
        w4_arr,
        w5_arr,
        a1_arr,
        a2_arr,
        result1_arr,
        result2_arr,
        result3_arr,
        result4_arr,
        result5_arr,
        result6_arr,
        result7_arr,
        result8_arr,
        result9_arr,
        result10_arr,
        sign1_arr,
        sign2_arr,
        sign3_arr,
        sign4_arr,
        sign5_arr,
        sign6_arr,
        sign7_arr,
        sign8_arr,
        sign9_arr,
        sign10_arr);

    accumulate_results(
        output,
        result1_arr,
        result2_arr,
        result3_arr,
        result4_arr,
        result5_arr,
        result6_arr,
        result7_arr,
        result8_arr,
        result9_arr,
        result10_arr,
        sign1_arr,
        sign2_arr,
        sign3_arr,
        sign4_arr,
        sign5_arr,
        sign6_arr,
        sign7_arr,
        sign8_arr,
        sign9_arr,
        sign10_arr);
}

#undef PACK_PORT
