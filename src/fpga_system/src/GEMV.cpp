#include "GEMV.h"

#include "ap_int.h"

// Hybrid_INT4_INT4_PD black-box function prototype
void Hybrid_INT4_INT4_PD(
    ap_uint<1> mode,
    ap_int<4> w1[NUM_DSP],
    ap_int<4> w2[NUM_DSP],
    ap_int<4> w3[NUM_DSP],
    ap_int<4> w4[NUM_DSP],
    ap_int<4> w5[NUM_DSP],
    ap_int<4> w6[NUM_DSP],
    ap_int<4> w7[NUM_DSP],
    ap_int<4> a1[DSP_ARRAY_ROWS],
    ap_int<4> a2[DSP_ARRAY_ROWS],
    ap_int<4> a3[DSP_ARRAY_ROWS],
    ap_int<7> result1[NUM_DSP],
    ap_int<7> result2[NUM_DSP],
    ap_int<7> result3[NUM_DSP],
    ap_int<7> result4[NUM_DSP],
    ap_int<7> result5[NUM_DSP],
    ap_int<7> result6[NUM_DSP],
    ap_int<7> result7[NUM_DSP],
    ap_int<7> result8[NUM_DSP],
    ap_int<7> result9[NUM_DSP]);

static void load_prefill_activations(
    const ap_int<4> *act_packed,
    ap_int<4> a1[DSP_ARRAY_ROWS],
    ap_int<4> a2[DSP_ARRAY_ROWS],
    ap_int<4> a3[DSP_ARRAY_ROWS]) {
#pragma HLS INLINE

LOAD_PREFILL_ACTS:
    for (int row = 0; row < DSP_ARRAY_ROWS; ++row) {
#pragma HLS PIPELINE II=1
        a1[row] = act_packed[row];
        a2[row] = act_packed[DSP_ARRAY_ROWS + row];
        a3[row] = act_packed[(2 * DSP_ARRAY_ROWS) + row];
    }
}

static void load_decoding_activations(
    const ap_int<4> *act_packed,
    ap_int<4> a1[DSP_ARRAY_ROWS],
    ap_int<4> a2[DSP_ARRAY_ROWS],
    ap_int<4> a3[DSP_ARRAY_ROWS]) {
#pragma HLS INLINE

LOAD_DECODING_ACTS:
    for (int row = 0; row < DSP_ARRAY_ROWS; ++row) {
#pragma HLS PIPELINE II=1
        a1[row] = act_packed[row];
        a2[row] = 0;
        a3[row] = 0;
    }
}

static void pack_prefill_weight_port(
    const ap_int<4> *weight_port,
    int port_idx,
    ap_int<4> w1[NUM_DSP],
    ap_int<4> w2[NUM_DSP],
    ap_int<4> w3[NUM_DSP],
    ap_int<4> w4[NUM_DSP],
    ap_int<4> w5[NUM_DSP],
    ap_int<4> w6[NUM_DSP],
    ap_int<4> w7[NUM_DSP]) {
#pragma HLS INLINE

    const int col_stride = DSP_ARRAY_ROWS * PREFILL_WEIGHTS_PER_PE;

PACK_PREFILL_LOCAL_COL:
    for (int local_col = 0; local_col < LOGICAL_COLS_PER_WEIGHT_PORT; ++local_col) {
        const int logical_col =
            (port_idx * LOGICAL_COLS_PER_WEIGHT_PORT) + local_col;

        if (logical_col < DSP_ARRAY_COLS) {
PACK_PREFILL_ROW:
            for (int row = 0; row < DSP_ARRAY_ROWS; ++row) {
#pragma HLS PIPELINE II=1
                const int pe_index = (logical_col * DSP_ARRAY_ROWS) + row;
                const int base =
                    (local_col * col_stride) + (row * PREFILL_WEIGHTS_PER_PE);

                w1[pe_index] = weight_port[base];
                w2[pe_index] = weight_port[base + 1];
                w3[pe_index] = weight_port[base + 2];
                w4[pe_index] = 0;
                w5[pe_index] = 0;
                w6[pe_index] = 0;
                w7[pe_index] = 0;
            }
        }
    }
}

static void pack_decoding_weight_port(
    const ap_int<4> *weight_port,
    int port_idx,
    ap_int<4> w1[NUM_DSP],
    ap_int<4> w2[NUM_DSP],
    ap_int<4> w3[NUM_DSP],
    ap_int<4> w4[NUM_DSP],
    ap_int<4> w5[NUM_DSP],
    ap_int<4> w6[NUM_DSP],
    ap_int<4> w7[NUM_DSP]) {
#pragma HLS INLINE

    const int col_stride = DSP_ARRAY_ROWS * DECODING_WEIGHTS_PER_PE;

PACK_DECODING_LOCAL_COL:
    for (int local_col = 0; local_col < LOGICAL_COLS_PER_WEIGHT_PORT; ++local_col) {
        const int logical_col =
            (port_idx * LOGICAL_COLS_PER_WEIGHT_PORT) + local_col;

        if (logical_col < DSP_ARRAY_COLS) {
PACK_DECODING_ROW:
            for (int row = 0; row < DSP_ARRAY_ROWS; ++row) {
#pragma HLS PIPELINE II=1
                const int pe_index = (logical_col * DSP_ARRAY_ROWS) + row;
                const int base =
                    (local_col * col_stride) + (row * DECODING_WEIGHTS_PER_PE);

                w1[pe_index] = weight_port[base];
                w2[pe_index] = weight_port[base + 1];
                w3[pe_index] = weight_port[base + 2];
                w4[pe_index] = weight_port[base + 3];
                w5[pe_index] = weight_port[base + 4];
                w6[pe_index] = weight_port[base + 5];
                w7[pe_index] = weight_port[base + 6];
            }
        }
    }
}

static void accumulate_prefill_results(
    ap_int<32> *output,
    ap_int<7> result1[NUM_DSP],
    ap_int<7> result2[NUM_DSP],
    ap_int<7> result3[NUM_DSP],
    ap_int<7> result4[NUM_DSP],
    ap_int<7> result5[NUM_DSP],
    ap_int<7> result6[NUM_DSP],
    ap_int<7> result7[NUM_DSP],
    ap_int<7> result8[NUM_DSP],
    ap_int<7> result9[NUM_DSP]) {
#pragma HLS INLINE

ACC_PREFILL_COL:
    for (int col = 0; col < DSP_ARRAY_COLS; ++col) {
        ap_int<32> sum_a1 = 0;
        ap_int<32> sum_a2 = 0;
        ap_int<32> sum_a3 = 0;

    ACC_PREFILL_ROW:
        for (int row = 0; row < DSP_ARRAY_ROWS; ++row) {
#pragma HLS PIPELINE II=1
            const int pe_index = (col * DSP_ARRAY_ROWS) + row;

            sum_a1 += (ap_int<32>)result1[pe_index] +
                      (ap_int<32>)result4[pe_index] +
                      (ap_int<32>)result7[pe_index];
            sum_a2 += (ap_int<32>)result2[pe_index] +
                      (ap_int<32>)result5[pe_index] +
                      (ap_int<32>)result8[pe_index];
            sum_a3 += (ap_int<32>)result3[pe_index] +
                      (ap_int<32>)result6[pe_index] +
                      (ap_int<32>)result9[pe_index];
        }

        output[col] = sum_a1;
        output[DSP_ARRAY_COLS + col] = sum_a2;
        output[(2 * DSP_ARRAY_COLS) + col] = sum_a3;
    }
}

static void accumulate_decoding_results(
    ap_int<32> *output,
    ap_int<7> result1[NUM_DSP],
    ap_int<7> result2[NUM_DSP],
    ap_int<7> result3[NUM_DSP],
    ap_int<7> result4[NUM_DSP],
    ap_int<7> result5[NUM_DSP],
    ap_int<7> result6[NUM_DSP],
    ap_int<7> result7[NUM_DSP]) {
#pragma HLS INLINE

ACC_DECODING_COL:
    for (int col = 0; col < DSP_ARRAY_COLS; ++col) {
        ap_int<32> sum = 0;

    ACC_DECODING_ROW:
        for (int row = 0; row < DSP_ARRAY_ROWS; ++row) {
#pragma HLS PIPELINE II=1
            const int pe_index = (col * DSP_ARRAY_ROWS) + row;

            sum += (ap_int<32>)result1[pe_index] +
                   (ap_int<32>)result2[pe_index] +
                   (ap_int<32>)result3[pe_index] +
                   (ap_int<32>)result4[pe_index] +
                   (ap_int<32>)result5[pe_index] +
                   (ap_int<32>)result6[pe_index] +
                   (ap_int<32>)result7[pe_index];
        }

        output[col] = sum;
    }

ZERO_UNUSED_OUTPUTS:
    for (int idx = DECODING_OUTPUT_COUNT; idx < MAX_OUTPUT_COUNT; ++idx) {
#pragma HLS PIPELINE II=1
        output[idx] = 0;
    }
}

#define PACK_PREFILL_PORT(PORT_PTR, PORT_IDX) \
    pack_prefill_weight_port((PORT_PTR), (PORT_IDX), w1_arr, w2_arr, w3_arr, w4_arr, w5_arr, w6_arr, w7_arr)

#define PACK_DECODING_PORT(PORT_PTR, PORT_IDX) \
    pack_decoding_weight_port((PORT_PTR), (PORT_IDX), w1_arr, w2_arr, w3_arr, w4_arr, w5_arr, w6_arr, w7_arr)

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

    static ap_int<4> w1_arr[NUM_DSP];
    static ap_int<4> w2_arr[NUM_DSP];
    static ap_int<4> w3_arr[NUM_DSP];
    static ap_int<4> w4_arr[NUM_DSP];
    static ap_int<4> w5_arr[NUM_DSP];
    static ap_int<4> w6_arr[NUM_DSP];
    static ap_int<4> w7_arr[NUM_DSP];
    static ap_int<4> a1_arr[DSP_ARRAY_ROWS];
    static ap_int<4> a2_arr[DSP_ARRAY_ROWS];
    static ap_int<4> a3_arr[DSP_ARRAY_ROWS];

    static ap_int<7> result1_arr[NUM_DSP];
    static ap_int<7> result2_arr[NUM_DSP];
    static ap_int<7> result3_arr[NUM_DSP];
    static ap_int<7> result4_arr[NUM_DSP];
    static ap_int<7> result5_arr[NUM_DSP];
    static ap_int<7> result6_arr[NUM_DSP];
    static ap_int<7> result7_arr[NUM_DSP];
    static ap_int<7> result8_arr[NUM_DSP];
    static ap_int<7> result9_arr[NUM_DSP];

    if (mode == 0) {
        load_prefill_activations(act_packed, a1_arr, a2_arr, a3_arr);

        PACK_PREFILL_PORT(weight_col0, 0);
        PACK_PREFILL_PORT(weight_col1, 1);
        PACK_PREFILL_PORT(weight_col2, 2);
        PACK_PREFILL_PORT(weight_col3, 3);
        PACK_PREFILL_PORT(weight_col4, 4);
        PACK_PREFILL_PORT(weight_col5, 5);
        PACK_PREFILL_PORT(weight_col6, 6);
        PACK_PREFILL_PORT(weight_col7, 7);
        PACK_PREFILL_PORT(weight_col8, 8);
        PACK_PREFILL_PORT(weight_col9, 9);
        PACK_PREFILL_PORT(weight_col10, 10);
        PACK_PREFILL_PORT(weight_col11, 11);
        PACK_PREFILL_PORT(weight_col12, 12);
        PACK_PREFILL_PORT(weight_col13, 13);
        PACK_PREFILL_PORT(weight_col14, 14);
        PACK_PREFILL_PORT(weight_col15, 15);
        PACK_PREFILL_PORT(weight_col16, 16);
        PACK_PREFILL_PORT(weight_col17, 17);
        PACK_PREFILL_PORT(weight_col18, 18);
        PACK_PREFILL_PORT(weight_col19, 19);
        PACK_PREFILL_PORT(weight_col20, 20);
        PACK_PREFILL_PORT(weight_col21, 21);
        PACK_PREFILL_PORT(weight_col22, 22);
        PACK_PREFILL_PORT(weight_col23, 23);
        PACK_PREFILL_PORT(weight_col24, 24);
        PACK_PREFILL_PORT(weight_col25, 25);
        PACK_PREFILL_PORT(weight_col26, 26);
        PACK_PREFILL_PORT(weight_col27, 27);
        PACK_PREFILL_PORT(weight_col28, 28);
        PACK_PREFILL_PORT(weight_col29, 29);
        PACK_PREFILL_PORT(weight_col30, 30);
        PACK_PREFILL_PORT(weight_col31, 31);
    } else {
        load_decoding_activations(act_packed, a1_arr, a2_arr, a3_arr);

        PACK_DECODING_PORT(weight_col0, 0);
        PACK_DECODING_PORT(weight_col1, 1);
        PACK_DECODING_PORT(weight_col2, 2);
        PACK_DECODING_PORT(weight_col3, 3);
        PACK_DECODING_PORT(weight_col4, 4);
        PACK_DECODING_PORT(weight_col5, 5);
        PACK_DECODING_PORT(weight_col6, 6);
        PACK_DECODING_PORT(weight_col7, 7);
        PACK_DECODING_PORT(weight_col8, 8);
        PACK_DECODING_PORT(weight_col9, 9);
        PACK_DECODING_PORT(weight_col10, 10);
        PACK_DECODING_PORT(weight_col11, 11);
        PACK_DECODING_PORT(weight_col12, 12);
        PACK_DECODING_PORT(weight_col13, 13);
        PACK_DECODING_PORT(weight_col14, 14);
        PACK_DECODING_PORT(weight_col15, 15);
        PACK_DECODING_PORT(weight_col16, 16);
        PACK_DECODING_PORT(weight_col17, 17);
        PACK_DECODING_PORT(weight_col18, 18);
        PACK_DECODING_PORT(weight_col19, 19);
        PACK_DECODING_PORT(weight_col20, 20);
        PACK_DECODING_PORT(weight_col21, 21);
        PACK_DECODING_PORT(weight_col22, 22);
        PACK_DECODING_PORT(weight_col23, 23);
        PACK_DECODING_PORT(weight_col24, 24);
        PACK_DECODING_PORT(weight_col25, 25);
        PACK_DECODING_PORT(weight_col26, 26);
        PACK_DECODING_PORT(weight_col27, 27);
        PACK_DECODING_PORT(weight_col28, 28);
        PACK_DECODING_PORT(weight_col29, 29);
        PACK_DECODING_PORT(weight_col30, 30);
        PACK_DECODING_PORT(weight_col31, 31);
    }

    Hybrid_INT4_INT4_PD(
        mode,
        w1_arr,
        w2_arr,
        w3_arr,
        w4_arr,
        w5_arr,
        w6_arr,
        w7_arr,
        a1_arr,
        a2_arr,
        a3_arr,
        result1_arr,
        result2_arr,
        result3_arr,
        result4_arr,
        result5_arr,
        result6_arr,
        result7_arr,
        result8_arr,
        result9_arr);

    if (mode == 0) {
        accumulate_prefill_results(
            output,
            result1_arr,
            result2_arr,
            result3_arr,
            result4_arr,
            result5_arr,
            result6_arr,
            result7_arr,
            result8_arr,
            result9_arr);
    } else {
        accumulate_decoding_results(
            output,
            result1_arr,
            result2_arr,
            result3_arr,
            result4_arr,
            result5_arr,
            result6_arr,
            result7_arr);
    }
}

#undef PACK_PREFILL_PORT
#undef PACK_DECODING_PORT
