#include "GEMV.h"
#include "ap_int.h"

void WP521_4096(ap_int<4> w0[NUM_DSP],
                ap_int<4> w1[NUM_DSP],
                ap_uint<4> a0[DSP_ARRAY_ROWS],
                ap_uint<4> a1[DSP_ARRAY_ROWS],
                ap_int<8> result_a0w0[NUM_DSP],
                ap_int<8> result_a1w0[NUM_DSP],
                ap_int<8> result_a0w1[NUM_DSP],
                ap_int<8> result_a1w1[NUM_DSP]) {
#pragma HLS inline off

    for (int i = 0; i < NUM_DSP; i++) {
#pragma HLS UNROLL
        const int row = i % DSP_ARRAY_ROWS;
        const ap_int<8> w0_val = w0[i];
        const ap_int<8> w1_val = w1[i];
        const ap_int<8> a0_val = a0[row];
        const ap_int<8> a1_val = a1[row];

        result_a0w0[i] = a0_val * w0_val;
        result_a1w0[i] = a1_val * w0_val;
        result_a0w1[i] = a0_val * w1_val;
        result_a1w1[i] = a1_val * w1_val;
    }
}
