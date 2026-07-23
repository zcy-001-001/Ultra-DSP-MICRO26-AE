#include "GEMV.h"
#include "ap_int.h"

void DuoQ_4096(ap_int<4> w0[NUM_DSP],
               ap_int<4> w1[NUM_DSP],
               ap_int<4> w2[NUM_DSP],
               ap_int<4> w3[NUM_DSP],
               ap_int<4> a_in[DSP_ARRAY_ROWS],
               ap_int<8> p0[NUM_DSP],
               ap_int<8> p1[NUM_DSP],
               ap_int<8> p2[NUM_DSP],
               ap_int<8> p3[NUM_DSP]) {
#pragma HLS inline off

    for (int i = 0; i < NUM_DSP; i++) {
#pragma HLS UNROLL
        const int row = i % DSP_ARRAY_ROWS;
        const ap_int<8> act = a_in[row];
        const ap_int<8> w0_val = w0[i];
        const ap_int<8> w1_val = w1[i];
        const ap_int<8> w2_val = w2[i];
        const ap_int<8> w3_val = w3[i];

        p0[i] = act * w0_val;
        p1[i] = act * w1_val;
        p2[i] = act * w2_val;
        p3[i] = act * w3_val;
    }
}
