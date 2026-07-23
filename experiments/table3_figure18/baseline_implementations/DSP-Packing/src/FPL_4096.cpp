#include "ap_int.h"

static const int ARRAY_ROWS = 128;
static const int ARRAY_COLS = 32;
static const int NUM_PE = ARRAY_ROWS * ARRAY_COLS;

void FPL_4096(ap_int<5> w0[NUM_PE],
              ap_int<5> w1[NUM_PE],
              ap_uint<4> a0[ARRAY_ROWS],
              ap_uint<4> a1[ARRAY_ROWS],
              ap_uint<4> a2[ARRAY_ROWS],
              ap_int<9> result_a0w0[NUM_PE],
              ap_int<9> result_a1w0[NUM_PE],
              ap_int<9> result_a2w0[NUM_PE],
              ap_int<9> result_a0w1[NUM_PE],
              ap_int<9> result_a1w1[NUM_PE],
              ap_int<9> result_a2w1[NUM_PE]) {
#pragma HLS inline off
    // Functional model for the blackbox: each PE returns the six signed
    // 5x4 products that the corrected RTL `FPL` instance produces.
    for (int col = 0; col < ARRAY_COLS; ++col) {
        for (int row = 0; row < ARRAY_ROWS; ++row) {
            const int pe_index = (col * ARRAY_ROWS) + row;
            const ap_int<5> w0_val = w0[pe_index];
            const ap_int<5> w1_val = w1[pe_index];
            const ap_uint<4> a0_val = a0[row];
            const ap_uint<4> a1_val = a1[row];
            const ap_uint<4> a2_val = a2[row];

            result_a0w0[pe_index] = (ap_int<9>)w0_val * (ap_int<9>)a0_val;
            result_a1w0[pe_index] = (ap_int<9>)w0_val * (ap_int<9>)a1_val;
            result_a2w0[pe_index] = (ap_int<9>)w0_val * (ap_int<9>)a2_val;
            result_a0w1[pe_index] = (ap_int<9>)w1_val * (ap_int<9>)a0_val;
            result_a1w1[pe_index] = (ap_int<9>)w1_val * (ap_int<9>)a1_val;
            result_a2w1[pe_index] = (ap_int<9>)w1_val * (ap_int<9>)a2_val;
        }
    }
}
