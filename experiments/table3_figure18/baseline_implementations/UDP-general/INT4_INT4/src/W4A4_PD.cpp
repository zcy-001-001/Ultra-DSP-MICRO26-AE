#include "ap_int.h"

static const int ARRAY_ROWS = 128;
static const int ARRAY_COLS = 32;
static const int NUM_PE = ARRAY_ROWS * ARRAY_COLS;

// W4A4 64x64 packed-DSP array reference model.
// - w0/w1/w2 are flattened per-PE weights in column-major order.
// - a0/a1 are row vectors shared across all columns.
// - Each PE produces 3 x 2 = 6 signed int8 products.
// - result0 = w0*a0, result1 = w1*a0, result2 = w2*a0
// - result3 = w0*a1, result4 = w1*a1, result5 = w2*a1
void W4A4_array(
    ap_int<4> w0[NUM_PE],
    ap_int<4> w1[NUM_PE],
    ap_int<4> w2[NUM_PE],
    ap_int<4> a0[ARRAY_ROWS],
    ap_int<4> a1[ARRAY_ROWS],
    ap_int<8> result0[NUM_PE],
    ap_int<8> result1[NUM_PE],
    ap_int<8> result2[NUM_PE],
    ap_int<8> result3[NUM_PE],
    ap_int<8> result4[NUM_PE],
    ap_int<8> result5[NUM_PE]) {
#pragma HLS inline off

    for (int col = 0; col < ARRAY_COLS; ++col) {
        for (int row = 0; row < ARRAY_ROWS; ++row) {
            const int pe_index = (col * ARRAY_ROWS) + row;
            const ap_int<4> a0_row = a0[row];
            const ap_int<4> a1_row = a1[row];

            result0[pe_index] = (ap_int<8>)w0[pe_index] * (ap_int<8>)a0_row;
            result1[pe_index] = (ap_int<8>)w1[pe_index] * (ap_int<8>)a0_row;
            result2[pe_index] = (ap_int<8>)w2[pe_index] * (ap_int<8>)a0_row;
            result3[pe_index] = (ap_int<8>)w0[pe_index] * (ap_int<8>)a1_row;
            result4[pe_index] = (ap_int<8>)w1[pe_index] * (ap_int<8>)a1_row;
            result5[pe_index] = (ap_int<8>)w2[pe_index] * (ap_int<8>)a1_row;
        }
    }
}
