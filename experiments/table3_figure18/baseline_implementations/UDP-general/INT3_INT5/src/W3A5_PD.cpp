#include "ap_int.h"

static const int ARRAY_ROWS = 128;
static const int ARRAY_COLS = 32;
static const int NUM_PE = ARRAY_ROWS * ARRAY_COLS;

// W3A5_array black-box function reference model
// Single computation mode (NO mode signal).
// - w0,w1,w2 are flattened per-PE weights in column-major order:
//   pe_index = col * ARRAY_ROWS + row
// - a0,a1 are row vectors shared across all columns
// - result0..result5 keep one 8-bit result per PE using the same layout
//
// result0=w0*a0, result1=w1*a0, result2=w2*a0,
// result3=w0*a1, result4=w1*a1, result5=w2*a1
void W3A5_array(
    ap_int<3> w0[NUM_PE],
    ap_int<3> w1[NUM_PE],
    ap_int<3> w2[NUM_PE],
    ap_int<5> a0[ARRAY_ROWS],
    ap_int<5> a1[ARRAY_ROWS],
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
            const ap_int<5> a0_row = a0[row];
            const ap_int<5> a1_row = a1[row];

            result0[pe_index] = (ap_int<8>)w0[pe_index] * (ap_int<8>)a0_row;
            result1[pe_index] = (ap_int<8>)w1[pe_index] * (ap_int<8>)a0_row;
            result2[pe_index] = (ap_int<8>)w2[pe_index] * (ap_int<8>)a0_row;
            result3[pe_index] = (ap_int<8>)w0[pe_index] * (ap_int<8>)a1_row;
            result4[pe_index] = (ap_int<8>)w1[pe_index] * (ap_int<8>)a1_row;
            result5[pe_index] = (ap_int<8>)w2[pe_index] * (ap_int<8>)a1_row;
        }
    }
}
