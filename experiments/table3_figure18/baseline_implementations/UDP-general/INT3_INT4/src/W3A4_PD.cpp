#include "ap_int.h"

static const int ARRAY_ROWS = 128;
static const int ARRAY_COLS = 32;
static const int NUM_PE = ARRAY_ROWS * ARRAY_COLS;

// W3A4_array black-box function reference model
// - NO mode signal (single computation mode)
// - w0, w1, w2 are flattened per-PE weights (signed 4-bit) in column-major order:
//   pe_index = col * ARRAY_ROWS + row
// - a0, a1 are row vectors (signed 3-bit) shared across all columns
// - result0..result5 keep one 7-bit signed result per PE using the same layout
// - result0=w0*a0, result1=w1*a0, result2=w2*a0
// - result3=w0*a1, result4=w1*a1, result5=w2*a1
void W3A4_array(
    ap_int<4> w0[NUM_PE],
    ap_int<4> w1[NUM_PE],
    ap_int<4> w2[NUM_PE],
    ap_int<3> a0[ARRAY_ROWS],
    ap_int<3> a1[ARRAY_ROWS],
    ap_int<7> result0[NUM_PE],
    ap_int<7> result1[NUM_PE],
    ap_int<7> result2[NUM_PE],
    ap_int<7> result3[NUM_PE],
    ap_int<7> result4[NUM_PE],
    ap_int<7> result5[NUM_PE]) {
#pragma HLS inline off

    for (int col = 0; col < ARRAY_COLS; ++col) {
        for (int row = 0; row < ARRAY_ROWS; ++row) {
            const int pe_index = (col * ARRAY_ROWS) + row;
            const ap_int<3> a0_row = a0[row];
            const ap_int<3> a1_row = a1[row];

            result0[pe_index] = (ap_int<7>)w0[pe_index] * (ap_int<7>)a0_row;
            result1[pe_index] = (ap_int<7>)w1[pe_index] * (ap_int<7>)a0_row;
            result2[pe_index] = (ap_int<7>)w2[pe_index] * (ap_int<7>)a0_row;
            result3[pe_index] = (ap_int<7>)w0[pe_index] * (ap_int<7>)a1_row;
            result4[pe_index] = (ap_int<7>)w1[pe_index] * (ap_int<7>)a1_row;
            result5[pe_index] = (ap_int<7>)w2[pe_index] * (ap_int<7>)a1_row;
        }
    }
}
