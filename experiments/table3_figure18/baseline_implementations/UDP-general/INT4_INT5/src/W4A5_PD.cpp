#include "ap_int.h"

static const int ARRAY_ROWS = 128;
static const int ARRAY_COLS = 32;
static const int NUM_PE = ARRAY_ROWS * ARRAY_COLS;

// W4A5_array black-box function reference model
// - No mode signal (single mode).
// - w0/w1 are flattened per-PE weights in column-major order:
//   pe_index = col * ARRAY_ROWS + row
// - a0/a1 are row vectors shared across all columns.
// - 4 results per PE (9-bit signed):
//   result0 = w0*a0, result1 = w1*a0, result2 = w0*a1, result3 = w1*a1
void W4A5_array(
    ap_int<4> w0[NUM_PE],
    ap_int<4> w1[NUM_PE],
    ap_int<5> a0[ARRAY_ROWS],
    ap_int<5> a1[ARRAY_ROWS],
    ap_int<9> result0[NUM_PE],
    ap_int<9> result1[NUM_PE],
    ap_int<9> result2[NUM_PE],
    ap_int<9> result3[NUM_PE]) {
#pragma HLS inline off

    for (int col = 0; col < ARRAY_COLS; ++col) {
        for (int row = 0; row < ARRAY_ROWS; ++row) {
            const int pe_index = (col * ARRAY_ROWS) + row;
            const ap_int<5> a0_row = a0[row];
            const ap_int<5> a1_row = a1[row];

            result0[pe_index] = (ap_int<9>)w0[pe_index] * (ap_int<9>)a0_row;
            result1[pe_index] = (ap_int<9>)w1[pe_index] * (ap_int<9>)a0_row;
            result2[pe_index] = (ap_int<9>)w0[pe_index] * (ap_int<9>)a1_row;
            result3[pe_index] = (ap_int<9>)w1[pe_index] * (ap_int<9>)a1_row;
        }
    }
}
