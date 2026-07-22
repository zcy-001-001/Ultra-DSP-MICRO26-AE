#include "ap_int.h"

static const int ARRAY_ROWS = 64;
static const int ARRAY_COLS = 64;
static const int NUM_PE = ARRAY_ROWS * ARRAY_COLS;

// Hybrid_INT4_INT4_PD black-box function reference model
// - mode is shared by the whole ARRAY_ROWS x ARRAY_COLS array
// - w1~w7 are flattened per-PE weights in column-major order:
//   pe_index = col * ARRAY_ROWS + row
// - a1~a3 are row vectors shared across all columns
// - result1~result9 keep one 7-bit result per PE using the same layout
void Hybrid_INT4_INT4_PD(
    ap_uint<1> mode,
    ap_int<4> w1[NUM_PE],
    ap_int<4> w2[NUM_PE],
    ap_int<4> w3[NUM_PE],
    ap_int<4> w4[NUM_PE],
    ap_int<4> w5[NUM_PE],
    ap_int<4> w6[NUM_PE],
    ap_int<4> w7[NUM_PE],
    ap_int<4> a1[ARRAY_ROWS],
    ap_int<4> a2[ARRAY_ROWS],
    ap_int<4> a3[ARRAY_ROWS],
    ap_int<7> result1[NUM_PE],
    ap_int<7> result2[NUM_PE],
    ap_int<7> result3[NUM_PE],
    ap_int<7> result4[NUM_PE],
    ap_int<7> result5[NUM_PE],
    ap_int<7> result6[NUM_PE],
    ap_int<7> result7[NUM_PE],
    ap_int<7> result8[NUM_PE],
    ap_int<7> result9[NUM_PE]) {
#pragma HLS inline off

    for (int col = 0; col < ARRAY_COLS; ++col) {
        for (int row = 0; row < ARRAY_ROWS; ++row) {
            const int pe_index = (col * ARRAY_ROWS) + row;
            const ap_int<4> a1_row = a1[row];
            const ap_int<4> a2_row = a2[row];
            const ap_int<4> a3_row = a3[row];

            if (mode == 0) {
                result1[pe_index] = (ap_int<7>)w1[pe_index] * (ap_int<7>)a1_row;
                result2[pe_index] = (ap_int<7>)w1[pe_index] * (ap_int<7>)a2_row;
                result3[pe_index] = (ap_int<7>)w1[pe_index] * (ap_int<7>)a3_row;
                result4[pe_index] = (ap_int<7>)w2[pe_index] * (ap_int<7>)a1_row;
                result5[pe_index] = (ap_int<7>)w2[pe_index] * (ap_int<7>)a2_row;
                result6[pe_index] = (ap_int<7>)w2[pe_index] * (ap_int<7>)a3_row;
                result7[pe_index] = (ap_int<7>)w3[pe_index] * (ap_int<7>)a1_row;
                result8[pe_index] = (ap_int<7>)w3[pe_index] * (ap_int<7>)a2_row;
                result9[pe_index] = (ap_int<7>)w3[pe_index] * (ap_int<7>)a3_row;
            } else {
                result1[pe_index] = (ap_int<7>)w1[pe_index] * (ap_int<7>)a1_row;
                result2[pe_index] = (ap_int<7>)w2[pe_index] * (ap_int<7>)a1_row;
                result3[pe_index] = (ap_int<7>)w3[pe_index] * (ap_int<7>)a1_row;
                result4[pe_index] = (ap_int<7>)w4[pe_index] * (ap_int<7>)a1_row;
                result5[pe_index] = (ap_int<7>)w5[pe_index] * (ap_int<7>)a1_row;
                result6[pe_index] = (ap_int<7>)w6[pe_index] * (ap_int<7>)a1_row;
                result7[pe_index] = (ap_int<7>)w7[pe_index] * (ap_int<7>)a1_row;
                result8[pe_index] = 0;
                result9[pe_index] = 0;
            }
        }
    }
}
