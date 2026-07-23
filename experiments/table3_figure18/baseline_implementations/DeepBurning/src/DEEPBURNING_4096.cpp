#include "ap_int.h"

static const int ARRAY_ROWS = 128;
static const int ARRAY_COLS = 32;
static const int NUM_PE = ARRAY_ROWS * ARRAY_COLS;

void DEEPBURNING_4096(
    ap_int<4> w1[NUM_PE],
    ap_int<4> w2[NUM_PE],
    ap_int<4> a1[ARRAY_ROWS],
    ap_int<4> a2[ARRAY_ROWS],
    ap_int<4> a3[ARRAY_ROWS],
    ap_int<8> result1[NUM_PE],
    ap_int<8> result2[NUM_PE],
    ap_int<8> result3[NUM_PE],
    ap_int<8> result4[NUM_PE],
    ap_int<8> result5[NUM_PE],
    ap_int<8> result6[NUM_PE]
) {
#pragma HLS inline off

    for (int col = 0; col < ARRAY_COLS; ++col) {
        for (int row = 0; row < ARRAY_ROWS; ++row) {
            const int pe_index = (col * ARRAY_ROWS) + row;
            const ap_int<4> w1_val = w1[pe_index];
            const ap_int<4> w2_val = w2[pe_index];
            const ap_int<4> a1_val = a1[row];
            const ap_int<4> a2_val = a2[row];
            const ap_int<4> a3_val = a3[row];

            result1[pe_index] = (ap_int<8>)w1_val * (ap_int<8>)a1_val;
            result2[pe_index] = (ap_int<8>)w1_val * (ap_int<8>)a2_val;
            result3[pe_index] = (ap_int<8>)w1_val * (ap_int<8>)a3_val;
            result4[pe_index] = (ap_int<8>)w2_val * (ap_int<8>)a1_val;
            result5[pe_index] = (ap_int<8>)w2_val * (ap_int<8>)a2_val;
            result6[pe_index] = (ap_int<8>)w2_val * (ap_int<8>)a3_val;
        }
    }
}
