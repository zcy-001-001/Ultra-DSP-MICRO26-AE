#include "ap_int.h"

static const int ARRAY_ROWS = 64;
static const int ARRAY_COLS = 64;
static const int NUM_PE = ARRAY_ROWS * ARRAY_COLS;

static ap_uint<4> abs_act(ap_int<5> act) {
    const ap_uint<5> raw = (ap_uint<5>)act;
    const ap_uint<5> mag = act[4] ? (ap_uint<5>)(~raw + 1) : raw;
    return (mag == 16) ? (ap_uint<4>)15 : mag.range(3, 0);
}

static ap_int<8> encode_result(ap_int<5> weight, ap_int<5> act) {
    const ap_uint<1> sign = weight[4] ^ act[4];
    const ap_uint<4> w_mag = ((ap_uint<5>)weight).range(3, 0);
    const ap_uint<4> a_mag = abs_act(act);
    const ap_uint<8> magnitude = (ap_uint<8>)w_mag * (ap_uint<8>)a_mag;
    const ap_uint<8> encoded = sign ? (magnitude ^ ~ap_uint<8>(0)) : magnitude;
    return (ap_int<8>)encoded;
}

static ap_uint<1> encode_sign(ap_int<5> weight, ap_int<5> act) {
    return weight[4] ^ act[4];
}

void W5A5_P(
    ap_int<5> w1[NUM_PE],
    ap_int<5> w2[NUM_PE],
    ap_int<5> a1[ARRAY_ROWS],
    ap_int<5> a2[ARRAY_ROWS],
    ap_int<5> a3[ARRAY_ROWS],
    ap_int<8> result1[NUM_PE],
    ap_int<8> result2[NUM_PE],
    ap_int<8> result3[NUM_PE],
    ap_int<8> result4[NUM_PE],
    ap_int<8> result5[NUM_PE],
    ap_int<8> result6[NUM_PE],
    ap_uint<1> sign1[NUM_PE],
    ap_uint<1> sign2[NUM_PE],
    ap_uint<1> sign3[NUM_PE],
    ap_uint<1> sign4[NUM_PE],
    ap_uint<1> sign5[NUM_PE],
    ap_uint<1> sign6[NUM_PE]) {
#pragma HLS inline off

    for (int col = 0; col < ARRAY_COLS; ++col) {
        for (int row = 0; row < ARRAY_ROWS; ++row) {
            const int pe_index = (col * ARRAY_ROWS) + row;
            const ap_int<5> a1_row = a1[row];
            const ap_int<5> a2_row = a2[row];
            const ap_int<5> a3_row = a3[row];

            result1[pe_index] = encode_result(w1[pe_index], a1_row);
            result2[pe_index] = encode_result(w1[pe_index], a2_row);
            result3[pe_index] = encode_result(w1[pe_index], a3_row);
            result4[pe_index] = encode_result(w2[pe_index], a1_row);
            result5[pe_index] = encode_result(w2[pe_index], a2_row);
            result6[pe_index] = encode_result(w2[pe_index], a3_row);

            sign1[pe_index] = encode_sign(w1[pe_index], a1_row);
            sign2[pe_index] = encode_sign(w1[pe_index], a2_row);
            sign3[pe_index] = encode_sign(w1[pe_index], a3_row);
            sign4[pe_index] = encode_sign(w2[pe_index], a1_row);
            sign5[pe_index] = encode_sign(w2[pe_index], a2_row);
            sign6[pe_index] = encode_sign(w2[pe_index], a3_row);
        }
    }
}
