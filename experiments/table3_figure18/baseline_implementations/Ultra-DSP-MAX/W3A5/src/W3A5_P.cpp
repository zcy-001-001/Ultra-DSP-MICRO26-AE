#include "ap_int.h"

static const int ARRAY_ROWS = 64;
static const int ARRAY_COLS = 64;
static const int NUM_PE = ARRAY_ROWS * ARRAY_COLS;

static ap_uint<4> abs_act(ap_int<5> act) {
    const ap_uint<5> raw = (ap_uint<5>)act;
    const ap_uint<5> mag = act[4] ? (ap_uint<5>)(~raw + 1) : raw;
    return (mag == 16) ? (ap_uint<4>)15 : mag.range(3, 0);
}

static ap_int<7> encode_result(ap_int<3> weight, ap_int<5> act) {
    const ap_uint<1> sign = weight[2] ^ act[4];
    const ap_uint<2> w_mag = ((ap_uint<3>)weight).range(1, 0);
    const ap_uint<4> a_mag = abs_act(act);
    const ap_uint<6> magnitude = (ap_uint<6>)w_mag * (ap_uint<6>)a_mag;
    ap_uint<7> payload = 0;
    payload.range(5, 0) = magnitude;
    const ap_uint<7> encoded = sign ? (payload ^ ~ap_uint<7>(0)) : payload;
    return (ap_int<7>)encoded;
}

static ap_uint<1> encode_sign(ap_int<3> weight, ap_int<5> act) {
    return weight[2] ^ act[4];
}

void W3A5_P(
    ap_int<3> w1[NUM_PE],
    ap_int<3> w2[NUM_PE],
    ap_int<3> w3[NUM_PE],
    ap_int<3> w4[NUM_PE],
    ap_int<3> w5[NUM_PE],
    ap_int<5> a1[ARRAY_ROWS],
    ap_int<5> a2[ARRAY_ROWS],
    ap_int<7> result1[NUM_PE],
    ap_int<7> result2[NUM_PE],
    ap_int<7> result3[NUM_PE],
    ap_int<7> result4[NUM_PE],
    ap_int<7> result5[NUM_PE],
    ap_int<7> result6[NUM_PE],
    ap_int<7> result7[NUM_PE],
    ap_int<7> result8[NUM_PE],
    ap_int<7> result9[NUM_PE],
    ap_int<7> result10[NUM_PE],
    ap_uint<1> sign1[NUM_PE],
    ap_uint<1> sign2[NUM_PE],
    ap_uint<1> sign3[NUM_PE],
    ap_uint<1> sign4[NUM_PE],
    ap_uint<1> sign5[NUM_PE],
    ap_uint<1> sign6[NUM_PE],
    ap_uint<1> sign7[NUM_PE],
    ap_uint<1> sign8[NUM_PE],
    ap_uint<1> sign9[NUM_PE],
    ap_uint<1> sign10[NUM_PE]) {
#pragma HLS inline off

    for (int col = 0; col < ARRAY_COLS; ++col) {
        for (int row = 0; row < ARRAY_ROWS; ++row) {
            const int pe_index = (col * ARRAY_ROWS) + row;
            const ap_int<5> a1_row = a1[row];
            const ap_int<5> a2_row = a2[row];

            result1[pe_index] = encode_result(w1[pe_index], a1_row);
            result2[pe_index] = encode_result(w2[pe_index], a1_row);
            result3[pe_index] = encode_result(w3[pe_index], a1_row);
            result4[pe_index] = encode_result(w4[pe_index], a1_row);
            result5[pe_index] = encode_result(w5[pe_index], a1_row);
            result6[pe_index] = encode_result(w1[pe_index], a2_row);
            result7[pe_index] = encode_result(w2[pe_index], a2_row);
            result8[pe_index] = encode_result(w3[pe_index], a2_row);
            result9[pe_index] = encode_result(w4[pe_index], a2_row);
            result10[pe_index] = encode_result(w5[pe_index], a2_row);

            sign1[pe_index] = encode_sign(w1[pe_index], a1_row);
            sign2[pe_index] = encode_sign(w2[pe_index], a1_row);
            sign3[pe_index] = encode_sign(w3[pe_index], a1_row);
            sign4[pe_index] = encode_sign(w4[pe_index], a1_row);
            sign5[pe_index] = encode_sign(w5[pe_index], a1_row);
            sign6[pe_index] = encode_sign(w1[pe_index], a2_row);
            sign7[pe_index] = encode_sign(w2[pe_index], a2_row);
            sign8[pe_index] = encode_sign(w3[pe_index], a2_row);
            sign9[pe_index] = encode_sign(w4[pe_index], a2_row);
            sign10[pe_index] = encode_sign(w5[pe_index], a2_row);
        }
    }
}
