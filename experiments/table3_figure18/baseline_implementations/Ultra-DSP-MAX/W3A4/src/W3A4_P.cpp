#include "ap_int.h"

static const int ARRAY_ROWS = 64;
static const int ARRAY_COLS = 64;
static const int NUM_PE = ARRAY_ROWS * ARRAY_COLS;

static ap_uint<3> abs_act(ap_int<4> act) {
    const ap_uint<4> raw = (ap_uint<4>)act;
    const ap_uint<4> mag = act[3] ? (ap_uint<4>)(~raw + 1) : raw;
    return (mag == 8) ? (ap_uint<3>)7 : mag.range(2, 0);
}

static ap_int<6> encode_result(ap_int<3> weight, ap_int<4> act) {
    const ap_uint<1> sign = weight[2] ^ act[3];
    const ap_uint<2> w_mag = ((ap_uint<3>)weight).range(1, 0);
    const ap_uint<3> a_mag = abs_act(act);
    const ap_uint<5> magnitude = (ap_uint<5>)w_mag * (ap_uint<5>)a_mag;
    ap_uint<6> payload = 0;
    payload.range(4, 0) = magnitude;
    const ap_uint<6> encoded = sign ? (payload ^ ~ap_uint<6>(0)) : payload;
    return (ap_int<6>)encoded;
}

static ap_uint<1> encode_sign(ap_int<3> weight, ap_int<4> act) {
    return weight[2] ^ act[3];
}

void W3A4_P(
    ap_int<3> w1[NUM_PE],
    ap_int<3> w2[NUM_PE],
    ap_int<3> w3[NUM_PE],
    ap_int<3> w4[NUM_PE],
    ap_int<3> w5[NUM_PE],
    ap_int<3> w6[NUM_PE],
    ap_int<4> a1[ARRAY_ROWS],
    ap_int<4> a2[ARRAY_ROWS],
    ap_int<6> result1[NUM_PE],
    ap_int<6> result2[NUM_PE],
    ap_int<6> result3[NUM_PE],
    ap_int<6> result4[NUM_PE],
    ap_int<6> result5[NUM_PE],
    ap_int<6> result6[NUM_PE],
    ap_int<6> result7[NUM_PE],
    ap_int<6> result8[NUM_PE],
    ap_int<6> result9[NUM_PE],
    ap_int<6> result10[NUM_PE],
    ap_int<6> result11[NUM_PE],
    ap_int<6> result12[NUM_PE],
    ap_uint<1> sign1[NUM_PE],
    ap_uint<1> sign2[NUM_PE],
    ap_uint<1> sign3[NUM_PE],
    ap_uint<1> sign4[NUM_PE],
    ap_uint<1> sign5[NUM_PE],
    ap_uint<1> sign6[NUM_PE],
    ap_uint<1> sign7[NUM_PE],
    ap_uint<1> sign8[NUM_PE],
    ap_uint<1> sign9[NUM_PE],
    ap_uint<1> sign10[NUM_PE],
    ap_uint<1> sign11[NUM_PE],
    ap_uint<1> sign12[NUM_PE]) {
#pragma HLS inline off

    for (int col = 0; col < ARRAY_COLS; ++col) {
        for (int row = 0; row < ARRAY_ROWS; ++row) {
            const int pe_index = (col * ARRAY_ROWS) + row;
            const ap_int<4> a1_row = a1[row];
            const ap_int<4> a2_row = a2[row];

            result1[pe_index] = encode_result(w1[pe_index], a1_row);
            result2[pe_index] = encode_result(w2[pe_index], a1_row);
            result3[pe_index] = encode_result(w3[pe_index], a1_row);
            result4[pe_index] = encode_result(w4[pe_index], a1_row);
            result5[pe_index] = encode_result(w5[pe_index], a1_row);
            result6[pe_index] = encode_result(w6[pe_index], a1_row);
            result7[pe_index] = encode_result(w1[pe_index], a2_row);
            result8[pe_index] = encode_result(w2[pe_index], a2_row);
            result9[pe_index] = encode_result(w3[pe_index], a2_row);
            result10[pe_index] = encode_result(w4[pe_index], a2_row);
            result11[pe_index] = encode_result(w5[pe_index], a2_row);
            result12[pe_index] = encode_result(w6[pe_index], a2_row);

            sign1[pe_index] = encode_sign(w1[pe_index], a1_row);
            sign2[pe_index] = encode_sign(w2[pe_index], a1_row);
            sign3[pe_index] = encode_sign(w3[pe_index], a1_row);
            sign4[pe_index] = encode_sign(w4[pe_index], a1_row);
            sign5[pe_index] = encode_sign(w5[pe_index], a1_row);
            sign6[pe_index] = encode_sign(w6[pe_index], a1_row);
            sign7[pe_index] = encode_sign(w1[pe_index], a2_row);
            sign8[pe_index] = encode_sign(w2[pe_index], a2_row);
            sign9[pe_index] = encode_sign(w3[pe_index], a2_row);
            sign10[pe_index] = encode_sign(w4[pe_index], a2_row);
            sign11[pe_index] = encode_sign(w5[pe_index], a2_row);
            sign12[pe_index] = encode_sign(w6[pe_index], a2_row);
        }
    }
}
