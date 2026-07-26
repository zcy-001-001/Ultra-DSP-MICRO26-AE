#include "GEMV.h"

static ap_int<6> decode_weight(ap_uint<4> encoded) {
    const ap_int<6> magnitude = (ap_int<6>)encoded.range(2, 0);
    return encoded[3] ? (ap_int<6>)(-magnitude) : magnitude;
}

static ap_int<4> decode_activation(ap_uint<4> encoded) {
    const ap_int<4> value = (ap_int<4>)encoded;
    return value == -8 ? (ap_int<4>)-7 : value;
}

packed_core_sums_t W4A4_P(
    packed_weight_group_t w1g0,
    packed_weight_group_t w1g1,
    packed_weight_group_t w1g2,
    packed_weight_group_t w1g3,
    packed_weight_group_t w2g0,
    packed_weight_group_t w2g1,
    packed_weight_group_t w2g2,
    packed_weight_group_t w2g3,
    packed_weight_group_t w3g0,
    packed_weight_group_t w3g1,
    packed_weight_group_t w3g2,
    packed_weight_group_t w3g3,
    packed_act_t a1,
    packed_act_t a2,
    packed_act_t a3) {
    const packed_weight_group_t weights[P_WEIGHTS_PER_PE][NUM_PE_GROUPS] = {
        {w1g0, w1g1, w1g2, w1g3},
        {w2g0, w2g1, w2g2, w2g3},
        {w3g0, w3g1, w3g2, w3g3}};
    const packed_act_t acts[P_ACT_GROUPS] = {a1, a2, a3};
    packed_core_sums_t packed_sums = 0;

    for (int weight_lane = 0; weight_lane < P_WEIGHTS_PER_PE;
         ++weight_lane) {
        for (int act_group = 0; act_group < P_ACT_GROUPS; ++act_group) {
            const int channel = weight_lane * P_ACT_GROUPS + act_group;

        for (int col = 0; col < DSP_ARRAY_COLS; ++col) {
            const int pe_group = col / COLS_PER_PE_GROUP;
            const int local_col = col % COLS_PER_PE_GROUP;
            ap_int<32> channel_sum = 0;

            for (int row = 0; row < DSP_ARRAY_ROWS; ++row) {
                const int local_pe = local_col * DSP_ARRAY_ROWS + row;
                const int weight_lsb = local_pe * WEIGHT_BITS;
                const int act_lsb = row * ACT_BITS;
                const ap_uint<4> encoded_weight =
                    weights[weight_lane][pe_group].range(
                        weight_lsb + WEIGHT_BITS - 1, weight_lsb);
                const ap_uint<4> encoded_act =
                    acts[act_group].range(
                        act_lsb + ACT_BITS - 1, act_lsb);
                channel_sum += decode_weight(encoded_weight) *
                               decode_activation(encoded_act);
            }

            const int output_lsb =
                (channel * DSP_ARRAY_COLS + col) * CORE_SUM_BITS;
            packed_sums.range(
                output_lsb + CORE_SUM_BITS - 1,
                output_lsb) = (ap_uint<CORE_SUM_BITS>)channel_sum;
        }
        }
    }

    return packed_sums;
}
