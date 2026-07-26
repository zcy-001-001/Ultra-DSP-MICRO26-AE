#ifndef GEMV_H
#define GEMV_H

#ifndef AP_INT_MAX_W
#define AP_INT_MAX_W 16384
#endif
#include "ap_int.h"
#include "hls_stream.h"

#define DSP_ARRAY_ROWS 64
#define DSP_ARRAY_COLS 64
#define NUM_DSP (DSP_ARRAY_ROWS * DSP_ARRAY_COLS)

#define NUM_PE_GROUPS 4
#define PE_GROUP_SIZE (NUM_DSP / NUM_PE_GROUPS)
#define COLS_PER_PE_GROUP (DSP_ARRAY_COLS / NUM_PE_GROUPS)

#define NUM_WEIGHT_AXI_PORTS 32
#define LOGICAL_COLS_PER_WEIGHT_PORT 2
#define PES_PER_WEIGHT_PORT \
    (LOGICAL_COLS_PER_WEIGHT_PORT * DSP_ARRAY_ROWS)

#define P_WEIGHTS_PER_PE 3
#define P_ACT_GROUPS 3
#define P_RESULT_CHANNELS (P_WEIGHTS_PER_PE * P_ACT_GROUPS)
#define ACT_COUNT (P_ACT_GROUPS * DSP_ARRAY_ROWS)
#define OUTPUT_COUNT (P_RESULT_CHANNELS * DSP_ARRAY_COLS)
#define WEIGHT_PORT_DEPTH (PES_PER_WEIGHT_PORT * P_WEIGHTS_PER_PE)

// The existing host layout stores each ap_int<4> in the low nibble of one
// byte. A 512-bit AXI word therefore carries 64 existing elements.
#define AXI_WORD_BITS 512
#define VALUES_PER_AXI_WORD (AXI_WORD_BITS / 8)
#define WEIGHT_AXI_WORDS \
    (WEIGHT_PORT_DEPTH / VALUES_PER_AXI_WORD)
#define ACT_AXI_WORDS (ACT_COUNT / VALUES_PER_AXI_WORD)
#define OUTPUTS_PER_AXI_WORD (AXI_WORD_BITS / 32)
#define OUTPUT_AXI_WORDS (OUTPUT_COUNT / OUTPUTS_PER_AXI_WORD)

#define WEIGHT_BITS 4
#define ACT_BITS 4
#define ACT_BUS_BITS (DSP_ARRAY_ROWS * ACT_BITS)
#define WEIGHT_GROUP_BITS (PE_GROUP_SIZE * WEIGHT_BITS)
#define CORE_SUM_BITS 13
#define CORE_CHANNEL_SUM_BITS (DSP_ARRAY_COLS * CORE_SUM_BITS)
#define CORE_ALL_SUM_BITS (P_RESULT_CHANNELS * CORE_CHANNEL_SUM_BITS)
#define SUM_BITS 16
#define CHANNEL_SUM_BITS (DSP_ARRAY_COLS * SUM_BITS)

typedef ap_uint<AXI_WORD_BITS> axi_word_t;
typedef ap_uint<WEIGHT_GROUP_BITS> packed_weight_group_t;
typedef ap_uint<ACT_BUS_BITS> packed_act_t;
typedef ap_uint<CHANNEL_SUM_BITS> packed_channel_sum_t;
typedef ap_uint<CORE_ALL_SUM_BITS> packed_core_sums_t;
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
    packed_act_t a3);

// Results are channel-major. Channel = weight_lane * P_ACT_GROUPS + act_group,
// followed by all 64 columns. Every 64-row product reduction remains separate.
// The C pointer type consumes and produces complete 512-bit AXI beats.
extern "C" void gemv_kernel(
    axi_word_t *output,
    const axi_word_t *act_packed,
    const axi_word_t *weight_col0,
    const axi_word_t *weight_col1,
    const axi_word_t *weight_col2,
    const axi_word_t *weight_col3,
    const axi_word_t *weight_col4,
    const axi_word_t *weight_col5,
    const axi_word_t *weight_col6,
    const axi_word_t *weight_col7,
    const axi_word_t *weight_col8,
    const axi_word_t *weight_col9,
    const axi_word_t *weight_col10,
    const axi_word_t *weight_col11,
    const axi_word_t *weight_col12,
    const axi_word_t *weight_col13,
    const axi_word_t *weight_col14,
    const axi_word_t *weight_col15,
    const axi_word_t *weight_col16,
    const axi_word_t *weight_col17,
    const axi_word_t *weight_col18,
    const axi_word_t *weight_col19,
    const axi_word_t *weight_col20,
    const axi_word_t *weight_col21,
    const axi_word_t *weight_col22,
    const axi_word_t *weight_col23,
    const axi_word_t *weight_col24,
    const axi_word_t *weight_col25,
    const axi_word_t *weight_col26,
    const axi_word_t *weight_col27,
    const axi_word_t *weight_col28,
    const axi_word_t *weight_col29,
    const axi_word_t *weight_col30,
    const axi_word_t *weight_col31);

#endif // GEMV_H
