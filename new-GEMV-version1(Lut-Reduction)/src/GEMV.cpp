#include "GEMV.h"

static ap_uint<4> encode_activation(ap_uint<4> raw) {
#pragma HLS INLINE
    // Preserve the original -8 -> -7 clamp. All other values remain in
    // four-bit two's-complement form for the overpacked RTL core.
    return raw == 8 ? (ap_uint<4>)9 : raw;
}

template <int WORD_INDEX>
static void unpack_weight_word(
    const axi_word_t raw_words[NUM_WEIGHT_AXI_PORTS],
    packed_weight_group_t weights[P_WEIGHTS_PER_PE][NUM_PE_GROUPS]) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=raw_words complete dim=1
#pragma HLS ARRAY_PARTITION variable=weights complete dim=0

UNPACK_WEIGHT_PORTS:
    for (int port = 0; port < NUM_WEIGHT_AXI_PORTS; ++port) {
#pragma HLS UNROLL
    UNPACK_WEIGHT_BYTES:
        for (int byte = 0; byte < VALUES_PER_AXI_WORD; ++byte) {
#pragma HLS UNROLL
            const int element = WORD_INDEX * VALUES_PER_AXI_WORD + byte;
            const int local_col = element / (DSP_ARRAY_ROWS * P_WEIGHTS_PER_PE);
            const int within_col = element % (DSP_ARRAY_ROWS * P_WEIGHTS_PER_PE);
            const int row = within_col / P_WEIGHTS_PER_PE;
            const int weight_lane = within_col % P_WEIGHTS_PER_PE;
            const int logical_col =
                port * LOGICAL_COLS_PER_WEIGHT_PORT + local_col;
            const int group = logical_col / COLS_PER_PE_GROUP;
            const int group_col = logical_col % COLS_PER_PE_GROUP;
            const int local_pe = group_col * DSP_ARRAY_ROWS + row;

            weights[weight_lane][group].range(
                local_pe * WEIGHT_BITS + WEIGHT_BITS - 1,
                local_pe * WEIGHT_BITS) =
                raw_words[port].range(byte * 8 + 3, byte * 8);
        }
    }
}

template <int PORT_ID>
static void load_weight_port(
    const axi_word_t *weight_port,
    hls::stream<axi_word_t> &weight_words) {
#pragma HLS INLINE off

// PORT_ID gives every HBM master its own dataflow process and control pipeline.
// The downstream FIFO decouples return skew between otherwise parallel ports.
LOAD_WEIGHT_PORT:
    for (int word = 0; word < WEIGHT_AXI_WORDS; ++word) {
#pragma HLS PIPELINE II=1
        weight_words.write(weight_port[word]);
    }
}

static void repack_weight_streams(
    hls::stream<axi_word_t> weight_words[NUM_WEIGHT_AXI_PORTS],
    packed_weight_group_t weights[P_WEIGHTS_PER_PE][NUM_PE_GROUPS]) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=weight_words complete dim=1
#pragma HLS ARRAY_PARTITION variable=weights complete dim=0

REPACK_WEIGHT_WORDS:
    for (int word = 0; word < WEIGHT_AXI_WORDS; ++word) {
#pragma HLS PIPELINE II=1 rewind
        axi_word_t raw_words[NUM_WEIGHT_AXI_PORTS];
#pragma HLS ARRAY_PARTITION variable=raw_words complete dim=1
    READ_WEIGHT_STREAMS:
        for (int port = 0; port < NUM_WEIGHT_AXI_PORTS; ++port) {
#pragma HLS UNROLL
            raw_words[port] = weight_words[port].read();
        }

        switch (word) {
        case 0: unpack_weight_word<0>(raw_words, weights); break;
        case 1: unpack_weight_word<1>(raw_words, weights); break;
        case 2: unpack_weight_word<2>(raw_words, weights); break;
        case 3: unpack_weight_word<3>(raw_words, weights); break;
        case 4: unpack_weight_word<4>(raw_words, weights); break;
        default: unpack_weight_word<5>(raw_words, weights); break;
        }
    }
}

static void load_activation_words(
    const axi_word_t *act_packed,
    hls::stream<axi_word_t> &activation_words) {
#pragma HLS INLINE off

LOAD_ACTIVATION_WORDS:
    for (int group = 0; group < P_ACT_GROUPS; ++group) {
#pragma HLS PIPELINE II=1
        activation_words.write(act_packed[group]);
    }
}

static void encode_activation_words(
    hls::stream<axi_word_t> &activation_words,
    packed_act_t acts[P_ACT_GROUPS]) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=acts complete dim=1

ENCODE_ACTIVATION_WORDS:
    for (int group = 0; group < P_ACT_GROUPS; ++group) {
#pragma HLS PIPELINE II=1 rewind
        const axi_word_t raw_act = activation_words.read();
        packed_act_t packed_act = 0;

    ENCODE_ACT_ROWS:
        for (int row = 0; row < DSP_ARRAY_ROWS; ++row) {
#pragma HLS UNROLL
            packed_act.range(row * ACT_BITS + ACT_BITS - 1, row * ACT_BITS) =
                encode_activation(raw_act.range(row * 8 + 3, row * 8));
        }
        acts[group] = packed_act;
    }
}

// Keep the blackbox call in one pipelined task. Scalar ports connect directly
// to the array, avoiding HLS stream FIFOs whose depth-one implementation cannot
// push and pop in the same cycle. The complete 64x64x9 array therefore has a
// real transaction interval of one cycle, including its HLS control wrapper.
static void compute_gemv(
    const packed_weight_group_t weights[P_WEIGHTS_PER_PE][NUM_PE_GROUPS],
    const packed_act_t acts[P_ACT_GROUPS],
    packed_channel_sum_t sums[P_RESULT_CHANNELS]) {
#pragma HLS INLINE off
#pragma HLS PIPELINE II=1
#pragma HLS ARRAY_PARTITION variable=weights complete dim=0
#pragma HLS ARRAY_PARTITION variable=acts complete dim=1
#pragma HLS ARRAY_PARTITION variable=sums complete dim=1

    const packed_core_sums_t packed_sums = W4A4_P(
        weights[0][0], weights[0][1], weights[0][2], weights[0][3],
        weights[1][0], weights[1][1], weights[1][2], weights[1][3],
        weights[2][0], weights[2][1], weights[2][2], weights[2][3],
        acts[0], acts[1], acts[2]);

UNPACK_COMPUTE_CHANNELS:
    for (int channel = 0; channel < P_RESULT_CHANNELS; ++channel) {
#pragma HLS UNROLL
        packed_channel_sum_t channel_sum = 0;

    UNPACK_COMPUTE_COLUMNS:
        for (int col = 0; col < DSP_ARRAY_COLS; ++col) {
#pragma HLS UNROLL
            const int core_lsb =
                (channel * DSP_ARRAY_COLS + col) * CORE_SUM_BITS;
            const ap_int<CORE_SUM_BITS> value =
                (ap_int<CORE_SUM_BITS>)packed_sums.range(
                    core_lsb + CORE_SUM_BITS - 1, core_lsb);
            channel_sum.range(
                col * SUM_BITS + SUM_BITS - 1,
                col * SUM_BITS) = (ap_uint<SUM_BITS>)(ap_int<SUM_BITS>)value;
        }
        sums[channel] = channel_sum;
    }
}

static void stage_results(
    const packed_channel_sum_t packed_sums[P_RESULT_CHANNELS],
    axi_word_t result_buffer[OUTPUT_AXI_WORDS]) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=packed_sums complete dim=1

STAGE_CHANNELS:
    for (int channel = 0; channel < P_RESULT_CHANNELS; ++channel) {
        packed_channel_sum_t remaining = packed_sums[channel];

STAGE_CHANNEL_WORDS:
        for (int channel_word = 0;
             channel_word < DSP_ARRAY_COLS / OUTPUTS_PER_AXI_WORD;
             ++channel_word) {
#pragma HLS PIPELINE II=1 rewind
            axi_word_t output_word = 0;

        PACK_OUTPUT_LANES:
            for (int lane = 0; lane < OUTPUTS_PER_AXI_WORD; ++lane) {
#pragma HLS UNROLL
                const ap_int<SUM_BITS> value =
                    (ap_int<SUM_BITS>)remaining.range(
                        lane * SUM_BITS + SUM_BITS - 1,
                        lane * SUM_BITS);
                output_word.range(lane * 32 + 31, lane * 32) =
                    (ap_uint<32>)(ap_int<32>)value;
            }
            const int output_word_index =
                channel * (DSP_ARRAY_COLS / OUTPUTS_PER_AXI_WORD) +
                channel_word;
            result_buffer[output_word_index] = output_word;
            remaining >>= OUTPUTS_PER_AXI_WORD * SUM_BITS;
        }
    }
}

static void write_results(
    axi_word_t *output,
    const axi_word_t result_buffer[OUTPUT_AXI_WORDS]) {
#pragma HLS INLINE off

WRITE_OUTPUT_WORDS:
    for (int word = 0; word < OUTPUT_AXI_WORDS; ++word) {
#pragma HLS PIPELINE II=1 rewind
        output[word] = result_buffer[word];
    }
}

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
    const axi_word_t *weight_col31) {
// Six consecutive weight beats are issued before their data is consumed. The
// 16-cycle request window avoids the Vitis default 64-entry request FIFOs.
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem_out depth=OUTPUT_AXI_WORDS latency=16 num_write_outstanding=1 max_write_burst_length=64
#pragma HLS INTERFACE m_axi port=act_packed offset=slave bundle=gmem_act depth=ACT_AXI_WORDS latency=16 num_read_outstanding=1 max_read_burst_length=4
#pragma HLS INTERFACE m_axi port=weight_col0 offset=slave bundle=gmem_w0 depth=WEIGHT_AXI_WORDS latency=16 num_read_outstanding=1 max_read_burst_length=8
#pragma HLS INTERFACE m_axi port=weight_col1 offset=slave bundle=gmem_w1 depth=WEIGHT_AXI_WORDS latency=16 num_read_outstanding=1 max_read_burst_length=8
#pragma HLS INTERFACE m_axi port=weight_col2 offset=slave bundle=gmem_w2 depth=WEIGHT_AXI_WORDS latency=16 num_read_outstanding=1 max_read_burst_length=8
#pragma HLS INTERFACE m_axi port=weight_col3 offset=slave bundle=gmem_w3 depth=WEIGHT_AXI_WORDS latency=16 num_read_outstanding=1 max_read_burst_length=8
#pragma HLS INTERFACE m_axi port=weight_col4 offset=slave bundle=gmem_w4 depth=WEIGHT_AXI_WORDS latency=16 num_read_outstanding=1 max_read_burst_length=8
#pragma HLS INTERFACE m_axi port=weight_col5 offset=slave bundle=gmem_w5 depth=WEIGHT_AXI_WORDS latency=16 num_read_outstanding=1 max_read_burst_length=8
#pragma HLS INTERFACE m_axi port=weight_col6 offset=slave bundle=gmem_w6 depth=WEIGHT_AXI_WORDS latency=16 num_read_outstanding=1 max_read_burst_length=8
#pragma HLS INTERFACE m_axi port=weight_col7 offset=slave bundle=gmem_w7 depth=WEIGHT_AXI_WORDS latency=16 num_read_outstanding=1 max_read_burst_length=8
#pragma HLS INTERFACE m_axi port=weight_col8 offset=slave bundle=gmem_w8 depth=WEIGHT_AXI_WORDS latency=16 num_read_outstanding=1 max_read_burst_length=8
#pragma HLS INTERFACE m_axi port=weight_col9 offset=slave bundle=gmem_w9 depth=WEIGHT_AXI_WORDS latency=16 num_read_outstanding=1 max_read_burst_length=8
#pragma HLS INTERFACE m_axi port=weight_col10 offset=slave bundle=gmem_w10 depth=WEIGHT_AXI_WORDS latency=16 num_read_outstanding=1 max_read_burst_length=8
#pragma HLS INTERFACE m_axi port=weight_col11 offset=slave bundle=gmem_w11 depth=WEIGHT_AXI_WORDS latency=16 num_read_outstanding=1 max_read_burst_length=8
#pragma HLS INTERFACE m_axi port=weight_col12 offset=slave bundle=gmem_w12 depth=WEIGHT_AXI_WORDS latency=16 num_read_outstanding=1 max_read_burst_length=8
#pragma HLS INTERFACE m_axi port=weight_col13 offset=slave bundle=gmem_w13 depth=WEIGHT_AXI_WORDS latency=16 num_read_outstanding=1 max_read_burst_length=8
#pragma HLS INTERFACE m_axi port=weight_col14 offset=slave bundle=gmem_w14 depth=WEIGHT_AXI_WORDS latency=16 num_read_outstanding=1 max_read_burst_length=8
#pragma HLS INTERFACE m_axi port=weight_col15 offset=slave bundle=gmem_w15 depth=WEIGHT_AXI_WORDS latency=16 num_read_outstanding=1 max_read_burst_length=8
#pragma HLS INTERFACE m_axi port=weight_col16 offset=slave bundle=gmem_w16 depth=WEIGHT_AXI_WORDS latency=16 num_read_outstanding=1 max_read_burst_length=8
#pragma HLS INTERFACE m_axi port=weight_col17 offset=slave bundle=gmem_w17 depth=WEIGHT_AXI_WORDS latency=16 num_read_outstanding=1 max_read_burst_length=8
#pragma HLS INTERFACE m_axi port=weight_col18 offset=slave bundle=gmem_w18 depth=WEIGHT_AXI_WORDS latency=16 num_read_outstanding=1 max_read_burst_length=8
#pragma HLS INTERFACE m_axi port=weight_col19 offset=slave bundle=gmem_w19 depth=WEIGHT_AXI_WORDS latency=16 num_read_outstanding=1 max_read_burst_length=8
#pragma HLS INTERFACE m_axi port=weight_col20 offset=slave bundle=gmem_w20 depth=WEIGHT_AXI_WORDS latency=16 num_read_outstanding=1 max_read_burst_length=8
#pragma HLS INTERFACE m_axi port=weight_col21 offset=slave bundle=gmem_w21 depth=WEIGHT_AXI_WORDS latency=16 num_read_outstanding=1 max_read_burst_length=8
#pragma HLS INTERFACE m_axi port=weight_col22 offset=slave bundle=gmem_w22 depth=WEIGHT_AXI_WORDS latency=16 num_read_outstanding=1 max_read_burst_length=8
#pragma HLS INTERFACE m_axi port=weight_col23 offset=slave bundle=gmem_w23 depth=WEIGHT_AXI_WORDS latency=16 num_read_outstanding=1 max_read_burst_length=8
#pragma HLS INTERFACE m_axi port=weight_col24 offset=slave bundle=gmem_w24 depth=WEIGHT_AXI_WORDS latency=16 num_read_outstanding=1 max_read_burst_length=8
#pragma HLS INTERFACE m_axi port=weight_col25 offset=slave bundle=gmem_w25 depth=WEIGHT_AXI_WORDS latency=16 num_read_outstanding=1 max_read_burst_length=8
#pragma HLS INTERFACE m_axi port=weight_col26 offset=slave bundle=gmem_w26 depth=WEIGHT_AXI_WORDS latency=16 num_read_outstanding=1 max_read_burst_length=8
#pragma HLS INTERFACE m_axi port=weight_col27 offset=slave bundle=gmem_w27 depth=WEIGHT_AXI_WORDS latency=16 num_read_outstanding=1 max_read_burst_length=8
#pragma HLS INTERFACE m_axi port=weight_col28 offset=slave bundle=gmem_w28 depth=WEIGHT_AXI_WORDS latency=16 num_read_outstanding=1 max_read_burst_length=8
#pragma HLS INTERFACE m_axi port=weight_col29 offset=slave bundle=gmem_w29 depth=WEIGHT_AXI_WORDS latency=16 num_read_outstanding=1 max_read_burst_length=8
#pragma HLS INTERFACE m_axi port=weight_col30 offset=slave bundle=gmem_w30 depth=WEIGHT_AXI_WORDS latency=16 num_read_outstanding=1 max_read_burst_length=8
#pragma HLS INTERFACE m_axi port=weight_col31 offset=slave bundle=gmem_w31 depth=WEIGHT_AXI_WORDS latency=16 num_read_outstanding=1 max_read_burst_length=8
#pragma HLS DATAFLOW

    packed_weight_group_t weights[P_WEIGHTS_PER_PE][NUM_PE_GROUPS];
    packed_act_t acts[P_ACT_GROUPS];
    packed_channel_sum_t sums[P_RESULT_CHANNELS];
    axi_word_t result_buffer[OUTPUT_AXI_WORDS];
    hls::stream<axi_word_t> weight_words[NUM_WEIGHT_AXI_PORTS];
    hls::stream<axi_word_t> activation_words;
#pragma HLS ARRAY_PARTITION variable=weights complete dim=0
#pragma HLS ARRAY_PARTITION variable=acts complete dim=1
#pragma HLS ARRAY_PARTITION variable=sums complete dim=1
#pragma HLS ARRAY_PARTITION variable=weight_words complete dim=1
#pragma HLS STREAM variable=weight_words depth=8
#pragma HLS STREAM variable=activation_words depth=4
#pragma HLS BIND_STORAGE variable=weight_words type=fifo impl=bram
#pragma HLS BIND_STORAGE variable=activation_words type=fifo impl=bram
#pragma HLS BIND_STORAGE variable=result_buffer type=ram_2p impl=bram

    load_weight_port<0>(weight_col0, weight_words[0]);
    load_weight_port<1>(weight_col1, weight_words[1]);
    load_weight_port<2>(weight_col2, weight_words[2]);
    load_weight_port<3>(weight_col3, weight_words[3]);
    load_weight_port<4>(weight_col4, weight_words[4]);
    load_weight_port<5>(weight_col5, weight_words[5]);
    load_weight_port<6>(weight_col6, weight_words[6]);
    load_weight_port<7>(weight_col7, weight_words[7]);
    load_weight_port<8>(weight_col8, weight_words[8]);
    load_weight_port<9>(weight_col9, weight_words[9]);
    load_weight_port<10>(weight_col10, weight_words[10]);
    load_weight_port<11>(weight_col11, weight_words[11]);
    load_weight_port<12>(weight_col12, weight_words[12]);
    load_weight_port<13>(weight_col13, weight_words[13]);
    load_weight_port<14>(weight_col14, weight_words[14]);
    load_weight_port<15>(weight_col15, weight_words[15]);
    load_weight_port<16>(weight_col16, weight_words[16]);
    load_weight_port<17>(weight_col17, weight_words[17]);
    load_weight_port<18>(weight_col18, weight_words[18]);
    load_weight_port<19>(weight_col19, weight_words[19]);
    load_weight_port<20>(weight_col20, weight_words[20]);
    load_weight_port<21>(weight_col21, weight_words[21]);
    load_weight_port<22>(weight_col22, weight_words[22]);
    load_weight_port<23>(weight_col23, weight_words[23]);
    load_weight_port<24>(weight_col24, weight_words[24]);
    load_weight_port<25>(weight_col25, weight_words[25]);
    load_weight_port<26>(weight_col26, weight_words[26]);
    load_weight_port<27>(weight_col27, weight_words[27]);
    load_weight_port<28>(weight_col28, weight_words[28]);
    load_weight_port<29>(weight_col29, weight_words[29]);
    load_weight_port<30>(weight_col30, weight_words[30]);
    load_weight_port<31>(weight_col31, weight_words[31]);

    repack_weight_streams(weight_words, weights);

    load_activation_words(act_packed, activation_words);
    encode_activation_words(activation_words, acts);

    compute_gemv(weights, acts, sums);

    stage_results(sums, result_buffer);

    write_results(output, result_buffer);
}
