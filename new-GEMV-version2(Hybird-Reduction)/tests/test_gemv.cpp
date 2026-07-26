#include "../src/GEMV.h"

#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

static int reference_product(uint8_t weight_raw, int act) {
    const int weight_sign = (weight_raw >> 3) & 1;
    const int weight_mag = weight_raw & 7;
    const int act_sign = act < 0;
    int act_mag = act_sign ? -act : act;
    if (act_mag == 8) {
        act_mag = 7;
    }
    const int magnitude = weight_mag * act_mag;
    return (weight_sign ^ act_sign) ? -magnitude : magnitude;
}

int main() {
    std::mt19937 rng(0x4a4);
    std::uniform_int_distribution<int> nibble(0, 15);

    std::vector<ap_int<4> > acts(ACT_COUNT);
    std::vector<std::vector<ap_int<4> > > weights(
        NUM_WEIGHT_AXI_PORTS,
        std::vector<ap_int<4> >(WEIGHT_PORT_DEPTH));
    std::vector<axi_word_t> packed_acts(ACT_AXI_WORDS);
    std::vector<std::vector<axi_word_t> > packed_weights(
        NUM_WEIGHT_AXI_PORTS,
        std::vector<axi_word_t>(WEIGHT_AXI_WORDS));
    std::vector<axi_word_t> output(OUTPUT_AXI_WORDS);

    for (int trial = 0; trial < 8; ++trial) {
        for (int i = 0; i < ACT_COUNT; ++i) {
            acts[i] = nibble(rng);
        }
        for (int port = 0; port < NUM_WEIGHT_AXI_PORTS; ++port) {
            for (int i = 0; i < WEIGHT_PORT_DEPTH; ++i) {
                weights[port][i] = nibble(rng);
            }
        }

        for (int word = 0; word < ACT_AXI_WORDS; ++word) {
            packed_acts[word] = 0;
        }
        for (int i = 0; i < ACT_COUNT; ++i) {
            const int word = i / VALUES_PER_AXI_WORD;
            const int byte = i % VALUES_PER_AXI_WORD;
            packed_acts[word].range(byte * 8 + 3, byte * 8) =
                (ap_uint<4>)acts[i];
        }

        for (int port = 0; port < NUM_WEIGHT_AXI_PORTS; ++port) {
            for (int word = 0; word < WEIGHT_AXI_WORDS; ++word) {
                packed_weights[port][word] = 0;
            }
            for (int i = 0; i < WEIGHT_PORT_DEPTH; ++i) {
                const int word = i / VALUES_PER_AXI_WORD;
                const int byte = i % VALUES_PER_AXI_WORD;
                packed_weights[port][word].range(
                    byte * 8 + 3, byte * 8) =
                    (ap_uint<4>)weights[port][i];
            }
        }

        gemv_kernel(
            output.data(), packed_acts.data(),
            packed_weights[0].data(), packed_weights[1].data(),
            packed_weights[2].data(), packed_weights[3].data(),
            packed_weights[4].data(), packed_weights[5].data(),
            packed_weights[6].data(), packed_weights[7].data(),
            packed_weights[8].data(), packed_weights[9].data(),
            packed_weights[10].data(), packed_weights[11].data(),
            packed_weights[12].data(), packed_weights[13].data(),
            packed_weights[14].data(), packed_weights[15].data(),
            packed_weights[16].data(), packed_weights[17].data(),
            packed_weights[18].data(), packed_weights[19].data(),
            packed_weights[20].data(), packed_weights[21].data(),
            packed_weights[22].data(), packed_weights[23].data(),
            packed_weights[24].data(), packed_weights[25].data(),
            packed_weights[26].data(), packed_weights[27].data(),
            packed_weights[28].data(), packed_weights[29].data(),
            packed_weights[30].data(), packed_weights[31].data());

        for (int lane = 0; lane < P_WEIGHTS_PER_PE; ++lane) {
        for (int group = 0; group < P_ACT_GROUPS; ++group) {
            for (int col = 0; col < DSP_ARRAY_COLS; ++col) {
                const int port = col / LOGICAL_COLS_PER_WEIGHT_PORT;
                const int local_col = col % LOGICAL_COLS_PER_WEIGHT_PORT;
                int expected = 0;

                for (int row = 0; row < DSP_ARRAY_ROWS; ++row) {
                    const int act = (int)acts[group * DSP_ARRAY_ROWS + row];
                    const int base =
                        local_col * DSP_ARRAY_ROWS * P_WEIGHTS_PER_PE +
                        row * P_WEIGHTS_PER_PE;

                    const uint8_t weight_raw =
                        (uint8_t)(ap_uint<4>)weights[port][base + lane];
                    expected += reference_product(weight_raw, act);
                }

                const int channel = lane * P_ACT_GROUPS + group;
                const int index = channel * DSP_ARRAY_COLS + col;
                const int output_word = index / OUTPUTS_PER_AXI_WORD;
                const int output_lane = index % OUTPUTS_PER_AXI_WORD;
                const ap_int<32> actual_value =
                    (ap_int<32>)output[output_word].range(
                        output_lane * 32 + 31, output_lane * 32);
                const int actual = (int)actual_value;
                if (actual != expected) {
                    std::cerr << "mismatch trial=" << trial
                              << " group=" << group
                              << " weight_lane=" << lane
                              << " col=" << col
                              << " actual=" << actual
                              << " expected=" << expected << '\n';
                    return 1;
                }
            }
        }
        }
    }

    std::cout << "PASS: 8 randomized 9-channel GEMV trials" << std::endl;
    return 0;
}
