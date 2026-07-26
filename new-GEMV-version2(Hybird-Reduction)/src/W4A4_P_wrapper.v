`timescale 1ns/1ps

// Decode one signed four-bit activation once per array row. Its magnitude is
// then broadcast across all 64 column PEs instead of rebuilding this logic in
// every PE. The -8 input retains the design's original clamp to magnitude 7.
(* keep_hierarchy = "yes" *)
module w4a4_activation_decode (
    input wire [3:0] value,
    output wire sign,
    output wire [2:0] magnitude
);

wire low_is_zero = value[2:0] == 3'b000;
wire [2:0] negative_magnitude = {
    (value[2] ^ (value[1] | value[0])) | low_is_zero,
    (value[1] ^ value[0]) | low_is_zero,
    value[0] | low_is_zero
};

assign sign = value[3];
assign magnitude = value[3] ? negative_magnitude : value[2:0];

endmodule

// Convert two independent sign-magnitude products into one pair sum. The
// explicit LUT/CARRY8 network preserves the compact carry-chain mapping
// without a fabric output register; the following DSP PREG is the next
// pipeline boundary.
(* keep_hierarchy = "yes", use_dsp = "no" *)
module w4a4_signed_magnitude_pair_add (
    input wire [5:0] magnitude0,
    input wire sign0,
    input wire [5:0] magnitude1,
    input wire sign1,
    output wire signed [7:0] value
);

wire [7:0] carry_di;
wire [7:0] carry_s;
wire [7:0] carry_co;

assign carry_di[0] = sign0;
assign carry_di[7] = 1'b0;

LUT6_2 #(.INIT(64'h5A5A5A5A4EE44EE4)) bit0_and_di1 (
    .I0(magnitude1[0]), .I1(sign1),
    .I2(magnitude0[0]), .I3(sign0),
    .I4(1'b0), .I5(1'b1),
    .O5(carry_di[1]), .O6(carry_s[0])
);

genvar pair_bit;
generate
    for (pair_bit = 1; pair_bit < 6;
         pair_bit = pair_bit + 1) begin : bit_gen
        LUT6_2 #(.INIT(64'h9669699606600660)) bit_and_next_di (
            .I0(magnitude1[pair_bit]), .I1(sign1),
            .I2(magnitude0[pair_bit]), .I3(sign0),
            .I4(carry_di[pair_bit]), .I5(1'b1),
            .O5(carry_di[pair_bit + 1]), .O6(carry_s[pair_bit])
        );
    end
endgenerate

LUT4 #(.INIT(16'h1BD8)) sum_bit6 (
    .I0(magnitude0[5]), .I1(magnitude1[5]),
    .I2(sign1), .I3(sign0), .O(carry_s[6])
);
LUT2 #(.INIT(4'hE)) sum_bit7 (
    .I0(sign1), .I1(sign0), .O(carry_s[7])
);

CARRY8 pair_carry (
    .CI(1'b0), .CI_TOP(1'b0),
    .DI(carry_di), .S(carry_s),
    .CO(carry_co), .O(value)
);

endmodule

// Handshaked shell used by the standalone RTL regression. ap_ready is the
// actual array acceptance event, and ap_continue stalls all registered stages.
module W4A4_P_chain #(
    parameter integer ARRAY_ROWS = 64,
    parameter integer ARRAY_COLS = 64,
    parameter integer ACT_GROUPS = 3,
    parameter integer WEIGHT_LANES = 3,
    parameter integer GROUP_COUNT = 4,
    parameter integer W_BITS = 4,
    parameter integer A_BITS = 4,
    parameter integer SUM_BITS = 16,
    parameter integer NUM_PE = ARRAY_ROWS * ARRAY_COLS,
    parameter integer PE_PER_GROUP = NUM_PE / GROUP_COUNT,
    parameter integer W_GROUP_BITS = PE_PER_GROUP * W_BITS,
    parameter integer A_BUS_BITS = ARRAY_ROWS * A_BITS,
    parameter integer CHANNEL_SUM_BITS = ARRAY_COLS * SUM_BITS
)(
    input wire ap_clk,
    input wire ap_rst,
    input wire ap_ce,
    input wire ap_start,
    input wire ap_continue,
    output wire ap_done,
    output wire ap_ready,
    output wire ap_idle,

    input wire [W_GROUP_BITS-1:0] w1g0,
    input wire [W_GROUP_BITS-1:0] w1g1,
    input wire [W_GROUP_BITS-1:0] w1g2,
    input wire [W_GROUP_BITS-1:0] w1g3,
    input wire [W_GROUP_BITS-1:0] w2g0,
    input wire [W_GROUP_BITS-1:0] w2g1,
    input wire [W_GROUP_BITS-1:0] w2g2,
    input wire [W_GROUP_BITS-1:0] w2g3,
    input wire [W_GROUP_BITS-1:0] w3g0,
    input wire [W_GROUP_BITS-1:0] w3g1,
    input wire [W_GROUP_BITS-1:0] w3g2,
    input wire [W_GROUP_BITS-1:0] w3g3,
    input wire [A_BUS_BITS-1:0] a1,
    input wire [A_BUS_BITS-1:0] a2,
    input wire [A_BUS_BITS-1:0] a3,

    output wire [CHANNEL_SUM_BITS-1:0] sum0,
    output wire sum0_ap_vld,
    output wire [CHANNEL_SUM_BITS-1:0] sum1,
    output wire sum1_ap_vld,
    output wire [CHANNEL_SUM_BITS-1:0] sum2,
    output wire sum2_ap_vld,
    output wire [CHANNEL_SUM_BITS-1:0] sum3,
    output wire sum3_ap_vld,
    output wire [CHANNEL_SUM_BITS-1:0] sum4,
    output wire sum4_ap_vld,
    output wire [CHANNEL_SUM_BITS-1:0] sum5,
    output wire sum5_ap_vld,
    output wire [CHANNEL_SUM_BITS-1:0] sum6,
    output wire sum6_ap_vld,
    output wire [CHANNEL_SUM_BITS-1:0] sum7,
    output wire sum7_ap_vld,
    output wire [CHANNEL_SUM_BITS-1:0] sum8,
    output wire sum8_ap_vld
);

wire input_accepted;
wire result_valid;
wire array_idle;
wire unused_write0, unused_write1, unused_write2;
wire unused_write3, unused_write4, unused_write5;
wire unused_write6, unused_write7, unused_write8;
wire unused_read1, unused_read2, unused_read3;
wire unused_read4, unused_read5, unused_read6, unused_read7;
wire unused_read8, unused_read9, unused_read10, unused_read11;
wire unused_read12, unused_read13, unused_read14;

W4A4_P_stream array (
    .ap_clk(ap_clk), .ap_rst(ap_rst), .ap_ce(ap_ce),
    .w1g0(w1g0), .w1g0_empty_n(ap_start), .w1g0_read(input_accepted),
    .w1g1(w1g1), .w1g1_empty_n(ap_start), .w1g1_read(unused_read1),
    .w1g2(w1g2), .w1g2_empty_n(ap_start), .w1g2_read(unused_read2),
    .w1g3(w1g3), .w1g3_empty_n(ap_start), .w1g3_read(unused_read3),
    .w2g0(w2g0), .w2g0_empty_n(ap_start), .w2g0_read(unused_read4),
    .w2g1(w2g1), .w2g1_empty_n(ap_start), .w2g1_read(unused_read5),
    .w2g2(w2g2), .w2g2_empty_n(ap_start), .w2g2_read(unused_read6),
    .w2g3(w2g3), .w2g3_empty_n(ap_start), .w2g3_read(unused_read7),
    .w3g0(w3g0), .w3g0_empty_n(ap_start), .w3g0_read(unused_read8),
    .w3g1(w3g1), .w3g1_empty_n(ap_start), .w3g1_read(unused_read9),
    .w3g2(w3g2), .w3g2_empty_n(ap_start), .w3g2_read(unused_read10),
    .w3g3(w3g3), .w3g3_empty_n(ap_start), .w3g3_read(unused_read11),
    .a1(a1), .a1_empty_n(ap_start), .a1_read(unused_read12),
    .a2(a2), .a2_empty_n(ap_start), .a2_read(unused_read13),
    .a3(a3), .a3_empty_n(ap_start), .a3_read(unused_read14),
    .sum0(sum0), .sum0_full_n(ap_continue), .sum0_write(unused_write0),
    .sum1(sum1), .sum1_full_n(ap_continue), .sum1_write(unused_write1),
    .sum2(sum2), .sum2_full_n(ap_continue), .sum2_write(unused_write2),
    .sum3(sum3), .sum3_full_n(ap_continue), .sum3_write(unused_write3),
    .sum4(sum4), .sum4_full_n(ap_continue), .sum4_write(unused_write4),
    .sum5(sum5), .sum5_full_n(ap_continue), .sum5_write(unused_write5),
    .sum6(sum6), .sum6_full_n(ap_continue), .sum6_write(unused_write6),
    .sum7(sum7), .sum7_full_n(ap_continue), .sum7_write(unused_write7),
    .sum8(sum8), .sum8_full_n(ap_continue), .sum8_write(unused_write8),
    .pipeline_result_valid(result_valid), .pipeline_idle(array_idle)
);

assign ap_ready = input_accepted;
assign ap_done = result_valid;
assign ap_idle = array_idle & ~ap_start;
assign sum0_ap_vld = result_valid;
assign sum1_ap_vld = result_valid;
assign sum2_ap_vld = result_valid;
assign sum3_ap_vld = result_valid;
assign sum4_ap_vld = result_valid;
assign sum5_ap_vld = result_valid;
assign sum6_ap_vld = result_valid;
assign sum7_ap_vld = result_valid;
assign sum8_ap_vld = result_valid;

endmodule

// Free-running shell used by the pipelined HLS blackbox call. Vitis HLS 2023.2
// does not allow an ap_ctrl_chain blackbox inside a pipeline region. With
// ap_ctrl_none it schedules this fixed-latency datapath directly: ap_ce is the
// only pipeline stall control, and a new complete GEMV enters on every enabled
// cycle. Packing all nine channels into one wire also avoids per-output valid
// control and keeps every one of the 64*9 reductions independent.
module W4A4_P #(
    parameter integer ARRAY_ROWS = 64,
    parameter integer ARRAY_COLS = 64,
    parameter integer ACT_GROUPS = 3,
    parameter integer WEIGHT_LANES = 3,
    parameter integer GROUP_COUNT = 4,
    parameter integer W_BITS = 4,
    parameter integer A_BITS = 4,
    parameter integer SUM_BITS = 16,
    parameter integer NUM_PE = ARRAY_ROWS * ARRAY_COLS,
    parameter integer PE_PER_GROUP = NUM_PE / GROUP_COUNT,
    parameter integer W_GROUP_BITS = PE_PER_GROUP * W_BITS,
    parameter integer A_BUS_BITS = ARRAY_ROWS * A_BITS,
    parameter integer CHANNELS = ACT_GROUPS * WEIGHT_LANES,
    parameter integer CHANNEL_SUM_BITS = ARRAY_COLS * SUM_BITS,
    parameter integer CORE_SUM_BITS = 13,
    parameter integer CORE_ALL_SUM_BITS = CHANNELS * ARRAY_COLS * CORE_SUM_BITS
)(
    input wire ap_clk,
    input wire ap_rst,
    input wire ap_ce,

    input wire [W_GROUP_BITS-1:0] w1g0,
    input wire [W_GROUP_BITS-1:0] w1g1,
    input wire [W_GROUP_BITS-1:0] w1g2,
    input wire [W_GROUP_BITS-1:0] w1g3,
    input wire [W_GROUP_BITS-1:0] w2g0,
    input wire [W_GROUP_BITS-1:0] w2g1,
    input wire [W_GROUP_BITS-1:0] w2g2,
    input wire [W_GROUP_BITS-1:0] w2g3,
    input wire [W_GROUP_BITS-1:0] w3g0,
    input wire [W_GROUP_BITS-1:0] w3g1,
    input wire [W_GROUP_BITS-1:0] w3g2,
    input wire [W_GROUP_BITS-1:0] w3g3,
    input wire [A_BUS_BITS-1:0] a1,
    input wire [A_BUS_BITS-1:0] a2,
    input wire [A_BUS_BITS-1:0] a3,

    output wire [CORE_ALL_SUM_BITS-1:0] ap_return
);

wire [CHANNEL_SUM_BITS-1:0] sum0;
wire [CHANNEL_SUM_BITS-1:0] sum1;
wire [CHANNEL_SUM_BITS-1:0] sum2;
wire [CHANNEL_SUM_BITS-1:0] sum3;
wire [CHANNEL_SUM_BITS-1:0] sum4;
wire [CHANNEL_SUM_BITS-1:0] sum5;
wire [CHANNEL_SUM_BITS-1:0] sum6;
wire [CHANNEL_SUM_BITS-1:0] sum7;
wire [CHANNEL_SUM_BITS-1:0] sum8;
wire [CHANNELS * CHANNEL_SUM_BITS-1:0] extended_sums =
    {sum8, sum7, sum6, sum5, sum4, sum3, sum2, sum1, sum0};
wire unused_result_valid;
wire unused_idle;
wire [14:0] unused_read;
wire [8:0] unused_write;

W4A4_P_stream array (
    .ap_clk(ap_clk), .ap_rst(ap_rst), .ap_ce(ap_ce),
    .w1g0(w1g0), .w1g0_empty_n(1'b1), .w1g0_read(unused_read[0]),
    .w1g1(w1g1), .w1g1_empty_n(1'b1), .w1g1_read(unused_read[1]),
    .w1g2(w1g2), .w1g2_empty_n(1'b1), .w1g2_read(unused_read[2]),
    .w1g3(w1g3), .w1g3_empty_n(1'b1), .w1g3_read(unused_read[3]),
    .w2g0(w2g0), .w2g0_empty_n(1'b1), .w2g0_read(unused_read[4]),
    .w2g1(w2g1), .w2g1_empty_n(1'b1), .w2g1_read(unused_read[5]),
    .w2g2(w2g2), .w2g2_empty_n(1'b1), .w2g2_read(unused_read[6]),
    .w2g3(w2g3), .w2g3_empty_n(1'b1), .w2g3_read(unused_read[7]),
    .w3g0(w3g0), .w3g0_empty_n(1'b1), .w3g0_read(unused_read[8]),
    .w3g1(w3g1), .w3g1_empty_n(1'b1), .w3g1_read(unused_read[9]),
    .w3g2(w3g2), .w3g2_empty_n(1'b1), .w3g2_read(unused_read[10]),
    .w3g3(w3g3), .w3g3_empty_n(1'b1), .w3g3_read(unused_read[11]),
    .a1(a1), .a1_empty_n(1'b1), .a1_read(unused_read[12]),
    .a2(a2), .a2_empty_n(1'b1), .a2_read(unused_read[13]),
    .a3(a3), .a3_empty_n(1'b1), .a3_read(unused_read[14]),
    .sum0(sum0), .sum0_full_n(1'b1), .sum0_write(unused_write[0]),
    .sum1(sum1), .sum1_full_n(1'b1), .sum1_write(unused_write[1]),
    .sum2(sum2), .sum2_full_n(1'b1), .sum2_write(unused_write[2]),
    .sum3(sum3), .sum3_full_n(1'b1), .sum3_write(unused_write[3]),
    .sum4(sum4), .sum4_full_n(1'b1), .sum4_write(unused_write[4]),
    .sum5(sum5), .sum5_full_n(1'b1), .sum5_write(unused_write[5]),
    .sum6(sum6), .sum6_full_n(1'b1), .sum6_write(unused_write[6]),
    .sum7(sum7), .sum7_full_n(1'b1), .sum7_write(unused_write[7]),
    .sum8(sum8), .sum8_full_n(1'b1), .sum8_write(unused_write[8]),
    .pipeline_result_valid(unused_result_valid), .pipeline_idle(unused_idle)
);

genvar core_output_lane;
generate
    for (core_output_lane = 0;
         core_output_lane < CHANNELS * ARRAY_COLS;
         core_output_lane = core_output_lane + 1) begin : core_output_pack_gen
        assign ap_return[core_output_lane * CORE_SUM_BITS +: CORE_SUM_BITS] =
            extended_sums[core_output_lane * SUM_BITS +: CORE_SUM_BITS];
    end
endgenerate

endmodule

// Registered fabric adder used by the LUT-heavy tree levels. Every instance is
// a separate pipeline lane, so moving a level into fabric does not reduce the
// complete GEMV array's one-input-per-cycle throughput.
(* keep_hierarchy = "yes", use_dsp = "no" *)
module w4a4_registered_fabric_add #(
    parameter integer INPUT_WIDTH = 11
)(
    input wire clk,
    input wire ce,
    input wire [INPUT_WIDTH-1:0] lhs,
    input wire [INPUT_WIDTH-1:0] rhs,
    output reg [INPUT_WIDTH:0] value
);

wire signed [INPUT_WIDTH:0] lhs_extended = {lhs[INPUT_WIDTH-1], lhs};
wire signed [INPUT_WIDTH:0] rhs_extended = {rhs[INPUT_WIDTH-1], rhs};
wire signed [INPUT_WIDTH:0] fabric_sum = lhs_extended + rhs_extended;

always @(posedge clk) begin
    if (ce)
        value <= fabric_sum;
end

endmodule

// Four independent signed additions share one DSP48E2 in FOUR12 SIMD mode.
// Inputs are sign-extended to the fixed 12-bit lane boundaries; the SIMD carry
// barriers prevent a negative value or an overflow from affecting its neighbor.
module w4a4_packed_dsp_add #(
    parameter integer INPUT_WIDTH = 8,
    parameter integer LANES = 4
)(
    input wire clk,
    input wire ce,
    input wire [LANES * INPUT_WIDTH - 1:0] lhs,
    input wire [LANES * INPUT_WIDTH - 1:0] rhs,
    output wire [LANES * (INPUT_WIDTH + 1) - 1:0] value
);

wire [47:0] packed_lhs;
wire [47:0] packed_rhs;
wire [47:0] dsp_p;

genvar packed_lane;
generate
    for (packed_lane = 0; packed_lane < LANES;
         packed_lane = packed_lane + 1) begin : pack_gen
        localparam integer INPUT_BASE = packed_lane * INPUT_WIDTH;
        localparam integer PACKED_BASE = packed_lane * 12;
        localparam integer OUTPUT_BASE = packed_lane * (INPUT_WIDTH + 1);

        assign packed_lhs[PACKED_BASE +: 12] =
            {{(12 - INPUT_WIDTH){lhs[INPUT_BASE + INPUT_WIDTH - 1]}},
             lhs[INPUT_BASE +: INPUT_WIDTH]};
        assign packed_rhs[PACKED_BASE +: 12] =
            {{(12 - INPUT_WIDTH){rhs[INPUT_BASE + INPUT_WIDTH - 1]}},
             rhs[INPUT_BASE +: INPUT_WIDTH]};
        assign value[OUTPUT_BASE +: INPUT_WIDTH + 1] =
            dsp_p[PACKED_BASE +: INPUT_WIDTH + 1];
    end
endgenerate

// UG579 tables 2-3 through 2-6: OPMODE 000110011 selects
// W=0, X=A:B, Y=0, Z=C. With ALUMODE=0000 this computes A:B + C.
DSP48E2 #(
    .AMULTSEL("A"), .A_INPUT("DIRECT"),
    .BMULTSEL("B"), .B_INPUT("DIRECT"), .PREADDINSEL("A"),
    .RND(48'h000000000000),
    .USE_MULT("NONE"), .USE_SIMD("FOUR12"),
    .USE_WIDEXOR("FALSE"), .XORSIMD("XOR12"),
    .AUTORESET_PATDET("NO_RESET"), .AUTORESET_PRIORITY("RESET"),
    .MASK(48'h3fffffffffff), .PATTERN(48'h000000000000),
    .SEL_MASK("MASK"), .SEL_PATTERN("PATTERN"),
    .USE_PATTERN_DETECT("NO_PATDET"),
    .IS_ALUMODE_INVERTED(4'b0000), .IS_CARRYIN_INVERTED(1'b0),
    .IS_CLK_INVERTED(1'b0), .IS_INMODE_INVERTED(5'b00000),
    .IS_OPMODE_INVERTED(9'b000000000),
    .IS_RSTALLCARRYIN_INVERTED(1'b0),
    .IS_RSTALUMODE_INVERTED(1'b0), .IS_RSTA_INVERTED(1'b0),
    .IS_RSTB_INVERTED(1'b0), .IS_RSTCTRL_INVERTED(1'b0),
    .IS_RSTC_INVERTED(1'b0), .IS_RSTD_INVERTED(1'b0),
    .IS_RSTINMODE_INVERTED(1'b0), .IS_RSTM_INVERTED(1'b0),
    .IS_RSTP_INVERTED(1'b0),
    .ACASCREG(0), .ADREG(0), .ALUMODEREG(0), .AREG(0),
    .BCASCREG(0), .BREG(0), .CARRYINREG(0),
    .CARRYINSELREG(0), .CREG(0), .DREG(0), .INMODEREG(0),
    .MREG(0), .OPMODEREG(0), .PREG(1)
) simd_add (
    .ACOUT(), .BCOUT(), .CARRYCASCOUT(), .MULTSIGNOUT(), .PCOUT(),
    .OVERFLOW(), .PATTERNBDETECT(), .PATTERNDETECT(), .UNDERFLOW(),
    .CARRYOUT(), .P(dsp_p), .XOROUT(),
    .ACIN(30'b0), .BCIN(18'b0), .CARRYCASCIN(1'b0),
    .MULTSIGNIN(1'b0), .PCIN(48'b0),
    .ALUMODE(4'b0000), .CARRYINSEL(3'b000), .CLK(clk),
    .INMODE(5'b00000), .OPMODE(9'b000110011),
    .A(packed_lhs[47:18]), .B(packed_lhs[17:0]),
    .C(packed_rhs), .CARRYIN(1'b0), .D(27'b0),
    .CEA1(1'b0), .CEA2(1'b0), .CEAD(1'b0), .CEALUMODE(1'b0),
    .CEB1(1'b0), .CEB2(1'b0), .CEC(1'b0), .CECARRYIN(1'b0),
    .CECTRL(1'b0), .CED(1'b0), .CEINMODE(1'b0), .CEM(1'b0),
    .CEP(ce),
    .RSTA(1'b0), .RSTALLCARRYIN(1'b0), .RSTALUMODE(1'b0),
    .RSTB(1'b0), .RSTC(1'b0), .RSTCTRL(1'b0), .RSTD(1'b0),
    .RSTINMODE(1'b0), .RSTM(1'b0), .RSTP(1'b0)
);

endmodule

// One DSP48E2 ALU stage for a packed cascade. Every production instance uses
// PREG=1. A/B/C input registers align later operands with the registered PCIN
// running sum while keeping those delay registers inside DSP48E2.
module w4a4_dsp_cascade_add #(
    parameter USE_SIMD = "FOUR12",
    parameter integer AREG_DEPTH = 0,
    parameter integer BREG_DEPTH = 0,
    parameter integer CREG_DEPTH = 0,
    parameter integer PREG_DEPTH = 1,
    parameter integer USE_PCIN = 0
)(
    input wire clk,
    input wire ce,
    input wire [47:0] ab_operand,
    input wire [47:0] c_operand,
    input wire [47:0] pcin,
    output wire [47:0] value,
    output wire [47:0] pcout
);

DSP48E2 #(
    .AMULTSEL("A"), .A_INPUT("DIRECT"),
    .BMULTSEL("B"), .B_INPUT("DIRECT"), .PREADDINSEL("A"),
    .RND(48'h000000000000),
    .USE_MULT("NONE"), .USE_SIMD(USE_SIMD),
    .USE_WIDEXOR("FALSE"), .XORSIMD("XOR12"),
    .AUTORESET_PATDET("NO_RESET"), .AUTORESET_PRIORITY("RESET"),
    .MASK(48'h3fffffffffff), .PATTERN(48'h000000000000),
    .SEL_MASK("MASK"), .SEL_PATTERN("PATTERN"),
    .USE_PATTERN_DETECT("NO_PATDET"),
    .IS_ALUMODE_INVERTED(4'b0000),
    .IS_CARRYIN_INVERTED(1'b0), .IS_CLK_INVERTED(1'b0),
    .IS_INMODE_INVERTED(5'b00000),
    .IS_OPMODE_INVERTED(9'b000000000),
    .IS_RSTALLCARRYIN_INVERTED(1'b0),
    .IS_RSTALUMODE_INVERTED(1'b0), .IS_RSTA_INVERTED(1'b0),
    .IS_RSTB_INVERTED(1'b0), .IS_RSTCTRL_INVERTED(1'b0),
    .IS_RSTC_INVERTED(1'b0), .IS_RSTD_INVERTED(1'b0),
    .IS_RSTINMODE_INVERTED(1'b0), .IS_RSTM_INVERTED(1'b0),
    .IS_RSTP_INVERTED(1'b0),
    .ACASCREG(AREG_DEPTH), .ADREG(0), .ALUMODEREG(0),
    .AREG(AREG_DEPTH),
    .BCASCREG(BREG_DEPTH), .BREG(BREG_DEPTH), .CARRYINREG(0),
    .CARRYINSELREG(0), .CREG(CREG_DEPTH), .DREG(0),
    .INMODEREG(0), .MREG(0), .OPMODEREG(0), .PREG(PREG_DEPTH)
) add_dsp (
    .ACOUT(), .BCOUT(), .CARRYCASCOUT(), .MULTSIGNOUT(), .PCOUT(pcout),
    .OVERFLOW(), .PATTERNBDETECT(), .PATTERNDETECT(), .UNDERFLOW(),
    .CARRYOUT(), .P(value), .XOROUT(),
    .ACIN(30'b0), .BCIN(18'b0), .CARRYCASCIN(1'b0),
    .MULTSIGNIN(1'b0), .PCIN(pcin),
    .ALUMODE(4'b0000), .CARRYINSEL(3'b000), .CLK(clk),
    .INMODE(5'b00000),
    .OPMODE(USE_PCIN ? 9'b110010011 : 9'b000110011),
    .A(ab_operand[47:18]), .B(ab_operand[17:0]), .C(c_operand),
    .CARRYIN(1'b0), .D(27'b0),
    .CEA1((AREG_DEPTH == 2) ? ce : 1'b0),
    .CEA2((AREG_DEPTH > 0) ? ce : 1'b0), .CEAD(1'b0),
    .CEALUMODE(1'b0),
    .CEB1((BREG_DEPTH == 2) ? ce : 1'b0),
    .CEB2((BREG_DEPTH > 0) ? ce : 1'b0),
    .CEC((CREG_DEPTH > 0) ? ce : 1'b0), .CECARRYIN(1'b0),
    .CECTRL(1'b0), .CED(1'b0), .CEINMODE(1'b0), .CEM(1'b0),
    .CEP((PREG_DEPTH > 0) ? ce : 1'b0),
    .RSTA(1'b0), .RSTALLCARRYIN(1'b0), .RSTALUMODE(1'b0),
    .RSTB(1'b0), .RSTC(1'b0), .RSTCTRL(1'b0), .RSTD(1'b0),
    .RSTINMODE(1'b0), .RSTM(1'b0), .RSTP(1'b0)
);

endmodule

// Four independent trees each contribute eight signed pair sums. Every DSP in
// this FOUR12 PCIN cascade is registered, giving latency four and II=1.
module w4a4_overlap_chain8_4x12 (
    input wire clk,
    input wire ce,
    input wire [4 * 8 * 8 - 1:0] inputs,
    output wire [4 * 12 - 1:0] sums
);

wire [47:0] term0;
wire [47:0] term1;
wire [47:0] term2;
wire [47:0] term3;
wire [47:0] term4;
wire [47:0] term5;
wire [47:0] term6;
wire [47:0] term7;
reg [47:0] term5_delay;
reg [47:0] term6_delay;
reg [47:0] term7_delay0;
reg [47:0] term7_delay1;
wire [47:0] stage0_pcout;
wire [47:0] stage1_pcout;
wire [47:0] stage2_pcout;
wire [47:0] stage3_value;

genvar overlap_lane;
generate
    for (overlap_lane = 0; overlap_lane < 4;
         overlap_lane = overlap_lane + 1) begin : pack_lane_gen
        assign term0[overlap_lane * 12 +: 12] =
            {{4{inputs[(overlap_lane * 8 + 0) * 8 + 7]}},
             inputs[(overlap_lane * 8 + 0) * 8 +: 8]};
        assign term1[overlap_lane * 12 +: 12] =
            {{4{inputs[(overlap_lane * 8 + 1) * 8 + 7]}},
             inputs[(overlap_lane * 8 + 1) * 8 +: 8]};
        assign term2[overlap_lane * 12 +: 12] =
            {{4{inputs[(overlap_lane * 8 + 2) * 8 + 7]}},
             inputs[(overlap_lane * 8 + 2) * 8 +: 8]};
        assign term3[overlap_lane * 12 +: 12] =
            {{4{inputs[(overlap_lane * 8 + 3) * 8 + 7]}},
             inputs[(overlap_lane * 8 + 3) * 8 +: 8]};
        assign term4[overlap_lane * 12 +: 12] =
            {{4{inputs[(overlap_lane * 8 + 4) * 8 + 7]}},
             inputs[(overlap_lane * 8 + 4) * 8 +: 8]};
        assign term5[overlap_lane * 12 +: 12] =
            {{4{inputs[(overlap_lane * 8 + 5) * 8 + 7]}},
             inputs[(overlap_lane * 8 + 5) * 8 +: 8]};
        assign term6[overlap_lane * 12 +: 12] =
            {{4{inputs[(overlap_lane * 8 + 6) * 8 + 7]}},
             inputs[(overlap_lane * 8 + 6) * 8 +: 8]};
        assign term7[overlap_lane * 12 +: 12] =
            {{4{inputs[(overlap_lane * 8 + 7) * 8 + 7]}},
             inputs[(overlap_lane * 8 + 7) * 8 +: 8]};
    end
endgenerate

always @(posedge clk) begin
    if (ce) begin
        term5_delay <= term5;
        term6_delay <= term6;
        term7_delay0 <= term7;
        term7_delay1 <= term7_delay0;
    end
end

w4a4_dsp_cascade_add #(
    .USE_SIMD("FOUR12"), .AREG_DEPTH(0), .BREG_DEPTH(0),
    .CREG_DEPTH(0), .PREG_DEPTH(1), .USE_PCIN(0)
) stage0 (
    .clk(clk), .ce(ce), .ab_operand(term0), .c_operand(term1),
    .pcin(48'b0), .value(), .pcout(stage0_pcout)
);

w4a4_dsp_cascade_add #(
    .USE_SIMD("FOUR12"), .AREG_DEPTH(1), .BREG_DEPTH(1),
    .CREG_DEPTH(1), .PREG_DEPTH(1), .USE_PCIN(1)
) stage1 (
    .clk(clk), .ce(ce), .ab_operand(term2), .c_operand(term3),
    .pcin(stage0_pcout), .value(), .pcout(stage1_pcout)
);

w4a4_dsp_cascade_add #(
    .USE_SIMD("FOUR12"), .AREG_DEPTH(2), .BREG_DEPTH(2),
    .CREG_DEPTH(1), .PREG_DEPTH(1), .USE_PCIN(1)
) stage2 (
    .clk(clk), .ce(ce), .ab_operand(term4), .c_operand(term5_delay),
    .pcin(stage1_pcout), .value(), .pcout(stage2_pcout)
);

w4a4_dsp_cascade_add #(
    .USE_SIMD("FOUR12"), .AREG_DEPTH(2), .BREG_DEPTH(2),
    .CREG_DEPTH(1), .PREG_DEPTH(1), .USE_PCIN(1)
) stage3 (
    .clk(clk), .ce(ce), .ab_operand(term6_delay),
    .c_operand(term7_delay1),
    .pcin(stage2_pcout), .value(stage3_value), .pcout()
);

assign sums = stage3_value;

endmodule

// Four independent trees each contribute four 12-bit partial sums. A balanced
// two-level LUT/CARRY tree keeps the same two-cycle latency and II=1 without
// spending TWO24 DSP lanes on 13-bit results.
(* keep_hierarchy = "yes", use_dsp = "no" *)
module w4a4_lut_reduce4x12 (
    input wire clk,
    input wire ce,
    input wire [4 * 4 * 12 - 1:0] inputs,
    output wire [4 * 13 - 1:0] sums
);

reg [4 * 13 - 1:0] pair_sums0;
reg [4 * 13 - 1:0] pair_sums1;
reg [4 * 13 - 1:0] sums_r;

genvar lut_tree;
generate
    for (lut_tree = 0; lut_tree < 4;
         lut_tree = lut_tree + 1) begin : tree_gen
        wire signed [12:0] input0 = {
            inputs[(lut_tree * 4 + 0) * 12 + 11],
            inputs[(lut_tree * 4 + 0) * 12 +: 12]};
        wire signed [12:0] input1 = {
            inputs[(lut_tree * 4 + 1) * 12 + 11],
            inputs[(lut_tree * 4 + 1) * 12 +: 12]};
        wire signed [12:0] input2 = {
            inputs[(lut_tree * 4 + 2) * 12 + 11],
            inputs[(lut_tree * 4 + 2) * 12 +: 12]};
        wire signed [12:0] input3 = {
            inputs[(lut_tree * 4 + 3) * 12 + 11],
            inputs[(lut_tree * 4 + 3) * 12 +: 12]};
        wire signed [12:0] pair_sum0 = input0 + input1;
        wire signed [12:0] pair_sum1 = input2 + input3;
        wire signed [13:0] final_sum =
            $signed(pair_sums0[lut_tree * 13 +: 13]) +
            $signed(pair_sums1[lut_tree * 13 +: 13]);

        always @(posedge clk) begin
            if (ce) begin
                pair_sums0[lut_tree * 13 +: 13] <= pair_sum0;
                pair_sums1[lut_tree * 13 +: 13] <= pair_sum1;
                sums_r[lut_tree * 13 +: 13] <= final_sum[12:0];
            end
        end
    end
endgenerate

assign sums = sums_r;

endmodule

// Four complete 64-row trees share four registered FOUR12 cascades followed by
// a two-level registered LUT tree. The explicit pair-add logic is folded in
// front of the first DSP stage, so reduction latency is six cycles at II=1.
(* keep_hierarchy = "yes" *)
module w4a4_overlap_tree4 (
    input wire clk,
    input wire ce,
    input wire [4 * 32 * 8 - 1:0] pair_values,
    output wire [4 * 13 - 1:0] sums
);

wire [4 * 4 * 8 * 8 - 1:0] group_inputs;
wire [4 * 4 * 12 - 1:0] group_sums;
wire [4 * 4 * 12 - 1:0] partial_sums;

genvar first_group;
genvar first_tree;
genvar first_term;
generate
    for (first_group = 0; first_group < 4;
         first_group = first_group + 1) begin : first_group_gen
        for (first_tree = 0; first_tree < 4;
             first_tree = first_tree + 1) begin : first_tree_gen
            for (first_term = 0; first_term < 8;
                 first_term = first_term + 1) begin : first_term_gen
                assign group_inputs[
                    ((first_group * 4 + first_tree) * 8 + first_term) * 8 +:
                    8] = pair_values[
                        (first_tree * 32 + first_group * 8 + first_term) * 8 +:
                        8];
            end
            assign partial_sums[(first_tree * 4 + first_group) * 12 +: 12] =
                group_sums[(first_group * 4 + first_tree) * 12 +: 12];
        end
        w4a4_overlap_chain8_4x12 chain (
            .clk(clk), .ce(ce),
            .inputs(group_inputs[first_group * 4 * 8 * 8 +: 4 * 8 * 8]),
            .sums(group_sums[first_group * 4 * 12 +: 4 * 12])
        );
    end
endgenerate

w4a4_lut_reduce4x12 final_lut_tree (
    .clk(clk), .ce(ce), .inputs(partial_sums), .sums(sums)
);

endmodule

module W4A4_P_stream #(
    parameter integer ARRAY_ROWS = 64,
    parameter integer ARRAY_COLS = 64,
    parameter integer ACT_GROUPS = 3,
    parameter integer WEIGHT_LANES = 3,
    parameter integer CHANNELS = ACT_GROUPS * WEIGHT_LANES,
    parameter integer GROUP_COUNT = 4,
    parameter integer W_BITS = 4,
    parameter integer A_BITS = 4,
    parameter integer MAG_BITS = 6,
    parameter integer SUM_BITS = 16,
    parameter integer NUM_PE = ARRAY_ROWS * ARRAY_COLS,
    parameter integer PE_PER_GROUP = NUM_PE / GROUP_COUNT,
    parameter integer W_BUS_BITS = NUM_PE * W_BITS,
    parameter integer W_GROUP_BITS = PE_PER_GROUP * W_BITS,
    parameter integer A_BUS_BITS = ARRAY_ROWS * A_BITS,
    parameter integer OUTPUT_LANES = CHANNELS * ARRAY_COLS,
    parameter integer CHANNEL_SUM_BITS = ARRAY_COLS * SUM_BITS,
    parameter integer SUM_BUS_BITS = OUTPUT_LANES * SUM_BITS
)(
    input wire ap_clk,
    input wire ap_rst,
    input wire ap_ce,

    input wire [W_GROUP_BITS-1:0] w1g0,
    input wire w1g0_empty_n,
    output wire w1g0_read,
    input wire [W_GROUP_BITS-1:0] w1g1,
    input wire w1g1_empty_n,
    output wire w1g1_read,
    input wire [W_GROUP_BITS-1:0] w1g2,
    input wire w1g2_empty_n,
    output wire w1g2_read,
    input wire [W_GROUP_BITS-1:0] w1g3,
    input wire w1g3_empty_n,
    output wire w1g3_read,
    input wire [W_GROUP_BITS-1:0] w2g0,
    input wire w2g0_empty_n,
    output wire w2g0_read,
    input wire [W_GROUP_BITS-1:0] w2g1,
    input wire w2g1_empty_n,
    output wire w2g1_read,
    input wire [W_GROUP_BITS-1:0] w2g2,
    input wire w2g2_empty_n,
    output wire w2g2_read,
    input wire [W_GROUP_BITS-1:0] w2g3,
    input wire w2g3_empty_n,
    output wire w2g3_read,
    input wire [W_GROUP_BITS-1:0] w3g0,
    input wire w3g0_empty_n,
    output wire w3g0_read,
    input wire [W_GROUP_BITS-1:0] w3g1,
    input wire w3g1_empty_n,
    output wire w3g1_read,
    input wire [W_GROUP_BITS-1:0] w3g2,
    input wire w3g2_empty_n,
    output wire w3g2_read,
    input wire [W_GROUP_BITS-1:0] w3g3,
    input wire w3g3_empty_n,
    output wire w3g3_read,
    input wire [A_BUS_BITS-1:0] a1,
    input wire a1_empty_n,
    output wire a1_read,
    input wire [A_BUS_BITS-1:0] a2,
    input wire a2_empty_n,
    output wire a2_read,
    input wire [A_BUS_BITS-1:0] a3,
    input wire a3_empty_n,
    output wire a3_read,
    output wire [CHANNEL_SUM_BITS-1:0] sum0,
    input wire sum0_full_n,
    output wire sum0_write,
    output wire [CHANNEL_SUM_BITS-1:0] sum1,
    input wire sum1_full_n,
    output wire sum1_write,
    output wire [CHANNEL_SUM_BITS-1:0] sum2,
    input wire sum2_full_n,
    output wire sum2_write,
    output wire [CHANNEL_SUM_BITS-1:0] sum3,
    input wire sum3_full_n,
    output wire sum3_write,
    output wire [CHANNEL_SUM_BITS-1:0] sum4,
    input wire sum4_full_n,
    output wire sum4_write,
    output wire [CHANNEL_SUM_BITS-1:0] sum5,
    input wire sum5_full_n,
    output wire sum5_write,
    output wire [CHANNEL_SUM_BITS-1:0] sum6,
    input wire sum6_full_n,
    output wire sum6_write,
    output wire [CHANNEL_SUM_BITS-1:0] sum7,
    input wire sum7_full_n,
    output wire sum7_write,
    output wire [CHANNEL_SUM_BITS-1:0] sum8,
    input wire sum8_full_n,
    output wire sum8_write,
    output wire pipeline_result_valid,
    output wire pipeline_idle
);

localparam integer WEIGHT_SIGN_BITS = WEIGHT_LANES * NUM_PE;
localparam integer ACT_SIGN_BITS = ACT_GROUPS * ARRAY_ROWS;
localparam integer ACT_MAG_BITS = ACT_GROUPS * ARRAY_ROWS * 3;
localparam integer SLR_GROUPS = 3;
localparam integer LEVEL32_NODES = CHANNELS * ARRAY_COLS * (ARRAY_ROWS / 2);
localparam integer CHANNEL_SUMS = CHANNELS * ARRAY_COLS;
localparam integer PAIRS_PER_TREE = ARRAY_ROWS / 2;
localparam integer PIPELINE_LATENCY = 8;

reg [PIPELINE_LATENCY-1:0] valid_pipe;
wire all_inputs_valid =
    w1g0_empty_n & w1g1_empty_n & w1g2_empty_n & w1g3_empty_n &
    w2g0_empty_n & w2g1_empty_n & w2g2_empty_n & w2g3_empty_n &
    w3g0_empty_n & w3g1_empty_n & w3g2_empty_n & w3g3_empty_n &
    a1_empty_n & a2_empty_n & a3_empty_n;
wire all_outputs_ready =
    sum0_full_n & sum1_full_n & sum2_full_n & sum3_full_n &
    sum4_full_n & sum5_full_n & sum6_full_n & sum7_full_n &
    sum8_full_n;
wire output_blocked =
    valid_pipe[PIPELINE_LATENCY-1] & ~all_outputs_ready;
wire pipeline_ce = ap_ce & ~output_blocked;
wire accept_input = pipeline_ce & all_inputs_valid;

assign w1g0_read = accept_input;
assign w1g1_read = accept_input;
assign w1g2_read = accept_input;
assign w1g3_read = accept_input;
assign w2g0_read = accept_input;
assign w2g1_read = accept_input;
assign w2g2_read = accept_input;
assign w2g3_read = accept_input;
assign w3g0_read = accept_input;
assign w3g1_read = accept_input;
assign w3g2_read = accept_input;
assign w3g3_read = accept_input;
assign a1_read = accept_input;
assign a2_read = accept_input;
assign a3_read = accept_input;

wire [W_BUS_BITS-1:0] weights1 = {w1g3, w1g2, w1g1, w1g0};
wire [W_BUS_BITS-1:0] weights2 = {w2g3, w2g2, w2g1, w2g0};
wire [W_BUS_BITS-1:0] weights3 = {w3g3, w3g2, w3g1, w3g0};
wire [CHANNELS * NUM_PE * MAG_BITS - 1:0] magnitudes;

wire [WEIGHT_SIGN_BITS-1:0] weight_signs;
wire [SLR_GROUPS * ACT_SIGN_BITS-1:0] activation_signs;
wire [SLR_GROUPS * ACT_MAG_BITS-1:0] activation_magnitudes;
reg [WEIGHT_SIGN_BITS-1:0] weight_signs_r1;
reg [WEIGHT_SIGN_BITS-1:0] weight_signs_r2;
reg [SLR_GROUPS * ACT_SIGN_BITS-1:0] activation_signs_r1;
reg [SLR_GROUPS * ACT_SIGN_BITS-1:0] activation_signs_r2;

genvar sign_pe;
generate
    for (sign_pe = 0; sign_pe < NUM_PE; sign_pe = sign_pe + 1) begin : weight_sign_gen
        assign weight_signs[0 * NUM_PE + sign_pe] = weights1[sign_pe * W_BITS + 3];
        assign weight_signs[1 * NUM_PE + sign_pe] = weights2[sign_pe * W_BITS + 3];
        assign weight_signs[2 * NUM_PE + sign_pe] = weights3[sign_pe * W_BITS + 3];
    end
endgenerate

// Decode one activation copy per SLR instead of driving all 64 columns from a
// single decoder. DONT_TOUCH prevents synthesis from merging the three copies
// back into the high-fanout cross-SLR network this hierarchy is meant to avoid.
genvar act_slr_group;
genvar sign_row;
generate
    for (act_slr_group = 0; act_slr_group < SLR_GROUPS;
         act_slr_group = act_slr_group + 1) begin : act_slr_gen
        for (sign_row = 0; sign_row < ARRAY_ROWS;
             sign_row = sign_row + 1) begin : act_row_gen
            (* DONT_TOUCH = "TRUE" *) w4a4_activation_decode decode_a1 (
                .value(a1[sign_row * A_BITS +: A_BITS]),
                .sign(activation_signs[
                    act_slr_group * ACT_SIGN_BITS +
                    0 * ARRAY_ROWS + sign_row]),
                .magnitude(activation_magnitudes[
                    act_slr_group * ACT_MAG_BITS +
                    (0 * ARRAY_ROWS + sign_row) * 3 +: 3])
            );
            (* DONT_TOUCH = "TRUE" *) w4a4_activation_decode decode_a2 (
                .value(a2[sign_row * A_BITS +: A_BITS]),
                .sign(activation_signs[
                    act_slr_group * ACT_SIGN_BITS +
                    1 * ARRAY_ROWS + sign_row]),
                .magnitude(activation_magnitudes[
                    act_slr_group * ACT_MAG_BITS +
                    (1 * ARRAY_ROWS + sign_row) * 3 +: 3])
            );
            (* DONT_TOUCH = "TRUE" *) w4a4_activation_decode decode_a3 (
                .value(a3[sign_row * A_BITS +: A_BITS]),
                .sign(activation_signs[
                    act_slr_group * ACT_SIGN_BITS +
                    2 * ARRAY_ROWS + sign_row]),
                .magnitude(activation_magnitudes[
                    act_slr_group * ACT_MAG_BITS +
                    (2 * ARRAY_ROWS + sign_row) * 3 +: 3])
            );
        end
    end
endgenerate

always @(posedge ap_clk) begin
    if (pipeline_ce) begin
        weight_signs_r1 <= weight_signs;
        weight_signs_r2 <= weight_signs_r1;
        activation_signs_r1 <= activation_signs;
        activation_signs_r2 <= activation_signs_r1;
    end
end

// One DSP per (column,row) retains the original 3W x 3A overpacking: all nine
// product magnitudes remain independent through the complete 64-row reduction.
genvar core_col;
genvar core_row;
generate
    for (core_col = 0; core_col < ARRAY_COLS; core_col = core_col + 1) begin : core_col_gen
        for (core_row = 0; core_row < ARRAY_ROWS; core_row = core_row + 1) begin : core_row_gen
            localparam integer PE_INDEX = core_col * ARRAY_ROWS + core_row;
            localparam integer W_OFFSET = PE_INDEX * W_BITS;
            localparam integer A_OFFSET = core_row * A_BITS;
            localparam integer CORE_SLR =
                (core_col < 22) ? 0 : ((core_col < 42) ? 1 : 2);

            INT4_INT4_P_MAG dsp_core (
                .clk(ap_clk),
                .ce(pipeline_ce),
                .w1_magnitude(weights1[W_OFFSET +: 3]),
                .w2_magnitude(weights2[W_OFFSET +: 3]),
                .w3_magnitude(weights3[W_OFFSET +: 3]),
                .a1_magnitude(activation_magnitudes[
                    CORE_SLR * ACT_MAG_BITS +
                    (0 * ARRAY_ROWS + core_row) * 3 +: 3]),
                .a2_magnitude(activation_magnitudes[
                    CORE_SLR * ACT_MAG_BITS +
                    (1 * ARRAY_ROWS + core_row) * 3 +: 3]),
                .a3_magnitude(activation_magnitudes[
                    CORE_SLR * ACT_MAG_BITS +
                    (2 * ARRAY_ROWS + core_row) * 3 +: 3]),
                .magnitude1(magnitudes[(0 * NUM_PE + PE_INDEX) * MAG_BITS +: MAG_BITS]),
                .magnitude2(magnitudes[(1 * NUM_PE + PE_INDEX) * MAG_BITS +: MAG_BITS]),
                .magnitude3(magnitudes[(2 * NUM_PE + PE_INDEX) * MAG_BITS +: MAG_BITS]),
                .magnitude4(magnitudes[(3 * NUM_PE + PE_INDEX) * MAG_BITS +: MAG_BITS]),
                .magnitude5(magnitudes[(4 * NUM_PE + PE_INDEX) * MAG_BITS +: MAG_BITS]),
                .magnitude6(magnitudes[(5 * NUM_PE + PE_INDEX) * MAG_BITS +: MAG_BITS]),
                .magnitude7(magnitudes[(6 * NUM_PE + PE_INDEX) * MAG_BITS +: MAG_BITS]),
                .magnitude8(magnitudes[(7 * NUM_PE + PE_INDEX) * MAG_BITS +: MAG_BITS]),
                .magnitude9(magnitudes[(8 * NUM_PE + PE_INDEX) * MAG_BITS +: MAG_BITS])
            );
        end
    end
endgenerate

wire [LEVEL32_NODES * 8 - 1:0] level32;
wire [CHANNEL_SUMS * 13 - 1:0] channel_sums;
wire [SUM_BUS_BITS-1:0] packed_sums;

// The first row-tree level performs sign-magnitude conversion with explicit
// LUT/CARRY8 pair adders. Its output is registered by the following DSP stage.
// There are nine separate trees per column; no weight channel is merged here.
genvar level32_node;
generate
    for (level32_node = 0; level32_node < LEVEL32_NODES;
         level32_node = level32_node + 1) begin : level32_gen
        localparam integer CHANNEL_COL_INDEX = level32_node / (ARRAY_ROWS / 2);
        localparam integer ROW_PAIR = level32_node % (ARRAY_ROWS / 2);
        localparam integer CHANNEL = CHANNEL_COL_INDEX / ARRAY_COLS;
        localparam integer COL = CHANNEL_COL_INDEX % ARRAY_COLS;
        localparam integer WEIGHT_LANE = CHANNEL / ACT_GROUPS;
        localparam integer ACT_GROUP = CHANNEL % ACT_GROUPS;
        localparam integer TREE_SLR =
            (COL < 22) ? 0 : ((COL < 42) ? 1 : 2);
        localparam integer PE0 = COL * ARRAY_ROWS + 2 * ROW_PAIR;
        localparam integer PE1 = PE0 + 1;
        localparam integer ROW0 = 2 * ROW_PAIR;
        localparam integer ROW1 = ROW0 + 1;
        wire product_sign0;
        wire product_sign1;

        assign product_sign0 =
            weight_signs_r2[WEIGHT_LANE * NUM_PE + PE0] ^
            activation_signs_r2[
                TREE_SLR * ACT_SIGN_BITS + ACT_GROUP * ARRAY_ROWS + ROW0];
        assign product_sign1 =
            weight_signs_r2[WEIGHT_LANE * NUM_PE + PE1] ^
            activation_signs_r2[
                TREE_SLR * ACT_SIGN_BITS + ACT_GROUP * ARRAY_ROWS + ROW1];

        w4a4_signed_magnitude_pair_add add (
            .magnitude0(magnitudes[(CHANNEL * NUM_PE + PE0) * MAG_BITS +: MAG_BITS]),
            .sign0(product_sign0),
            .magnitude1(magnitudes[(CHANNEL * NUM_PE + PE1) * MAG_BITS +: MAG_BITS]),
            .sign1(product_sign1),
            .value(level32[level32_node * 8 +: 8])
        );
    end
endgenerate

// Channels 0..7 are grouped four at a time inside each physical column. This
// keeps all 16 DSPs of a tree group in the same SLR as its multiplier column.
genvar tree_col;
genvar tree_channel_group;
genvar tree_lane;
genvar tree_pair;
generate
    for (tree_col = 0; tree_col < ARRAY_COLS;
         tree_col = tree_col + 1) begin : tree_col_gen
        for (tree_channel_group = 0; tree_channel_group < 2;
             tree_channel_group = tree_channel_group + 1) begin : tree_channel_group_gen
            wire [4 * PAIRS_PER_TREE * 8 - 1:0] pair_values;
            wire [4 * 13 - 1:0] tree_sums;
            for (tree_lane = 0; tree_lane < 4;
                 tree_lane = tree_lane + 1) begin : lane_gen
                localparam integer TREE_CHANNEL =
                    tree_channel_group * 4 + tree_lane;
                localparam integer TREE_INDEX =
                    TREE_CHANNEL * ARRAY_COLS + tree_col;
                for (tree_pair = 0; tree_pair < PAIRS_PER_TREE;
                     tree_pair = tree_pair + 1) begin : pair_gen
                    assign pair_values[
                        (tree_lane * PAIRS_PER_TREE + tree_pair) * 8 +: 8] =
                        level32[
                            (TREE_INDEX * PAIRS_PER_TREE + tree_pair) * 8 +:
                            8];
                end
                assign channel_sums[TREE_INDEX * 13 +: 13] =
                    tree_sums[tree_lane * 13 +: 13];
            end
            w4a4_overlap_tree4 tree (
                .clk(ap_clk), .ce(pipeline_ce),
                .pair_values(pair_values), .sums(tree_sums)
            );
        end
    end
endgenerate

// Channel 8 is packed across four neighboring columns. Groups are restarted at
// columns 22 and 42, so no PCIN cascade crosses an SLR boundary. The two unused
// lanes in the edge groups are tied to zero.
genvar channel8_group;
genvar channel8_lane;
genvar channel8_pair;
generate
    for (channel8_group = 0; channel8_group < 17;
         channel8_group = channel8_group + 1) begin : channel8_group_gen
        localparam integer BASE_COL =
            (channel8_group < 6) ?
                channel8_group * 4 :
            ((channel8_group < 11) ?
                22 + (channel8_group - 6) * 4 :
                42 + (channel8_group - 11) * 4);
        localparam integer END_COL =
            (channel8_group < 6) ? 22 :
            ((channel8_group < 11) ? 42 : 64);
        wire [4 * PAIRS_PER_TREE * 8 - 1:0] pair_values;
        wire [4 * 13 - 1:0] tree_sums;
        for (channel8_lane = 0; channel8_lane < 4;
             channel8_lane = channel8_lane + 1) begin : lane_gen
            localparam integer COL = BASE_COL + channel8_lane;
            if (COL < END_COL) begin : active_lane_gen
                localparam integer TREE_INDEX =
                    8 * ARRAY_COLS + COL;
                for (channel8_pair = 0;
                     channel8_pair < PAIRS_PER_TREE;
                     channel8_pair = channel8_pair + 1) begin : pair_gen
                    assign pair_values[
                        (channel8_lane * PAIRS_PER_TREE +
                         channel8_pair) * 8 +: 8] =
                        level32[
                            (TREE_INDEX * PAIRS_PER_TREE +
                             channel8_pair) * 8 +: 8];
                end
                assign channel_sums[TREE_INDEX * 13 +: 13] =
                    tree_sums[channel8_lane * 13 +: 13];
            end else begin : inactive_lane_gen
                assign pair_values[
                    channel8_lane * PAIRS_PER_TREE * 8 +:
                    PAIRS_PER_TREE * 8] = {PAIRS_PER_TREE * 8{1'b0}};
            end
        end
        w4a4_overlap_tree4 tree (
            .clk(ap_clk), .ce(pipeline_ce),
            .pair_values(pair_values), .sums(tree_sums)
        );
    end
endgenerate

genvar output_lane;
generate
    for (output_lane = 0; output_lane < OUTPUT_LANES;
         output_lane = output_lane + 1) begin : output_gen
        assign packed_sums[output_lane * SUM_BITS +: SUM_BITS] =
            {{(SUM_BITS - 13){channel_sums[output_lane * 13 + 12]}},
             channel_sums[output_lane * 13 +: 13]};
    end
endgenerate

assign sum0 = packed_sums[0 * CHANNEL_SUM_BITS +: CHANNEL_SUM_BITS];
assign sum1 = packed_sums[1 * CHANNEL_SUM_BITS +: CHANNEL_SUM_BITS];
assign sum2 = packed_sums[2 * CHANNEL_SUM_BITS +: CHANNEL_SUM_BITS];
assign sum3 = packed_sums[3 * CHANNEL_SUM_BITS +: CHANNEL_SUM_BITS];
assign sum4 = packed_sums[4 * CHANNEL_SUM_BITS +: CHANNEL_SUM_BITS];
assign sum5 = packed_sums[5 * CHANNEL_SUM_BITS +: CHANNEL_SUM_BITS];
assign sum6 = packed_sums[6 * CHANNEL_SUM_BITS +: CHANNEL_SUM_BITS];
assign sum7 = packed_sums[7 * CHANNEL_SUM_BITS +: CHANNEL_SUM_BITS];
assign sum8 = packed_sums[8 * CHANNEL_SUM_BITS +: CHANNEL_SUM_BITS];

always @(posedge ap_clk) begin
    if (ap_rst) begin
        valid_pipe <= {PIPELINE_LATENCY{1'b0}};
    end else if (pipeline_ce) begin
        valid_pipe <= {
            valid_pipe[PIPELINE_LATENCY-2:0], accept_input};
    end
end

wire write_results =
    valid_pipe[PIPELINE_LATENCY-1] & all_outputs_ready & ap_ce;
assign sum0_write = write_results;
assign sum1_write = write_results;
assign sum2_write = write_results;
assign sum3_write = write_results;
assign sum4_write = write_results;
assign sum5_write = write_results;
assign sum6_write = write_results;
assign sum7_write = write_results;
assign sum8_write = write_results;
assign pipeline_result_valid = valid_pipe[PIPELINE_LATENCY-1];
assign pipeline_idle = ~(|valid_pipe);

endmodule
