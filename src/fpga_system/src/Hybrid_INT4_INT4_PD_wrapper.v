/**
 * @module Hybrid_INT4_INT4_PD
 * @brief Parameterized DSP array wrapper for the Hybrid INT4xINT4 packed PE.
 *
 * Interface mapping:
 * - mode is shared by the whole ARRAY_ROWS x ARRAY_COLS array.
 * - w1~w7 remain per-PE weights, flattened in column-major order:
 *   pe_index = col * ARRAY_ROWS + row
 * - a1~a3 are row vectors shared across all columns:
 *   a?(row) is broadcast to every PE in that row.
 * - result1~result9 keep one 7-bit result per PE, using the same
 *   column-major flattening as the weight ports.
 */

`timescale 1ns/1ps

(* dont_touch = "1" *)
module Hybrid_INT4_INT4_PD #(
    parameter integer ARRAY_ROWS = 64,
    parameter integer ARRAY_COLS = 64,
    parameter integer PE_ADDR_BITS = 12,
    parameter integer ROW_ADDR_BITS = 6,
    parameter integer W_BITS = 4,
    parameter integer A_BITS = 4,
    parameter integer R_BITS = 7,
    parameter integer NUM_PE = ARRAY_ROWS * ARRAY_COLS,
    parameter integer W_BUS_BITS = NUM_PE * W_BITS,
    parameter integer A_BUS_BITS = ARRAY_ROWS * A_BITS,
    parameter integer R_BUS_BITS = NUM_PE * R_BITS
)(
    // Clock and ap_ctrl_chain signals
    input wire        ap_clk,
    input wire        ap_rst,
    input wire        ap_ce,
    input wire        ap_start,
    input wire        ap_continue,
    output wire       ap_done,
    output wire       ap_idle,
    output wire       ap_ready,

    // Shared mode for the whole array
    input wire        mode,

    // Per-PE weights (NUM_PE x 4-bit for each port)
    input wire [PE_ADDR_BITS-1:0] w1_address,
    input wire            w1_ce,
    input wire [W_BUS_BITS-1:0] w1,

    input wire [PE_ADDR_BITS-1:0] w2_address,
    input wire            w2_ce,
    input wire [W_BUS_BITS-1:0] w2,

    input wire [PE_ADDR_BITS-1:0] w3_address,
    input wire            w3_ce,
    input wire [W_BUS_BITS-1:0] w3,

    input wire [PE_ADDR_BITS-1:0] w4_address,
    input wire            w4_ce,
    input wire [W_BUS_BITS-1:0] w4,

    input wire [PE_ADDR_BITS-1:0] w5_address,
    input wire            w5_ce,
    input wire [W_BUS_BITS-1:0] w5,

    input wire [PE_ADDR_BITS-1:0] w6_address,
    input wire            w6_ce,
    input wire [W_BUS_BITS-1:0] w6,

    input wire [PE_ADDR_BITS-1:0] w7_address,
    input wire            w7_ce,
    input wire [W_BUS_BITS-1:0] w7,

    // Row-shared activations (ARRAY_ROWS x 4-bit for each port)
    input wire [ROW_ADDR_BITS-1:0] a1_address,
    input wire            a1_ce,
    input wire [A_BUS_BITS-1:0] a1,

    input wire [ROW_ADDR_BITS-1:0] a2_address,
    input wire            a2_ce,
    input wire [A_BUS_BITS-1:0] a2,

    input wire [ROW_ADDR_BITS-1:0] a3_address,
    input wire            a3_ce,
    input wire [A_BUS_BITS-1:0] a3,

    // Per-PE results (NUM_PE x 7-bit for each output port)
    input wire [PE_ADDR_BITS-1:0] result1_address,
    input wire            result1_ce,
    input wire            result1_we,
    output wire [R_BUS_BITS-1:0] result1,

    input wire [PE_ADDR_BITS-1:0] result2_address,
    input wire            result2_ce,
    input wire            result2_we,
    output wire [R_BUS_BITS-1:0] result2,

    input wire [PE_ADDR_BITS-1:0] result3_address,
    input wire            result3_ce,
    input wire            result3_we,
    output wire [R_BUS_BITS-1:0] result3,

    input wire [PE_ADDR_BITS-1:0] result4_address,
    input wire            result4_ce,
    input wire            result4_we,
    output wire [R_BUS_BITS-1:0] result4,

    input wire [PE_ADDR_BITS-1:0] result5_address,
    input wire            result5_ce,
    input wire            result5_we,
    output wire [R_BUS_BITS-1:0] result5,

    input wire [PE_ADDR_BITS-1:0] result6_address,
    input wire            result6_ce,
    input wire            result6_we,
    output wire [R_BUS_BITS-1:0] result6,

    input wire [PE_ADDR_BITS-1:0] result7_address,
    input wire            result7_ce,
    input wire            result7_we,
    output wire [R_BUS_BITS-1:0] result7,

    input wire [PE_ADDR_BITS-1:0] result8_address,
    input wire            result8_ce,
    input wire            result8_we,
    output wire [R_BUS_BITS-1:0] result8,

    input wire [PE_ADDR_BITS-1:0] result9_address,
    input wire            result9_ce,
    input wire            result9_we,
    output wire [R_BUS_BITS-1:0] result9
);

wire ce = ap_ce;
reg dly1, dly2, dly3;

always @(posedge ap_clk) begin
    if (ap_rst) begin
        dly1 <= 1'b0;
        dly2 <= 1'b0;
        dly3 <= 1'b0;
    end else if (ce) begin
        dly1 <= ap_start;
        dly2 <= dly1;
        dly3 <= dly2;
    end
end

genvar col;
genvar row;
generate
    for (col = 0; col < ARRAY_COLS; col = col + 1) begin : col_gen
        for (row = 0; row < ARRAY_ROWS; row = row + 1) begin : row_gen
            localparam integer PE_INDEX = (col * ARRAY_ROWS) + row;
            localparam integer W_OFFSET = PE_INDEX * W_BITS;
            localparam integer A_OFFSET = row * A_BITS;
            localparam integer R_OFFSET = PE_INDEX * R_BITS;

            wire [3:0] w1_local = w1[W_OFFSET +: W_BITS];
            wire [3:0] w2_local = w2[W_OFFSET +: W_BITS];
            wire [3:0] w3_local = w3[W_OFFSET +: W_BITS];
            wire [3:0] w4_local = w4[W_OFFSET +: W_BITS];
            wire [3:0] w5_local = w5[W_OFFSET +: W_BITS];
            wire [3:0] w6_local = w6[W_OFFSET +: W_BITS];
            wire [3:0] w7_local = w7[W_OFFSET +: W_BITS];

            wire [3:0] a1_local = a1[A_OFFSET +: A_BITS];
            wire [3:0] a2_local = a2[A_OFFSET +: A_BITS];
            wire [3:0] a3_local = a3[A_OFFSET +: A_BITS];

            wire [6:0] result1_local;
            wire [6:0] result2_local;
            wire [6:0] result3_local;
            wire [6:0] result4_local;
            wire [6:0] result5_local;
            wire [6:0] result6_local;
            wire [6:0] result7_local;
            wire [6:0] result8_local;
            wire [6:0] result9_local;

            Hybrid_INT4_INT4_PD_core dsp_core (
                .clk(ap_clk),
                .mode(mode),
                .w1(w1_local),
                .w2(w2_local),
                .w3(w3_local),
                .w4(w4_local),
                .w5(w5_local),
                .w6(w6_local),
                .w7(w7_local),
                .a1(a1_local),
                .a2(a2_local),
                .a3(a3_local),
                .result1(result1_local),
                .result2(result2_local),
                .result3(result3_local),
                .result4(result4_local),
                .result5(result5_local),
                .result6(result6_local),
                .result7(result7_local),
                .result8(result8_local),
                .result9(result9_local)
            );

            assign result1[R_OFFSET +: R_BITS] = result1_local;
            assign result2[R_OFFSET +: R_BITS] = result2_local;
            assign result3[R_OFFSET +: R_BITS] = result3_local;
            assign result4[R_OFFSET +: R_BITS] = result4_local;
            assign result5[R_OFFSET +: R_BITS] = result5_local;
            assign result6[R_OFFSET +: R_BITS] = result6_local;
            assign result7[R_OFFSET +: R_BITS] = result7_local;
            assign result8[R_OFFSET +: R_BITS] = result8_local;
            assign result9[R_OFFSET +: R_BITS] = result9_local;
        end
    end
endgenerate

assign ap_idle = ~(ap_start | dly1 | dly2 | dly3);
assign ap_ready = dly3;
assign ap_done = dly3;

endmodule

// Self-contained copy of the validated single-DSP PE so the HLS black-box
// RTL stays in one file and avoids a module-name collision at the top level.
module Hybrid_INT4_INT4_PD_core(
    input wire        clk,
    input wire        mode,
    input wire [3:0]  w1,
    input wire [3:0]  w2,
    input wire [3:0]  w3,
    input wire [3:0]  w4,
    input wire [3:0]  w5,
    input wire [3:0]  w6,
    input wire [3:0]  w7,
    input wire [3:0]  a1,
    input wire [3:0]  a2,
    input wire [3:0]  a3,
    output wire [6:0] result1,
    output wire [6:0] result2,
    output wire [6:0] result3,
    output wire [6:0] result4,
    output wire [6:0] result5,
    output wire [6:0] result6,
    output wire [6:0] result7,
    output wire [6:0] result8,
    output wire [6:0] result9
);

wire [47:0] dsp_P;

//================================================================
// Input processing: sign and magnitude extraction
//================================================================
wire w1_sign, w2_sign, w3_sign, w4_sign, w5_sign, w6_sign, w7_sign;
wire [2:0] w1_mag, w2_mag, w3_mag, w4_mag, w5_mag, w6_mag, w7_mag;

assign w1_sign = w1[3]; assign w1_mag = w1[2:0];
assign w2_sign = w2[3]; assign w2_mag = w2[2:0];
assign w3_sign = w3[3]; assign w3_mag = w3[2:0];
assign w4_sign = w4[3]; assign w4_mag = w4[2:0];
assign w5_sign = w5[3]; assign w5_mag = w5[2:0];
assign w6_sign = w6[3]; assign w6_mag = w6[2:0];
assign w7_sign = w7[3]; assign w7_mag = w7[2:0];

wire a1_sign, a2_sign, a3_sign;
wire [2:0] a1_mag, a2_mag, a3_mag;
wire [3:0] a1_temp_mag, a2_temp_mag, a3_temp_mag;

assign a1_sign = a1[3];
assign a1_temp_mag = a1[3] ? (~a1[3:0] + 1'b1) : a1[3:0];
assign a1_mag = (a1_temp_mag == 4'b1000) ? 3'b111 : a1_temp_mag[2:0];

assign a2_sign = a2[3];
assign a2_temp_mag = a2[3] ? (~a2[3:0] + 1'b1) : a2[3:0];
assign a2_mag = (a2_temp_mag == 4'b1000) ? 3'b111 : a2_temp_mag[2:0];

assign a3_sign = a3[3];
assign a3_temp_mag = a3[3] ? (~a3[3:0] + 1'b1) : a3[3:0];
assign a3_mag = (a3_temp_mag == 4'b1000) ? 3'b111 : a3_temp_mag[2:0];

//================================================================
// LSB compensation
//================================================================
wire [1:0] LSB_Res2, LSB_Res3, LSB_Res5, LSB_Res6, LSB_Res7, LSB_Res8, LSB_Res9;
wire [2:0] LSB_Res4_P, LSB_Res4_D;

assign LSB_Res2[0] = (mode == 0) ? (w1_mag[0] & a2_mag[0]) : (w2_mag[0] & a1_mag[0]);
assign LSB_Res2[1] = (mode == 0) ? ((w1_mag[0] & a2_mag[1]) ^ (w1_mag[1] & a2_mag[0])) :
                                    ((w2_mag[0] & a1_mag[1]) ^ (w2_mag[1] & a1_mag[0]));

assign LSB_Res3[0] = (mode == 0) ? (w1_mag[0] & a3_mag[0]) : (w3_mag[0] & a1_mag[0]);
assign LSB_Res3[1] = (mode == 0) ? ((w1_mag[0] & a3_mag[1]) ^ (w1_mag[1] & a3_mag[0])) :
                                    ((w3_mag[0] & a1_mag[1]) ^ (w3_mag[1] & a1_mag[0]));

wire c1_lsb4_P, c1_lsb4_D;
assign LSB_Res4_P[0] = w2_mag[0] & a1_mag[0];
assign LSB_Res4_P[1] = (w2_mag[0] & a1_mag[1]) ^ (w2_mag[1] & a1_mag[0]);
assign c1_lsb4_P = (w2_mag[0] & a1_mag[1]) & (w2_mag[1] & a1_mag[0]);
assign LSB_Res4_P[2] = (w2_mag[0] & a1_mag[2]) ^ (w2_mag[1] & a1_mag[1]) ^ (w2_mag[2] & a1_mag[0]) ^ c1_lsb4_P;

assign LSB_Res4_D[0] = w4_mag[0] & a1_mag[0];
assign LSB_Res4_D[1] = (w4_mag[0] & a1_mag[1]) ^ (w4_mag[1] & a1_mag[0]);
assign c1_lsb4_D = (w4_mag[0] & a1_mag[1]) & (w4_mag[1] & a1_mag[0]);
assign LSB_Res4_D[2] = (w4_mag[0] & a1_mag[2]) ^ (w4_mag[1] & a1_mag[1]) ^ (w4_mag[2] & a1_mag[0]) ^ c1_lsb4_D;

assign LSB_Res5[0] = (mode == 0) ? (w2_mag[0] & a2_mag[0]) : (w5_mag[0] & a1_mag[0]);
assign LSB_Res5[1] = (mode == 0) ? ((w2_mag[0] & a2_mag[1]) ^ (w2_mag[1] & a2_mag[0])) :
                                    ((w5_mag[0] & a1_mag[1]) ^ (w5_mag[1] & a1_mag[0]));

assign LSB_Res6[0] = (mode == 0) ? (w2_mag[0] & a3_mag[0]) : (w6_mag[0] & a1_mag[0]);
assign LSB_Res6[1] = (mode == 0) ? ((w2_mag[0] & a3_mag[1]) ^ (w2_mag[1] & a3_mag[0])) :
                                    ((w6_mag[0] & a1_mag[1]) ^ (w6_mag[1] & a1_mag[0]));

assign LSB_Res7[0] = (mode == 0) ? (w3_mag[0] & a1_mag[0]) : (w7_mag[0] & a1_mag[0]);
assign LSB_Res7[1] = (mode == 0) ? ((w3_mag[0] & a1_mag[1]) ^ (w3_mag[1] & a1_mag[0])) :
                                    ((w7_mag[0] & a1_mag[1]) ^ (w7_mag[1] & a1_mag[0]));

assign LSB_Res8[0] = w3_mag[0] & a2_mag[0];
assign LSB_Res8[1] = (w3_mag[0] & a2_mag[1]) ^ (w3_mag[1] & a2_mag[0]);

assign LSB_Res9[0] = w3_mag[0] & a3_mag[0];
assign LSB_Res9[1] = (w3_mag[0] & a3_mag[1]) ^ (w3_mag[1] & a3_mag[0]);

//================================================================
// Sign bits
//================================================================
wire [8:0] signs_comb = (mode == 0) ? {
    (w3_sign ^ a3_sign), (w3_sign ^ a2_sign), (w3_sign ^ a1_sign),
    (w2_sign ^ a3_sign), (w2_sign ^ a2_sign), (w2_sign ^ a1_sign),
    (w1_sign ^ a3_sign), (w1_sign ^ a2_sign), (w1_sign ^ a1_sign)
} : {
    2'b0,
    (w7_sign ^ a1_sign), (w6_sign ^ a1_sign), (w5_sign ^ a1_sign),
    (w4_sign ^ a1_sign), (w3_sign ^ a1_sign), (w2_sign ^ a1_sign), (w1_sign ^ a1_sign)
};

//================================================================
// DSP packing
//================================================================
wire [26:0] dsp_A = (mode == 0) ?
    {1'b0, w3_mag, 9'b0, w2_mag, 7'b0, w1_mag} :
    {1'b0, w7_mag, 1'b0, w6_mag, 1'b0, w5_mag, 1'b0, w4_mag, w3_mag, 1'b0, w2_mag, 1'b0, w1_mag};

wire [17:0] dsp_B = (mode == 0) ?
    {7'b0, a3_mag, 1'b0, a2_mag, 1'b0, a1_mag} :
    {15'b0, a1_mag};

//================================================================
// Pipeline registers
//================================================================
reg mode_r1, mode_r2, mode_r3;
reg [8:0] signs_r1, signs_r2, signs_r3;
reg [1:0] LSB_Res2_r1, LSB_Res3_r1, LSB_Res5_r1, LSB_Res6_r1, LSB_Res7_r1, LSB_Res8_r1, LSB_Res9_r1;
reg [2:0] LSB_Res4_r1;
reg [1:0] LSB_Res2_r2, LSB_Res3_r2, LSB_Res5_r2, LSB_Res6_r2, LSB_Res7_r2, LSB_Res8_r2, LSB_Res9_r2;
reg [2:0] LSB_Res4_r2;
reg [1:0] LSB_Res2_r3, LSB_Res3_r3, LSB_Res5_r3, LSB_Res6_r3, LSB_Res7_r3, LSB_Res8_r3, LSB_Res9_r3;
reg [2:0] LSB_Res4_r3;

wire [47:0] dsp_C = (mode_r1 == 0) ? {
    15'b0, LSB_Res9_r1, 2'b0, LSB_Res8_r1, 2'b0, LSB_Res7_r1, 2'b0,
    LSB_Res6_r1, 2'b0, LSB_Res5_r1, 1'b0, LSB_Res4_r1, 1'b0,
    LSB_Res3_r1, 2'b0, LSB_Res2_r1, 4'b0
} : {
    23'b0, LSB_Res7_r1, 2'b0, LSB_Res6_r1, 2'b0, LSB_Res5_r1, 1'b0,
    LSB_Res4_r1, 1'b0, LSB_Res3_r1, 2'b0, LSB_Res2_r1, 4'b0
};

always @(posedge clk) begin
    mode_r1 <= mode;
    signs_r1 <= signs_comb;
    LSB_Res2_r1 <= LSB_Res2;
    LSB_Res3_r1 <= LSB_Res3;
    LSB_Res4_r1 <= (mode == 0) ? LSB_Res4_P : LSB_Res4_D;
    LSB_Res5_r1 <= LSB_Res5;
    LSB_Res6_r1 <= LSB_Res6;
    LSB_Res7_r1 <= LSB_Res7;
    LSB_Res8_r1 <= LSB_Res8;
    LSB_Res9_r1 <= LSB_Res9;
end

always @(posedge clk) begin
    mode_r2 <= mode_r1;
    signs_r2 <= signs_r1;
    LSB_Res2_r2 <= LSB_Res2_r1;
    LSB_Res3_r2 <= LSB_Res3_r1;
    LSB_Res4_r2 <= LSB_Res4_r1;
    LSB_Res5_r2 <= LSB_Res5_r1;
    LSB_Res6_r2 <= LSB_Res6_r1;
    LSB_Res7_r2 <= LSB_Res7_r1;
    LSB_Res8_r2 <= LSB_Res8_r1;
    LSB_Res9_r2 <= LSB_Res9_r1;
end

always @(posedge clk) begin
    mode_r3 <= mode_r2;
    signs_r3 <= signs_r2;
    LSB_Res2_r3 <= LSB_Res2_r2;
    LSB_Res3_r3 <= LSB_Res3_r2;
    LSB_Res4_r3 <= LSB_Res4_r2;
    LSB_Res5_r3 <= LSB_Res5_r2;
    LSB_Res6_r3 <= LSB_Res6_r2;
    LSB_Res7_r3 <= LSB_Res7_r2;
    LSB_Res8_r3 <= LSB_Res8_r2;
    LSB_Res9_r3 <= LSB_Res9_r2;
end

//================================================================
// DSP48E2 instance
//================================================================
DSP48E2 #(
    .AMULTSEL("A"), .A_INPUT("DIRECT"), .BMULTSEL("B"), .B_INPUT("DIRECT"), .PREADDINSEL("A"),
    .RND(48'h000000000000), .USE_MULT("MULTIPLY"), .USE_SIMD("ONE48"), .USE_WIDEXOR("FALSE"), .XORSIMD("XOR24_48_96"),
    .AUTORESET_PATDET("NO_RESET"), .AUTORESET_PRIORITY("RESET"), .MASK(48'h3fffffffffff), .PATTERN(48'h000000000000),
    .SEL_MASK("MASK"), .SEL_PATTERN("PATTERN"), .USE_PATTERN_DETECT("NO_PATDET"),
    .IS_ALUMODE_INVERTED(4'b0000), .IS_CARRYIN_INVERTED(1'b0), .IS_CLK_INVERTED(1'b0), .IS_INMODE_INVERTED(5'b00000),
    .IS_OPMODE_INVERTED(9'b000000000), .IS_RSTALLCARRYIN_INVERTED(1'b0), .IS_RSTALUMODE_INVERTED(1'b0),
    .IS_RSTA_INVERTED(1'b0), .IS_RSTB_INVERTED(1'b0), .IS_RSTCTRL_INVERTED(1'b0), .IS_RSTC_INVERTED(1'b0),
    .IS_RSTD_INVERTED(1'b0), .IS_RSTINMODE_INVERTED(1'b0), .IS_RSTM_INVERTED(1'b0), .IS_RSTP_INVERTED(1'b0),
    .ACASCREG(1), .ADREG(0), .ALUMODEREG(1), .AREG(1), .BCASCREG(1), .BREG(1), .CARRYINREG(1),
    .CARRYINSELREG(1), .CREG(1), .DREG(0), .INMODEREG(1), .MREG(1), .OPMODEREG(1), .PREG(1)
) dsp_inst (
    .ACOUT(), .BCOUT(), .CARRYCASCOUT(), .MULTSIGNOUT(), .PCOUT(), .OVERFLOW(), .PATTERNBDETECT(), .PATTERNDETECT(), .UNDERFLOW(),
    .CARRYOUT(), .P(dsp_P), .XOROUT(),
    .ACIN(30'b0), .BCIN(18'b0), .CARRYCASCIN(1'b0), .MULTSIGNIN(1'b0), .PCIN(48'b0),
    .ALUMODE(4'b0001), .CARRYINSEL(3'b000), .CLK(clk), .INMODE(5'b00000), .OPMODE(9'b000110101),
    .A(dsp_A), .B(dsp_B), .C(dsp_C), .CARRYIN(1'b1), .D(27'b0),
    .CEA1(1'b0), .CEA2(1'b1), .CEAD(1'b0), .CEALUMODE(1'b1), .CEB1(1'b0), .CEB2(1'b1), .CEC(1'b1),
    .CECARRYIN(1'b1), .CECTRL(1'b1), .CED(1'b0), .CEINMODE(1'b1), .CEM(1'b1), .CEP(1'b1),
    .RSTA(1'b0), .RSTALLCARRYIN(1'b0), .RSTALUMODE(1'b0), .RSTB(1'b0), .RSTC(1'b0), .RSTCTRL(1'b0),
    .RSTD(1'b0), .RSTINMODE(1'b0), .RSTM(1'b0), .RSTP(1'b0)
);

//================================================================
// Result extraction
//================================================================
wire [5:0] res1_mag, res2_mag, res3_mag, res4_mag, res5_mag, res6_mag, res7_mag, res8_mag, res9_mag;

assign res1_mag = dsp_P[5:0];
assign res2_mag = {dsp_P[9:6], LSB_Res2_r3};
assign res3_mag = {dsp_P[13:10], LSB_Res3_r3};
assign res4_mag = {dsp_P[16:14], LSB_Res4_r3};
assign res5_mag = {dsp_P[20:17], LSB_Res5_r3};
assign res6_mag = {dsp_P[24:21], LSB_Res6_r3};
assign res7_mag = {dsp_P[28:25], LSB_Res7_r3};
assign res8_mag = (mode_r3 == 0) ? {dsp_P[32:29], LSB_Res8_r3} : 6'b0;
assign res9_mag = (mode_r3 == 0) ? {dsp_P[36:33], LSB_Res9_r3} : 6'b0;

//================================================================
// Outputs
//================================================================
assign result1 = signs_r3[0] ? -$signed({1'b0, res1_mag}) : {1'b0, res1_mag};
assign result2 = signs_r3[1] ? -$signed({1'b0, res2_mag}) : {1'b0, res2_mag};
assign result3 = signs_r3[2] ? -$signed({1'b0, res3_mag}) : {1'b0, res3_mag};
assign result4 = signs_r3[3] ? -$signed({1'b0, res4_mag}) : {1'b0, res4_mag};
assign result5 = signs_r3[4] ? -$signed({1'b0, res5_mag}) : {1'b0, res5_mag};
assign result6 = signs_r3[5] ? -$signed({1'b0, res6_mag}) : {1'b0, res6_mag};
assign result7 = signs_r3[6] ? -$signed({1'b0, res7_mag}) : {1'b0, res7_mag};
assign result8 = signs_r3[7] ? -$signed({1'b0, res8_mag}) : {1'b0, res8_mag};
assign result9 = signs_r3[8] ? -$signed({1'b0, res9_mag}) : {1'b0, res9_mag};

endmodule
