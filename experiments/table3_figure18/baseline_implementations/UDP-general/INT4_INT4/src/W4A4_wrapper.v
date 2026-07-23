/**
 * @module W4A4_array
 * @brief 64x64 systolic array wrapper for the W4A4 3x2 int4 packed multiplier.
 *
 * Physical array:
 * - 64 rows x 64 columns = 4096 PEs.
 * - Each PE packs 3 signed int4 weights with 2 signed int4 activations.
 * - w0/w1/w2 are per-PE weights in column-major order:
 *   pe_index = col * ARRAY_ROWS + row
 * - a0/a1 are row vectors shared across all columns.
 * - The six output ports keep one signed 8-bit result per PE.
 * - Latency = 1 cycle, II = 1.
 */

`timescale 1ns/1ps

(* dont_touch = "1" *)
module W4A4_array #(
    parameter integer ARRAY_ROWS = 128,
    parameter integer ARRAY_COLS = 32,
    parameter integer PE_ADDR_BITS = 12,
    parameter integer ROW_ADDR_BITS = 6,
    parameter integer W_BITS = 4,
    parameter integer A_BITS = 4,
    parameter integer R_BITS = 8,
    parameter integer NUM_PE = ARRAY_ROWS * ARRAY_COLS,
    parameter integer W_BUS_BITS = NUM_PE * W_BITS,
    parameter integer A_BUS_BITS = ARRAY_ROWS * A_BITS,
    parameter integer R_BUS_BITS = NUM_PE * R_BITS
)(
    input wire ap_clk,
    input wire ap_rst,
    input wire ap_ce,
    input wire ap_start,
    input wire ap_continue,
    output wire ap_done,
    output wire ap_idle,
    output wire ap_ready,

    input wire [PE_ADDR_BITS-1:0] w0_address,
    input wire w0_ce,
    input wire [W_BUS_BITS-1:0] w0,

    input wire [PE_ADDR_BITS-1:0] w1_address,
    input wire w1_ce,
    input wire [W_BUS_BITS-1:0] w1,

    input wire [PE_ADDR_BITS-1:0] w2_address,
    input wire w2_ce,
    input wire [W_BUS_BITS-1:0] w2,

    input wire [ROW_ADDR_BITS-1:0] a0_address,
    input wire a0_ce,
    input wire [A_BUS_BITS-1:0] a0,

    input wire [ROW_ADDR_BITS-1:0] a1_address,
    input wire a1_ce,
    input wire [A_BUS_BITS-1:0] a1,

    input wire [PE_ADDR_BITS-1:0] result0_address,
    input wire result0_ce,
    input wire result0_we,
    output wire [R_BUS_BITS-1:0] result0,

    input wire [PE_ADDR_BITS-1:0] result1_address,
    input wire result1_ce,
    input wire result1_we,
    output wire [R_BUS_BITS-1:0] result1,

    input wire [PE_ADDR_BITS-1:0] result2_address,
    input wire result2_ce,
    input wire result2_we,
    output wire [R_BUS_BITS-1:0] result2,

    input wire [PE_ADDR_BITS-1:0] result3_address,
    input wire result3_ce,
    input wire result3_we,
    output wire [R_BUS_BITS-1:0] result3,

    input wire [PE_ADDR_BITS-1:0] result4_address,
    input wire result4_ce,
    input wire result4_we,
    output wire [R_BUS_BITS-1:0] result4,

    input wire [PE_ADDR_BITS-1:0] result5_address,
    input wire result5_ce,
    input wire result5_we,
    output wire [R_BUS_BITS-1:0] result5
);

wire ce = ap_ce;
reg dly1;

always @(posedge ap_clk) begin
    if (ap_rst) begin
        dly1 <= 1'b0;
    end else if (ce) begin
        dly1 <= ap_start;
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

            wire [W_BITS-1:0] w0_local = w0[W_OFFSET +: W_BITS];
            wire [W_BITS-1:0] w1_local = w1[W_OFFSET +: W_BITS];
            wire [W_BITS-1:0] w2_local = w2[W_OFFSET +: W_BITS];
            wire [A_BITS-1:0] a0_local = a0[A_OFFSET +: A_BITS];
            wire [A_BITS-1:0] a1_local = a1[A_OFFSET +: A_BITS];

            wire [R_BITS-1:0] result0_local;
            wire [R_BITS-1:0] result1_local;
            wire [R_BITS-1:0] result2_local;
            wire [R_BITS-1:0] result3_local;
            wire [R_BITS-1:0] result4_local;
            wire [R_BITS-1:0] result5_local;

            W4A4 pe_core (
                .clock(ap_clk),
                .reset(ap_rst),
                .wArr_in_0(w0_local),
                .wArr_in_1(w1_local),
                .wArr_in_2(w2_local),
                .aArr_in_0(a0_local),
                .aArr_in_1(a1_local),
                .valid_in(ap_start),
                .valid_out(),
                .pArr_out_0(result0_local),
                .pArr_out_1(result1_local),
                .pArr_out_2(result2_local),
                .pArr_out_3(result3_local),
                .pArr_out_4(result4_local),
                .pArr_out_5(result5_local)
            );

            assign result0[R_OFFSET +: R_BITS] = result0_local;
            assign result1[R_OFFSET +: R_BITS] = result1_local;
            assign result2[R_OFFSET +: R_BITS] = result2_local;
            assign result3[R_OFFSET +: R_BITS] = result3_local;
            assign result4[R_OFFSET +: R_BITS] = result4_local;
            assign result5[R_OFFSET +: R_BITS] = result5_local;
        end
    end
endgenerate

assign ap_idle = ~(ap_start | dly1);
assign ap_ready = dly1;
assign ap_done = dly1;

endmodule

// ============================================================================
// Embedded PE module: W4A4
// LUT:48  FF:15
// ============================================================================
module W4A4(
    input  wire                     clock,
    input  wire                     reset,
    input  wire signed [3:0]   wArr_in_0,
    input  wire signed [3:0]   wArr_in_1,
    input  wire signed [3:0]   wArr_in_2,
    input  wire signed [3:0]   aArr_in_0,
    input  wire signed [3:0]   aArr_in_1,
    input  wire                     valid_in,
    output reg                      valid_out,
    output wire signed [7:0]   pArr_out_0,
    output wire signed [7:0]   pArr_out_1,
    output wire signed [7:0]   pArr_out_2,
    output wire signed [7:0]   pArr_out_3,
    output wire signed [7:0]   pArr_out_4,
    output wire signed [7:0]   pArr_out_5
);

    // reset is unused; kept for compatibility with Chisel-generated top ports.

    localparam [3:0] POST_BIAS = 4'd8;

    wire signed [3:0] neg_w_0 = -wArr_in_0;
    wire signed [3:0] neg_w_1 = -wArr_in_1;
    wire signed [3:0] neg_w_2 = -wArr_in_2;
    wire signed [3:0] neg_a_0 = -aArr_in_0;
    wire signed [3:0] neg_a_1 = -aArr_in_1;

    wire [6:0] packedElem_0 = {4'd0, wArr_in_0[2:0]};
    wire [6:0] packedElem_1 = {4'd0, wArr_in_1[2:0]};
    wire [6:0] packedElem_2 = {4'd0, wArr_in_2[2:0]};
    wire signed [20:0] packedportS = {packedElem_2, packedElem_1, packedElem_0};

    wire [20:0] packedL_0 = {17'd0, aArr_in_0[3:0]};
    wire [20:0] packedL_1 = {17'd0, aArr_in_1[3:0]};
    wire signed [41:0] packedportL = {packedL_1, packedL_0};

    wire [6:0] preprocess_0 = (wArr_in_0 < 0) ? {neg_a_0[3:0], 3'd0} : 7'd0;
    wire [6:0] preprocess_1 = (wArr_in_1 < 0) ? {neg_a_0[3:0], 3'd0} : 7'd0;
    wire [6:0] preprocess_2 = (wArr_in_2 < 0) ? {neg_a_0[3:0], 3'd0} : 7'd0;
    wire [6:0] preprocess_3 = (wArr_in_0 < 0) ? {neg_a_1[3:0], 3'd0} : 7'd0;
    wire [6:0] preprocess_4 = (wArr_in_1 < 0) ? {neg_a_1[3:0], 3'd0} : 7'd0;
    wire [6:0] preprocess_5 = (wArr_in_2 < 0) ? {neg_a_1[3:0], 3'd0} : 7'd0;
    wire [41:0] packedpreprocess = {preprocess_5, preprocess_4, preprocess_3, preprocess_2, preprocess_1, preprocess_0};

    reg  [3:0] postprocessArr_0;
    reg  [3:0] postprocessArr_1;
    reg  [3:0] postprocessArr_2;
    reg  signed [3:0] aArr_in_Reg_0;
    reg  signed [3:0] aArr_in_Reg_1;
    reg  signed [44:0] packedProduct;

    wire signed [17:0] packedportS_dsp = packedportS[17:0];
    wire signed [26:0] packedportL_dsp = packedportL[26:0];
    wire signed [44:0] packedpreprocess_dsp = $signed({{3{1'b0}}, packedpreprocess});

    always @(posedge clock) begin
        valid_out <= valid_in;
        if (wArr_in_0 < 0)
            postprocessArr_0 <= neg_w_0[3:0] - POST_BIAS;
        else
            postprocessArr_0 <= neg_w_0[3:0];
        if (wArr_in_1 < 0)
            postprocessArr_1 <= neg_w_1[3:0] - POST_BIAS;
        else
            postprocessArr_1 <= neg_w_1[3:0];
        if (wArr_in_2 < 0)
            postprocessArr_2 <= neg_w_2[3:0] - POST_BIAS;
        else
            postprocessArr_2 <= neg_w_2[3:0];
        aArr_in_Reg_0 <= aArr_in_0;
        aArr_in_Reg_1 <= aArr_in_1;
        packedProduct <= (packedportS_dsp * packedportL_dsp) + packedpreprocess_dsp;
    end

    wire signed [6:0] unpackedProduct_0 = packedProduct[6:0];
    wire [3:0] unpackedProductZext_0 = {1'b0, unpackedProduct_0[6:4]};
    wire [4:0] sum_0 = unpackedProductZext_0 + postprocessArr_0;
    assign pArr_out_0 = (aArr_in_Reg_0 < 0) ?
                          $signed({sum_0[3:0], unpackedProduct_0[3:0]}) :
                          $signed({unpackedProduct_0[6], unpackedProduct_0});

    wire signed [6:0] unpackedProduct_1 = packedProduct[13:7];
    wire [3:0] unpackedProductZext_1 = {1'b0, unpackedProduct_1[6:4]};
    wire [4:0] sum_1 = unpackedProductZext_1 + postprocessArr_1;
    assign pArr_out_1 = (aArr_in_Reg_0 < 0) ?
                          $signed({sum_1[3:0], unpackedProduct_1[3:0]}) :
                          $signed({unpackedProduct_1[6], unpackedProduct_1});

    wire signed [6:0] unpackedProduct_2 = packedProduct[20:14];
    wire [3:0] unpackedProductZext_2 = {1'b0, unpackedProduct_2[6:4]};
    wire [4:0] sum_2 = unpackedProductZext_2 + postprocessArr_2;
    assign pArr_out_2 = (aArr_in_Reg_0 < 0) ?
                          $signed({sum_2[3:0], unpackedProduct_2[3:0]}) :
                          $signed({unpackedProduct_2[6], unpackedProduct_2});

    wire signed [6:0] unpackedProduct_3 = packedProduct[27:21];
    wire [3:0] unpackedProductZext_3 = {1'b0, unpackedProduct_3[6:4]};
    wire [4:0] sum_3 = unpackedProductZext_3 + postprocessArr_0;
    assign pArr_out_3 = (aArr_in_Reg_1 < 0) ?
                          $signed({sum_3[3:0], unpackedProduct_3[3:0]}) :
                          $signed({unpackedProduct_3[6], unpackedProduct_3});

    wire signed [6:0] unpackedProduct_4 = packedProduct[34:28];
    wire [3:0] unpackedProductZext_4 = {1'b0, unpackedProduct_4[6:4]};
    wire [4:0] sum_4 = unpackedProductZext_4 + postprocessArr_1;
    assign pArr_out_4 = (aArr_in_Reg_1 < 0) ?
                          $signed({sum_4[3:0], unpackedProduct_4[3:0]}) :
                          $signed({unpackedProduct_4[6], unpackedProduct_4});

    wire signed [6:0] unpackedProduct_5 = packedProduct[41:35];
    wire [3:0] unpackedProductZext_5 = {1'b0, unpackedProduct_5[6:4]};
    wire [4:0] sum_5 = unpackedProductZext_5 + postprocessArr_2;
    assign pArr_out_5 = (aArr_in_Reg_1 < 0) ?
                          $signed({sum_5[3:0], unpackedProduct_5[3:0]}) :
                          $signed({unpackedProduct_5[6], unpackedProduct_5});

endmodule
