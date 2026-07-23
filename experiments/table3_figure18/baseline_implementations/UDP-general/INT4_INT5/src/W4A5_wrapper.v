/**
 * @module W4A5_array
 * @brief 64x64 systolic array wrapper for the W4A5 packed PE.
 *
 * Physical array:
 * - 64 rows x 64 columns = 4096 PEs.
 * - Each PE packs 2 signed int4 weights with 2 row-shared signed int5 activations.
 * - w0/w1 are per-PE weights in column-major order:
 *   pe_index = col * ARRAY_ROWS + row
 * - a0/a1 are row vectors shared across all columns.
 * - The four output ports keep one signed 9-bit result per PE.
 *   result0 = w0*a0, result1 = w1*a0, result2 = w0*a1, result3 = w1*a1
 *
 * No mode signal. Latency = 1. II = 1.
 */

`timescale 1ns/1ps

(* dont_touch = "1" *)
module W4A5_array #(
    parameter integer ARRAY_ROWS = 128,
    parameter integer ARRAY_COLS = 32,
    parameter integer PE_ADDR_BITS = 12,
    parameter integer ROW_ADDR_BITS = 6,
    parameter integer W_BITS = 4,
    parameter integer A_BITS = 5,
    parameter integer R_BITS = 9,
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
    output wire [R_BUS_BITS-1:0] result3
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
            wire [A_BITS-1:0] a0_local = a0[A_OFFSET +: A_BITS];
            wire [A_BITS-1:0] a1_local = a1[A_OFFSET +: A_BITS];

            wire [R_BITS-1:0] result0_local;
            wire [R_BITS-1:0] result1_local;
            wire [R_BITS-1:0] result2_local;
            wire [R_BITS-1:0] result3_local;

            W4A5 pe_core (
                .clock(ap_clk),
                .reset(ap_rst),
                .wArr_in_0(w0_local),
                .wArr_in_1(w1_local),
                .aArr_in_0(a0_local),
                .aArr_in_1(a1_local),
                .valid_in(ap_start),
                .valid_out(),
                .pArr_out_0(result0_local),
                .pArr_out_1(result1_local),
                .pArr_out_2(result2_local),
                .pArr_out_3(result3_local)
            );

            assign result0[R_OFFSET +: R_BITS] = result0_local;
            assign result1[R_OFFSET +: R_BITS] = result1_local;
            assign result2[R_OFFSET +: R_BITS] = result2_local;
            assign result3[R_OFFSET +: R_BITS] = result3_local;
        end
    end
endgenerate

assign ap_idle = ~(ap_start | dly1);
assign ap_ready = dly1;
assign ap_done = dly1;

endmodule

// LUT:36  FF:11
module W4A5(
    input  wire                     clock,
    input  wire                     reset,
    input  wire signed [3:0]   wArr_in_0,
    input  wire signed [3:0]   wArr_in_1,
    input  wire signed [4:0]   aArr_in_0,
    input  wire signed [4:0]   aArr_in_1,
    input  wire                     valid_in,
    output reg                      valid_out,
    output wire signed [8:0]   pArr_out_0,
    output wire signed [8:0]   pArr_out_1,
    output wire signed [8:0]   pArr_out_2,
    output wire signed [8:0]   pArr_out_3
);

    // reset is unused; kept for compatibility with Chisel-generated top ports.

    localparam [3:0] POST_BIAS = 4'd8;

    wire signed [3:0] neg_w_0 = -wArr_in_0;
    wire signed [3:0] neg_w_1 = -wArr_in_1;
    wire signed [4:0] neg_a_0 = -aArr_in_0;
    wire signed [4:0] neg_a_1 = -aArr_in_1;

    wire [7:0] packedElem_0 = {5'd0, wArr_in_0[2:0]};
    wire [7:0] packedElem_1 = {5'd0, wArr_in_1[2:0]};
    wire signed [15:0] packedportS = {packedElem_1, packedElem_0};

    wire [15:0] packedL_0 = {11'd0, aArr_in_0[4:0]};
    wire [15:0] packedL_1 = {11'd0, aArr_in_1[4:0]};
    wire signed [31:0] packedportL = {packedL_1, packedL_0};

    wire [7:0] preprocess_0 = (wArr_in_0 < 0) ? {neg_a_0[4:0], 3'd0} : 8'd0;
    wire [7:0] preprocess_1 = (wArr_in_1 < 0) ? {neg_a_0[4:0], 3'd0} : 8'd0;
    wire [7:0] preprocess_2 = (wArr_in_0 < 0) ? {neg_a_1[4:0], 3'd0} : 8'd0;
    wire [7:0] preprocess_3 = (wArr_in_1 < 0) ? {neg_a_1[4:0], 3'd0} : 8'd0;
    wire [31:0] packedpreprocess = {preprocess_3, preprocess_2, preprocess_1, preprocess_0};

    reg  [3:0] postprocessArr_0;
    reg  [3:0] postprocessArr_1;
    reg  signed [4:0] aArr_in_Reg_0;
    reg  signed [4:0] aArr_in_Reg_1;
    reg  signed [44:0] packedProduct;

    wire signed [17:0] packedportS_dsp = $signed({{2{1'b0}}, packedportS});
    wire signed [26:0] packedportL_dsp = packedportL[26:0];
    wire signed [44:0] packedpreprocess_dsp = $signed({{13{1'b0}}, packedpreprocess});

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
        aArr_in_Reg_0 <= aArr_in_0;
        aArr_in_Reg_1 <= aArr_in_1;
        packedProduct <= (packedportS_dsp * packedportL_dsp) + packedpreprocess_dsp;
    end

    wire signed [7:0] unpackedProduct_0 = packedProduct[7:0];
    wire [3:0] unpackedProductZext_0 = {1'b0, unpackedProduct_0[7:5]};
    wire [4:0] sum_0 = unpackedProductZext_0 + postprocessArr_0;
    assign pArr_out_0 = (aArr_in_Reg_0 < 0) ?
                          $signed({sum_0[3:0], unpackedProduct_0[4:0]}) :
                          $signed({unpackedProduct_0[7], unpackedProduct_0});

    wire signed [7:0] unpackedProduct_1 = packedProduct[15:8];
    wire [3:0] unpackedProductZext_1 = {1'b0, unpackedProduct_1[7:5]};
    wire [4:0] sum_1 = unpackedProductZext_1 + postprocessArr_1;
    assign pArr_out_1 = (aArr_in_Reg_0 < 0) ?
                          $signed({sum_1[3:0], unpackedProduct_1[4:0]}) :
                          $signed({unpackedProduct_1[7], unpackedProduct_1});

    wire signed [7:0] unpackedProduct_2 = packedProduct[23:16];
    wire [3:0] unpackedProductZext_2 = {1'b0, unpackedProduct_2[7:5]};
    wire [4:0] sum_2 = unpackedProductZext_2 + postprocessArr_0;
    assign pArr_out_2 = (aArr_in_Reg_1 < 0) ?
                          $signed({sum_2[3:0], unpackedProduct_2[4:0]}) :
                          $signed({unpackedProduct_2[7], unpackedProduct_2});

    wire signed [7:0] unpackedProduct_3 = packedProduct[31:24];
    wire [3:0] unpackedProductZext_3 = {1'b0, unpackedProduct_3[7:5]};
    wire [4:0] sum_3 = unpackedProductZext_3 + postprocessArr_1;
    assign pArr_out_3 = (aArr_in_Reg_1 < 0) ?
                          $signed({sum_3[3:0], unpackedProduct_3[4:0]}) :
                          $signed({unpackedProduct_3[7], unpackedProduct_3});

endmodule
