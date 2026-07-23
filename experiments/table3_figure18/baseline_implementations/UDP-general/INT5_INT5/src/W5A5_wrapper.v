/**
 * @module W5A5_array
 * @brief Parameterized systolic array wrapper for the W5A5 PE.
 *
 * Interface mapping (NO mode signal):
 * - w0, w1: per-PE weights, flattened in column-major order:
 *   pe_index = col * ARRAY_ROWS + row
 * - a0, a1: row vectors shared across all columns:
 *   a?(row) is broadcast to every PE in that row.
 * - result0..result3: one 10-bit result per PE, same column-major flattening.
 *   result0 = w0*a0, result1 = w1*a0, result2 = w0*a1, result3 = w1*a1
 */

`timescale 1ns/1ps

(* dont_touch = "1" *)
module W5A5_array #(
    parameter integer ARRAY_ROWS = 128,
    parameter integer ARRAY_COLS = 32,
    parameter integer PE_ADDR_BITS = 12,
    parameter integer ROW_ADDR_BITS = 6,
    parameter integer W_BITS = 5,
    parameter integer A_BITS = 5,
    parameter integer R_BITS = 10,
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

    // Per-PE weights (NUM_PE x 5-bit for each port)
    input wire [PE_ADDR_BITS-1:0] w0_address,
    input wire            w0_ce,
    input wire [W_BUS_BITS-1:0] w0,

    input wire [PE_ADDR_BITS-1:0] w1_address,
    input wire            w1_ce,
    input wire [W_BUS_BITS-1:0] w1,

    // Row-shared activations (ARRAY_ROWS x 5-bit for each port)
    input wire [ROW_ADDR_BITS-1:0] a0_address,
    input wire            a0_ce,
    input wire [A_BUS_BITS-1:0] a0,

    input wire [ROW_ADDR_BITS-1:0] a1_address,
    input wire            a1_ce,
    input wire [A_BUS_BITS-1:0] a1,

    // Per-PE results (NUM_PE x 10-bit for each output port)
    input wire [PE_ADDR_BITS-1:0] result0_address,
    input wire            result0_ce,
    input wire            result0_we,
    output wire [R_BUS_BITS-1:0] result0,

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

            wire [4:0] w0_local = w0[W_OFFSET +: W_BITS];
            wire [4:0] w1_local = w1[W_OFFSET +: W_BITS];

            wire [4:0] a0_local = a0[A_OFFSET +: A_BITS];
            wire [4:0] a1_local = a1[A_OFFSET +: A_BITS];

            wire [9:0] result0_local;
            wire [9:0] result1_local;
            wire [9:0] result2_local;
            wire [9:0] result3_local;

            W5A5 pe_core (
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

// ============================================================================
// Self-contained copy of the validated W5A5 PE so the HLS black-box
// RTL stays in one file and avoids a module-name collision at the top level.
// ============================================================================
// LUT:42  FF:13
module W5A5(
    input  wire                     clock,
    input  wire                     reset,
    input  wire signed [4:0]   wArr_in_0,
    input  wire signed [4:0]   wArr_in_1,
    input  wire signed [4:0]   aArr_in_0,
    input  wire signed [4:0]   aArr_in_1,
    input  wire                     valid_in,
    output reg                      valid_out,
    output wire signed [9:0]   pArr_out_0,
    output wire signed [9:0]   pArr_out_1,
    output wire signed [9:0]   pArr_out_2,
    output wire signed [9:0]   pArr_out_3
);

    // reset is unused; kept for compatibility with Chisel-generated top ports.

    localparam [4:0] POST_BIAS = 5'd16;

    wire signed [4:0] neg_w_0 = -wArr_in_0;
    wire signed [4:0] neg_w_1 = -wArr_in_1;
    wire signed [4:0] neg_a_0 = -aArr_in_0;
    wire signed [4:0] neg_a_1 = -aArr_in_1;

    wire [8:0] packedElem_0 = {5'd0, wArr_in_0[3:0]};
    wire [8:0] packedElem_1 = {5'd0, wArr_in_1[3:0]};
    wire signed [17:0] packedportS = {packedElem_1, packedElem_0};

    wire [17:0] packedL_0 = {13'd0, aArr_in_0[4:0]};
    wire [17:0] packedL_1 = {13'd0, aArr_in_1[4:0]};
    wire signed [35:0] packedportL = {packedL_1, packedL_0};

    wire [8:0] preprocess_0 = (wArr_in_0 < 0) ? {neg_a_0[4:0], 4'd0} : 9'd0;
    wire [8:0] preprocess_1 = (wArr_in_1 < 0) ? {neg_a_0[4:0], 4'd0} : 9'd0;
    wire [8:0] preprocess_2 = (wArr_in_0 < 0) ? {neg_a_1[4:0], 4'd0} : 9'd0;
    wire [8:0] preprocess_3 = (wArr_in_1 < 0) ? {neg_a_1[4:0], 4'd0} : 9'd0;
    wire [35:0] packedpreprocess = {preprocess_3, preprocess_2, preprocess_1, preprocess_0};

    reg  [4:0] postprocessArr_0;
    reg  [4:0] postprocessArr_1;
    reg  signed [4:0] aArr_in_Reg_0;
    reg  signed [4:0] aArr_in_Reg_1;
    reg  signed [44:0] packedProduct;

    wire signed [17:0] packedportS_dsp = packedportS[17:0];
    wire signed [26:0] packedportL_dsp = packedportL[26:0];
    wire signed [44:0] packedpreprocess_dsp = $signed({{9{1'b0}}, packedpreprocess});

    always @(posedge clock) begin
        valid_out <= valid_in;
        if (wArr_in_0 < 0)
            postprocessArr_0 <= neg_w_0[4:0] - POST_BIAS;
        else
            postprocessArr_0 <= neg_w_0[4:0];
        if (wArr_in_1 < 0)
            postprocessArr_1 <= neg_w_1[4:0] - POST_BIAS;
        else
            postprocessArr_1 <= neg_w_1[4:0];
        aArr_in_Reg_0 <= aArr_in_0;
        aArr_in_Reg_1 <= aArr_in_1;
        packedProduct <= (packedportS_dsp * packedportL_dsp) + packedpreprocess_dsp;
    end

    wire signed [8:0] unpackedProduct_0 = packedProduct[8:0];
    wire [4:0] unpackedProductZext_0 = {1'b0, unpackedProduct_0[8:5]};
    wire [5:0] sum_0 = unpackedProductZext_0 + postprocessArr_0;
    assign pArr_out_0 = (aArr_in_Reg_0 < 0) ?
                          $signed({sum_0[4:0], unpackedProduct_0[4:0]}) :
                          $signed({unpackedProduct_0[8], unpackedProduct_0});

    wire signed [8:0] unpackedProduct_1 = packedProduct[17:9];
    wire [4:0] unpackedProductZext_1 = {1'b0, unpackedProduct_1[8:5]};
    wire [5:0] sum_1 = unpackedProductZext_1 + postprocessArr_1;
    assign pArr_out_1 = (aArr_in_Reg_0 < 0) ?
                          $signed({sum_1[4:0], unpackedProduct_1[4:0]}) :
                          $signed({unpackedProduct_1[8], unpackedProduct_1});

    wire signed [8:0] unpackedProduct_2 = packedProduct[26:18];
    wire [4:0] unpackedProductZext_2 = {1'b0, unpackedProduct_2[8:5]};
    wire [5:0] sum_2 = unpackedProductZext_2 + postprocessArr_0;
    assign pArr_out_2 = (aArr_in_Reg_1 < 0) ?
                          $signed({sum_2[4:0], unpackedProduct_2[4:0]}) :
                          $signed({unpackedProduct_2[8], unpackedProduct_2});

    wire signed [8:0] unpackedProduct_3 = packedProduct[35:27];
    wire [4:0] unpackedProductZext_3 = {1'b0, unpackedProduct_3[8:5]};
    wire [5:0] sum_3 = unpackedProductZext_3 + postprocessArr_1;
    assign pArr_out_3 = (aArr_in_Reg_1 < 0) ?
                          $signed({sum_3[4:0], unpackedProduct_3[4:0]}) :
                          $signed({unpackedProduct_3[8], unpackedProduct_3});

endmodule
