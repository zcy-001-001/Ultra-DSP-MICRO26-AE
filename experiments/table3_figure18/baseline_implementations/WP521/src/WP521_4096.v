/**
 * @module WP521_4096
 * @brief 64x64 wrapper built from the corrected WP521 packed DSP cell.
 *
 * This wrapper mirrors the column-major PE layout used by the C model:
 * - `w0`/`w1` are flattened per-PE weights, `pe = col * ARRAY_ROWS + row`
 * - `a0`/`a1` are row activations broadcast across all columns
 * - outputs keep the same per-PE flattening
 *
 * The arithmetic now follows `WP521.v` exactly:
 * - no LSB compensation
 * - no post-processing two's-complement conversion
 * - direct signed 8-bit extraction from the packed DSP output
 */

`timescale 1ns/1ps

(* dont_touch = "1" *)
module WP521_4096 #(
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

    input wire [ROW_ADDR_BITS-1:0] a0_address,
    input wire a0_ce,
    input wire [A_BUS_BITS-1:0] a0,

    input wire [ROW_ADDR_BITS-1:0] a1_address,
    input wire a1_ce,
    input wire [A_BUS_BITS-1:0] a1,

    input wire [PE_ADDR_BITS-1:0] result_a0w0_address,
    input wire result_a0w0_ce,
    input wire result_a0w0_we,
    output wire [R_BUS_BITS-1:0] result_a0w0,

    input wire [PE_ADDR_BITS-1:0] result_a1w0_address,
    input wire result_a1w0_ce,
    input wire result_a1w0_we,
    output wire [R_BUS_BITS-1:0] result_a1w0,

    input wire [PE_ADDR_BITS-1:0] result_a0w1_address,
    input wire result_a0w1_ce,
    input wire result_a0w1_we,
    output wire [R_BUS_BITS-1:0] result_a0w1,

    input wire [PE_ADDR_BITS-1:0] result_a1w1_address,
    input wire result_a1w1_ce,
    input wire result_a1w1_we,
    output wire [R_BUS_BITS-1:0] result_a1w1
);

wire ce = ap_ce;
reg dly1;
reg dly2;
reg dly3;
reg dly4;

always @(posedge ap_clk) begin
    if (ap_rst) begin
        dly1 <= 1'b0;
        dly2 <= 1'b0;
        dly3 <= 1'b0;
        dly4 <= 1'b0;
    end else if (ce) begin
        dly1 <= ap_start;
        dly2 <= dly1;
        dly3 <= dly2;
        dly4 <= dly3;
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

            wire signed [W_BITS-1:0] w0_local = w0[W_OFFSET +: W_BITS];
            wire signed [W_BITS-1:0] w1_local = w1[W_OFFSET +: W_BITS];
            wire [A_BITS-1:0] a0_local = a0[A_OFFSET +: A_BITS];
            wire [A_BITS-1:0] a1_local = a1[A_OFFSET +: A_BITS];

            wire signed [R_BITS-1:0] result_a0w0_local;
            wire signed [R_BITS-1:0] result_a0w1_local;
            wire signed [R_BITS-1:0] result_a1w0_local;
            wire signed [R_BITS-1:0] result_a1w1_local;

            WP521 dsp_core (
                .clk(ap_clk),
                .w0(w0_local),
                .w1(w1_local),
                .a0(a0_local),
                .a1(a1_local),
                .r00(result_a0w0_local),
                .r01(result_a0w1_local),
                .r10(result_a1w0_local),
                .r11(result_a1w1_local)
            );

            assign result_a0w0[R_OFFSET +: R_BITS] = result_a0w0_local;
            assign result_a1w0[R_OFFSET +: R_BITS] = result_a1w0_local;
            assign result_a0w1[R_OFFSET +: R_BITS] = result_a0w1_local;
            assign result_a1w1[R_OFFSET +: R_BITS] = result_a1w1_local;
        end
    end
endgenerate

assign ap_idle = ~(ap_start | dly1 | dly2 | dly3 | dly4);
assign ap_ready = dly4;
assign ap_done = dly4;

endmodule
