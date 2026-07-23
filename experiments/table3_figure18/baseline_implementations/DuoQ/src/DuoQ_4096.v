/**
 * @module DuoQ_4096
 * @brief 64x64 wrapper built from the corrected DuoQ INT4 x INT4 packed DSP cell.
 *
 * This wrapper mirrors the column-major PE layout used by the C model:
 * - `w0`/`w1`/`w2`/`w3` are flattened per-PE weights, `pe = col * ARRAY_ROWS + row`
 * - `a_in` is row activation broadcast across all columns
 * - outputs keep the same per-PE flattening
 *
 * The arithmetic follows `DuoQ.v` exactly:
 * - signed 4-way packing on a single DSP48E2
 * - byte-level correction/carry compensation
 * - one fabric output register (latency = 1)
 */

`timescale 1ns/1ps

(* dont_touch = "1" *)
module DuoQ_4096 #(
    parameter integer ARRAY_ROWS = 128,
    parameter integer ARRAY_COLS = 32,
    parameter integer PE_ADDR_BITS = 12,
    parameter integer ROW_ADDR_BITS = 6,
    parameter integer W_BITS = 4,
    parameter integer A_BITS = 4,
    parameter integer P_BITS = 8,
    parameter integer NUM_PE = ARRAY_ROWS * ARRAY_COLS,
    parameter integer W_BUS_BITS = NUM_PE * W_BITS,
    parameter integer A_BUS_BITS = ARRAY_ROWS * A_BITS,
    parameter integer P_BUS_BITS = NUM_PE * P_BITS
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

    input wire [PE_ADDR_BITS-1:0] w3_address,
    input wire w3_ce,
    input wire [W_BUS_BITS-1:0] w3,

    input wire [ROW_ADDR_BITS-1:0] a_in_address,
    input wire a_in_ce,
    input wire [A_BUS_BITS-1:0] a_in,

    input wire [PE_ADDR_BITS-1:0] p0_address,
    input wire p0_ce,
    input wire p0_we,
    output wire [P_BUS_BITS-1:0] p0,

    input wire [PE_ADDR_BITS-1:0] p1_address,
    input wire p1_ce,
    input wire p1_we,
    output wire [P_BUS_BITS-1:0] p1,

    input wire [PE_ADDR_BITS-1:0] p2_address,
    input wire p2_ce,
    input wire p2_we,
    output wire [P_BUS_BITS-1:0] p2,

    input wire [PE_ADDR_BITS-1:0] p3_address,
    input wire p3_ce,
    input wire p3_we,
    output wire [P_BUS_BITS-1:0] p3
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
            localparam integer P_OFFSET = PE_INDEX * P_BITS;

            wire signed [W_BITS-1:0] w0_local = w0[W_OFFSET +: W_BITS];
            wire signed [W_BITS-1:0] w1_local = w1[W_OFFSET +: W_BITS];
            wire signed [W_BITS-1:0] w2_local = w2[W_OFFSET +: W_BITS];
            wire signed [W_BITS-1:0] w3_local = w3[W_OFFSET +: W_BITS];
            wire signed [A_BITS-1:0] a_in_local = a_in[A_OFFSET +: A_BITS];

            wire signed [P_BITS-1:0] p0_local;
            wire signed [P_BITS-1:0] p1_local;
            wire signed [P_BITS-1:0] p2_local;
            wire signed [P_BITS-1:0] p3_local;

            DuoQ dsp_core (
                .clk(ap_clk),
                .a_in(a_in_local),
                .w0(w0_local),
                .w1(w1_local),
                .w2(w2_local),
                .w3(w3_local),
                .p0(p0_local),
                .p1(p1_local),
                .p2(p2_local),
                .p3(p3_local)
            );

            assign p0[P_OFFSET +: P_BITS] = p0_local;
            assign p1[P_OFFSET +: P_BITS] = p1_local;
            assign p2[P_OFFSET +: P_BITS] = p2_local;
            assign p3[P_OFFSET +: P_BITS] = p3_local;
        end
    end
endgenerate

assign ap_idle = ~(ap_start | dly1);
assign ap_ready = dly1;
assign ap_done = dly1;

endmodule
