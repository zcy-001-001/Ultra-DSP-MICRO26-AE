/**
 * @module DEEPBURNING_4096
 * @brief 64x64 DSP array wrapper for the DeepBurning packed INT4 kernel.
 *
 * Interface mapping:
 * - w1 and w2 are per-PE weights, flattened in column-major order:
 *   pe_index = col * ARRAY_ROWS + row
 * - a1, a2, a3 are row vectors shared across all columns
 * - result1~result6 keep one result per PE using the same flattening
 */

`timescale 1ns/1ps

(* dont_touch = "1" *)
module DEEPBURNING_4096 #(
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
    input wire        ap_clk,
    input wire        ap_rst,
    input wire        ap_ce,
    input wire        ap_start,
    input wire        ap_continue,
    output wire       ap_done,
    output wire       ap_idle,
    output wire       ap_ready,

    input wire [PE_ADDR_BITS-1:0] w1_address,
    input wire                    w1_ce,
    input wire [W_BUS_BITS-1:0]   w1,

    input wire [PE_ADDR_BITS-1:0] w2_address,
    input wire                    w2_ce,
    input wire [W_BUS_BITS-1:0]   w2,

    input wire [ROW_ADDR_BITS-1:0] a1_address,
    input wire                     a1_ce,
    input wire [A_BUS_BITS-1:0]    a1,

    input wire [ROW_ADDR_BITS-1:0] a2_address,
    input wire                     a2_ce,
    input wire [A_BUS_BITS-1:0]    a2,

    input wire [ROW_ADDR_BITS-1:0] a3_address,
    input wire                     a3_ce,
    input wire [A_BUS_BITS-1:0]    a3,

    input wire [PE_ADDR_BITS-1:0] result1_address,
    input wire                    result1_ce,
    input wire                    result1_we,
    output wire [R_BUS_BITS-1:0]  result1,

    input wire [PE_ADDR_BITS-1:0] result2_address,
    input wire                    result2_ce,
    input wire                    result2_we,
    output wire [R_BUS_BITS-1:0]  result2,

    input wire [PE_ADDR_BITS-1:0] result3_address,
    input wire                    result3_ce,
    input wire                    result3_we,
    output wire [R_BUS_BITS-1:0]  result3,

    input wire [PE_ADDR_BITS-1:0] result4_address,
    input wire                    result4_ce,
    input wire                    result4_we,
    output wire [R_BUS_BITS-1:0]  result4,

    input wire [PE_ADDR_BITS-1:0] result5_address,
    input wire                    result5_ce,
    input wire                    result5_we,
    output wire [R_BUS_BITS-1:0]  result5,

    input wire [PE_ADDR_BITS-1:0] result6_address,
    input wire                    result6_ce,
    input wire                    result6_we,
    output wire [R_BUS_BITS-1:0]  result6
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

            wire [3:0] w1_local = w1[W_OFFSET +: W_BITS];
            wire [3:0] w2_local = w2[W_OFFSET +: W_BITS];
            wire [3:0] a1_local = a1[A_OFFSET +: A_BITS];
            wire [3:0] a2_local = a2[A_OFFSET +: A_BITS];
            wire [3:0] a3_local = a3[A_OFFSET +: A_BITS];

            wire [7:0] result1_local;
            wire [7:0] result2_local;
            wire [7:0] result3_local;
            wire [7:0] result4_local;
            wire [7:0] result5_local;
            wire [7:0] result6_local;

            INT4_INT4_DEEPBURNING6 dsp_core (
                .clk(ap_clk),
                .w1(w1_local),
                .w2(w2_local),
                .a1(a1_local),
                .a2(a2_local),
                .a3(a3_local),
                .result1(result1_local),
                .result2(result2_local),
                .result3(result3_local),
                .result4(result4_local),
                .result5(result5_local),
                .result6(result6_local)
            );

            assign result1[R_OFFSET +: R_BITS] = result1_local;
            assign result2[R_OFFSET +: R_BITS] = result2_local;
            assign result3[R_OFFSET +: R_BITS] = result3_local;
            assign result4[R_OFFSET +: R_BITS] = result4_local;
            assign result5[R_OFFSET +: R_BITS] = result5_local;
            assign result6[R_OFFSET +: R_BITS] = result6_local;
        end
    end
endgenerate

assign ap_idle = ~(ap_start | dly1);
assign ap_ready = dly1;
assign ap_done = dly1;

endmodule
