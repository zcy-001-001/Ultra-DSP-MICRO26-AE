`timescale 1ns/1ps

(* dont_touch = "1" *)
module W4A5_P #(
    parameter integer ARRAY_ROWS = 64,
    parameter integer ARRAY_COLS = 64,
    parameter integer PE_ADDR_BITS = 12,
    parameter integer ROW_ADDR_BITS = 6,
    parameter integer W_BITS = 4,
    parameter integer A_BITS = 5,
    parameter integer R_BITS = 8,
    parameter integer NUM_PE = ARRAY_ROWS * ARRAY_COLS,
    parameter integer W_BUS_BITS = NUM_PE * W_BITS,
    parameter integer A_BUS_BITS = ARRAY_ROWS * A_BITS,
    parameter integer R_BUS_BITS = NUM_PE * R_BITS,
    parameter integer S_BUS_BITS = NUM_PE
)(
    input wire ap_clk,
    input wire ap_rst,
    input wire ap_ce,
    input wire ap_start,
    input wire ap_continue,
    output wire ap_done,
    output wire ap_idle,
    output wire ap_ready,

    input wire [PE_ADDR_BITS-1:0] w1_address,
    input wire w1_ce,
    input wire [W_BUS_BITS-1:0] w1,

    input wire [PE_ADDR_BITS-1:0] w2_address,
    input wire w2_ce,
    input wire [W_BUS_BITS-1:0] w2,

    input wire [PE_ADDR_BITS-1:0] w3_address,
    input wire w3_ce,
    input wire [W_BUS_BITS-1:0] w3,

    input wire [PE_ADDR_BITS-1:0] w4_address,
    input wire w4_ce,
    input wire [W_BUS_BITS-1:0] w4,

    input wire [ROW_ADDR_BITS-1:0] a1_address,
    input wire a1_ce,
    input wire [A_BUS_BITS-1:0] a1,

    input wire [ROW_ADDR_BITS-1:0] a2_address,
    input wire a2_ce,
    input wire [A_BUS_BITS-1:0] a2,

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
    output wire [R_BUS_BITS-1:0] result5,

    input wire [PE_ADDR_BITS-1:0] result6_address,
    input wire result6_ce,
    input wire result6_we,
    output wire [R_BUS_BITS-1:0] result6,

    input wire [PE_ADDR_BITS-1:0] result7_address,
    input wire result7_ce,
    input wire result7_we,
    output wire [R_BUS_BITS-1:0] result7,

    input wire [PE_ADDR_BITS-1:0] result8_address,
    input wire result8_ce,
    input wire result8_we,
    output wire [R_BUS_BITS-1:0] result8,

    input wire [PE_ADDR_BITS-1:0] sign1_address,
    input wire sign1_ce,
    input wire sign1_we,
    output wire [S_BUS_BITS-1:0] sign1,

    input wire [PE_ADDR_BITS-1:0] sign2_address,
    input wire sign2_ce,
    input wire sign2_we,
    output wire [S_BUS_BITS-1:0] sign2,

    input wire [PE_ADDR_BITS-1:0] sign3_address,
    input wire sign3_ce,
    input wire sign3_we,
    output wire [S_BUS_BITS-1:0] sign3,

    input wire [PE_ADDR_BITS-1:0] sign4_address,
    input wire sign4_ce,
    input wire sign4_we,
    output wire [S_BUS_BITS-1:0] sign4,

    input wire [PE_ADDR_BITS-1:0] sign5_address,
    input wire sign5_ce,
    input wire sign5_we,
    output wire [S_BUS_BITS-1:0] sign5,

    input wire [PE_ADDR_BITS-1:0] sign6_address,
    input wire sign6_ce,
    input wire sign6_we,
    output wire [S_BUS_BITS-1:0] sign6,

    input wire [PE_ADDR_BITS-1:0] sign7_address,
    input wire sign7_ce,
    input wire sign7_we,
    output wire [S_BUS_BITS-1:0] sign7,

    input wire [PE_ADDR_BITS-1:0] sign8_address,
    input wire sign8_ce,
    input wire sign8_we,
    output wire [S_BUS_BITS-1:0] sign8
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

            wire [4:0] a1_local = a1[A_OFFSET +: A_BITS];
            wire [4:0] a2_local = a2[A_OFFSET +: A_BITS];

            wire [7:0] result1_local;
            wire [7:0] result2_local;
            wire [7:0] result3_local;
            wire [7:0] result4_local;
            wire [7:0] result5_local;
            wire [7:0] result6_local;
            wire [7:0] result7_local;
            wire [7:0] result8_local;
            wire sign1_local;
            wire sign2_local;
            wire sign3_local;
            wire sign4_local;
            wire sign5_local;
            wire sign6_local;
            wire sign7_local;
            wire sign8_local;

            INT5_INT4_P dsp_core (
                .clk(ap_clk),
                .a1(a1_local),
                .a2(a2_local),
                .w1(w1_local),
                .w2(w2_local),
                .w3(w3_local),
                .w4(w4_local),
                .result1(result1_local),
                .result2(result2_local),
                .result3(result3_local),
                .result4(result4_local),
                .result5(result5_local),
                .result6(result6_local),
                .result7(result7_local),
                .result8(result8_local),
                .sign1(sign1_local),
                .sign2(sign2_local),
                .sign3(sign3_local),
                .sign4(sign4_local),
                .sign5(sign5_local),
                .sign6(sign6_local),
                .sign7(sign7_local),
                .sign8(sign8_local)
            );

            assign result1[R_OFFSET +: R_BITS] = result1_local;
            assign result2[R_OFFSET +: R_BITS] = result2_local;
            assign result3[R_OFFSET +: R_BITS] = result3_local;
            assign result4[R_OFFSET +: R_BITS] = result4_local;
            assign result5[R_OFFSET +: R_BITS] = result5_local;
            assign result6[R_OFFSET +: R_BITS] = result6_local;
            assign result7[R_OFFSET +: R_BITS] = result7_local;
            assign result8[R_OFFSET +: R_BITS] = result8_local;
            assign sign1[PE_INDEX] = sign1_local;
            assign sign2[PE_INDEX] = sign2_local;
            assign sign3[PE_INDEX] = sign3_local;
            assign sign4[PE_INDEX] = sign4_local;
            assign sign5[PE_INDEX] = sign5_local;
            assign sign6[PE_INDEX] = sign6_local;
            assign sign7[PE_INDEX] = sign7_local;
            assign sign8[PE_INDEX] = sign8_local;
        end
    end
endgenerate

assign ap_idle = ~(ap_start | dly1 | dly2 | dly3);
assign ap_ready = dly3;
assign ap_done = dly3;

endmodule
