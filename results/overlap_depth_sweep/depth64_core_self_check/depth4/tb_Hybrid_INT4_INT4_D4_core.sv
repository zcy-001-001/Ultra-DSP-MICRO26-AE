`timescale 1ns/1ps

module tb_Hybrid_INT4_INT4_D4_core;
reg clk = 1'b0;
reg mode = 1'b0;
reg [3:0] w1;
reg [3:0] w2;
reg [3:0] w3;
reg [3:0] w4;
reg [3:0] w5;
reg [3:0] w6;
reg [3:0] w7;
reg [3:0] a1;
reg [3:0] a2;
reg [3:0] a3;
reg [3:0] a4;
wire [6:0] result1;
wire [6:0] result2;
wire [6:0] result3;
wire [6:0] result4;
wire [6:0] result5;
wire [6:0] result6;
wire [6:0] result7;
wire [6:0] result8;
wire [6:0] result9;
wire [6:0] result10;
wire [6:0] result11;
wire [6:0] result12;
reg [31:0] seed = 32'h00000d68;
reg [31:0] rand_word;
integer cycle = 0;
integer i;
integer failures = 0;

Hybrid_INT4_INT4_D4_core dut(
    .clk(clk),
    .mode(mode),
    .w1(w1),
    .w2(w2),
    .w3(w3),
    .w4(w4),
    .w5(w5),
    .w6(w6),
    .w7(w7),
    .a1(a1),
    .a2(a2),
    .a3(a3),
    .a4(a4),
    .result1(result1),
    .result2(result2),
    .result3(result3),
    .result4(result4),
    .result5(result5),
    .result6(result6),
    .result7(result7),
    .result8(result8),
    .result9(result9),
    .result10(result10),
    .result11(result11),
    .result12(result12)
);

wire [83:0] result_bundle;
assign result_bundle[0 +: 7] = result1;
assign result_bundle[7 +: 7] = result2;
assign result_bundle[14 +: 7] = result3;
assign result_bundle[21 +: 7] = result4;
assign result_bundle[28 +: 7] = result5;
assign result_bundle[35 +: 7] = result6;
assign result_bundle[42 +: 7] = result7;
assign result_bundle[49 +: 7] = result8;
assign result_bundle[56 +: 7] = result9;
assign result_bundle[63 +: 7] = result10;
assign result_bundle[70 +: 7] = result11;
assign result_bundle[77 +: 7] = result12;

reg [83:0] next_result;
reg [83:0] exp_result_pipe [0:2];

always #5 clk = ~clk;

function [2:0] to_mag3;
    input [3:0] x;
    reg [3:0] temp;
    begin
        temp = x[3] ? (~x + 1'b1) : x;
        to_mag3 = (temp == 4'b1000) ? 3'b111 : temp[2:0];
    end
endfunction

function [6:0] encode_product;
    input [3:0] w;
    input [3:0] a;
    reg [5:0] mag;
    reg sign;
    begin
        mag = w[2:0] * to_mag3(a);
        sign = w[3] ^ a[3];
        encode_product = sign ? -$signed({1'b0, mag}) : {1'b0, mag};
    end
endfunction

task compute_expected;
    begin
        next_result = {84{1'b0}};
        if (mode == 1'b0) begin
            next_result[0 +: 7] = encode_product(w1, a1);
            next_result[7 +: 7] = encode_product(w1, a2);
            next_result[14 +: 7] = encode_product(w1, a3);
            next_result[21 +: 7] = encode_product(w2, a1);
            next_result[28 +: 7] = encode_product(w1, a4);
            next_result[35 +: 7] = encode_product(w2, a2);
            next_result[42 +: 7] = encode_product(w2, a3);
            next_result[49 +: 7] = encode_product(w3, a1);
            next_result[56 +: 7] = encode_product(w2, a4);
            next_result[63 +: 7] = encode_product(w3, a2);
            next_result[70 +: 7] = encode_product(w3, a3);
            next_result[77 +: 7] = encode_product(w3, a4);
        end
        else begin
            if (mode == 1'b1) begin
                next_result[0 +: 7] = encode_product(w1, a1);
                next_result[7 +: 7] = encode_product(w2, a1);
                next_result[14 +: 7] = encode_product(w3, a1);
                next_result[21 +: 7] = encode_product(w4, a1);
                next_result[28 +: 7] = encode_product(w5, a1);
                next_result[35 +: 7] = encode_product(w6, a1);
                next_result[42 +: 7] = encode_product(w7, a1);
                next_result[49 +: 7] = 7'b0;
                next_result[56 +: 7] = 7'b0;
                next_result[63 +: 7] = 7'b0;
                next_result[70 +: 7] = 7'b0;
                next_result[77 +: 7] = 7'b0;
            end
        end
    end
endtask

task drive_random;
    begin
        rand_word = $urandom(seed);
        w1 = {rand_word[0], rand_word[3:1]};
        rand_word = $urandom(seed);
        w2 = {rand_word[0], rand_word[3:1]};
        rand_word = $urandom(seed);
        w3 = {rand_word[0], rand_word[3:1]};
        rand_word = $urandom(seed);
        w4 = {rand_word[0], rand_word[3:1]};
        rand_word = $urandom(seed);
        w5 = {rand_word[0], rand_word[3:1]};
        rand_word = $urandom(seed);
        w6 = {rand_word[0], rand_word[3:1]};
        rand_word = $urandom(seed);
        w7 = {rand_word[0], rand_word[3:1]};
        rand_word = $urandom(seed);
        a1 = rand_word[3:0];
        rand_word = $urandom(seed);
        a2 = rand_word[3:0];
        rand_word = $urandom(seed);
        a3 = rand_word[3:0];
        rand_word = $urandom(seed);
        a4 = rand_word[3:0];
        rand_word = $urandom(seed);
        mode = rand_word[0];
        compute_expected();
    end
endtask

task drive_zero;
    begin
        w1 = 4'b0;
        w2 = 4'b0;
        w3 = 4'b0;
        w4 = 4'b0;
        w5 = 4'b0;
        w6 = 4'b0;
        w7 = 4'b0;
        a1 = 4'b0;
        a2 = 4'b0;
        a3 = 4'b0;
        a4 = 4'b0;
        mode = 1'b0;
        compute_expected();
    end
endtask

task check_outputs;
    begin
        if (cycle >= 16) begin
            if (result_bundle !== exp_result_pipe[2]) begin
                $display("FAIL depth=4 cycle=%0d result got=%h expected=%h", cycle, result_bundle, exp_result_pipe[2]);
                $display("  dut.dsp_p=%h", dut.dsp_p);
                failures = failures + 1;
            end
            if (failures > 0) $finish;
        end
    end
endtask

always @(posedge clk) begin
    exp_result_pipe[0] <= next_result;
    for (i = 1; i < 3; i = i + 1) begin
        exp_result_pipe[i] <= exp_result_pipe[i - 1];
    end
end

initial begin
    exp_result_pipe[0] = {84{1'b0}};
    exp_result_pipe[1] = {84{1'b0}};
    exp_result_pipe[2] = {84{1'b0}};
    drive_random();
    for (cycle = 0; cycle < 71; cycle = cycle + 1) begin
        @(negedge clk);
        check_outputs();
        if (cycle < 64) drive_random();
        else drive_zero();
    end
    if (failures == 0) begin
        $display("PASS depth=4 core=Hybrid_INT4_INT4_D4_core trials=64");
    end
    $finish;
end

endmodule
