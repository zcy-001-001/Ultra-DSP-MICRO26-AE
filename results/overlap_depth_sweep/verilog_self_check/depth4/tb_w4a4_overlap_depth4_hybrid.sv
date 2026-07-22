`timescale 1ns/1ps

module tb_w4a4_overlap_depth4_hybrid;
reg clk = 1'b0;
reg mode = 1'b0;
reg [31:0] seed = 32'h00001309;
reg [31:0] rand_word;
integer cycle = 0;
integer i;
integer failures = 0;

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
wire sign1;
wire sign2;
wire sign3;
wire sign4;
wire sign5;
wire sign6;
wire sign7;
wire sign8;
wire sign9;
wire sign10;
wire sign11;
wire sign12;
wire [4:0] valid_count;

w4a4_overlap_depth4_hybrid dut(
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
    .result12(result12),
    .sign1(sign1),
    .sign2(sign2),
    .sign3(sign3),
    .sign4(sign4),
    .sign5(sign5),
    .sign6(sign6),
    .sign7(sign7),
    .sign8(sign8),
    .sign9(sign9),
    .sign10(sign10),
    .sign11(sign11),
    .sign12(sign12),
    .valid_count(valid_count)
);

wire [83:0] result_bundle;
wire [11:0] sign_bundle;
assign result_bundle[0 +: 7] = result1;
assign sign_bundle[0] = sign1;
assign result_bundle[7 +: 7] = result2;
assign sign_bundle[1] = sign2;
assign result_bundle[14 +: 7] = result3;
assign sign_bundle[2] = sign3;
assign result_bundle[21 +: 7] = result4;
assign sign_bundle[3] = sign4;
assign result_bundle[28 +: 7] = result5;
assign sign_bundle[4] = sign5;
assign result_bundle[35 +: 7] = result6;
assign sign_bundle[5] = sign6;
assign result_bundle[42 +: 7] = result7;
assign sign_bundle[6] = sign7;
assign result_bundle[49 +: 7] = result8;
assign sign_bundle[7] = sign8;
assign result_bundle[56 +: 7] = result9;
assign sign_bundle[8] = sign9;
assign result_bundle[63 +: 7] = result10;
assign sign_bundle[9] = sign10;
assign result_bundle[70 +: 7] = result11;
assign sign_bundle[10] = sign11;
assign result_bundle[77 +: 7] = result12;
assign sign_bundle[11] = sign12;

reg [83:0] next_result;
reg [11:0] next_sign;
reg [4:0] next_valid;
reg [83:0] exp_result_pipe [0:2];
reg [11:0] exp_sign_pipe [0:2];
reg [4:0] exp_valid_pipe [0:2];

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
        encode_product = {1'b0, mag} ^ {7{sign}};
    end
endfunction

task compute_expected;
    begin
        next_result = {84{1'b0}};
        next_sign = {12{1'b0}};
        next_valid = 5'd0;
        if (mode == 1'b0) begin
            next_valid = 5'd12;
            next_result[0 +: 7] = encode_product(w1, a1);
            next_sign[0] = w1[3] ^ a1[3];
            next_result[7 +: 7] = encode_product(w1, a2);
            next_sign[1] = w1[3] ^ a2[3];
            next_result[14 +: 7] = encode_product(w1, a3);
            next_sign[2] = w1[3] ^ a3[3];
            next_result[21 +: 7] = encode_product(w2, a1);
            next_sign[3] = w2[3] ^ a1[3];
            next_result[28 +: 7] = encode_product(w1, a4);
            next_sign[4] = w1[3] ^ a4[3];
            next_result[35 +: 7] = encode_product(w2, a2);
            next_sign[5] = w2[3] ^ a2[3];
            next_result[42 +: 7] = encode_product(w2, a3);
            next_sign[6] = w2[3] ^ a3[3];
            next_result[49 +: 7] = encode_product(w3, a1);
            next_sign[7] = w3[3] ^ a1[3];
            next_result[56 +: 7] = encode_product(w2, a4);
            next_sign[8] = w2[3] ^ a4[3];
            next_result[63 +: 7] = encode_product(w3, a2);
            next_sign[9] = w3[3] ^ a2[3];
            next_result[70 +: 7] = encode_product(w3, a3);
            next_sign[10] = w3[3] ^ a3[3];
            next_result[77 +: 7] = encode_product(w3, a4);
            next_sign[11] = w3[3] ^ a4[3];
        end
        else begin
            if (mode == 1'b1) begin
                next_valid = 5'd7;
                next_result[0 +: 7] = encode_product(w1, a1);
                next_sign[0] = w1[3] ^ a1[3];
                next_result[7 +: 7] = encode_product(w2, a1);
                next_sign[1] = w2[3] ^ a1[3];
                next_result[14 +: 7] = encode_product(w3, a1);
                next_sign[2] = w3[3] ^ a1[3];
                next_result[21 +: 7] = encode_product(w4, a1);
                next_sign[3] = w4[3] ^ a1[3];
                next_result[28 +: 7] = encode_product(w5, a1);
                next_sign[4] = w5[3] ^ a1[3];
                next_result[35 +: 7] = encode_product(w6, a1);
                next_sign[5] = w6[3] ^ a1[3];
                next_result[42 +: 7] = encode_product(w7, a1);
                next_sign[6] = w7[3] ^ a1[3];
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
            if (valid_count !== exp_valid_pipe[2]) begin
                $display("FAIL depth=4 cycle=%0d valid got=%0d expected=%0d", cycle, valid_count, exp_valid_pipe[2]);
                failures = failures + 1;
            end
            if (sign_bundle !== exp_sign_pipe[2]) begin
                $display("FAIL depth=4 cycle=%0d sign got=%b expected=%b", cycle, sign_bundle, exp_sign_pipe[2]);
                failures = failures + 1;
            end
            if (result_bundle !== exp_result_pipe[2]) begin
                $display("FAIL depth=4 cycle=%0d result got=%h expected=%h", cycle, result_bundle, exp_result_pipe[2]);
                $display("  exp_result_pipe[0]=%h", exp_result_pipe[0]);
                $display("  exp_result_pipe[1]=%h", exp_result_pipe[1]);
                $display("  exp_result_pipe[2]=%h", exp_result_pipe[2]);
                $display("  dut.dsp_p=%h", dut.dsp_p);
                failures = failures + 1;
            end
            if (failures > 0) $finish;
        end
    end
endtask

always @(posedge clk) begin
    exp_result_pipe[0] <= next_result;
    exp_sign_pipe[0] <= next_sign;
    exp_valid_pipe[0] <= next_valid;
    for (i = 1; i < 3; i = i + 1) begin
        exp_result_pipe[i] <= exp_result_pipe[i - 1];
        exp_sign_pipe[i] <= exp_sign_pipe[i - 1];
        exp_valid_pipe[i] <= exp_valid_pipe[i - 1];
    end
end

initial begin
    exp_result_pipe[0] = {84{1'b0}};
    exp_result_pipe[1] = {84{1'b0}};
    exp_result_pipe[2] = {84{1'b0}};
    exp_sign_pipe[0] = {12{1'b0}};
    exp_sign_pipe[1] = {12{1'b0}};
    exp_sign_pipe[2] = {12{1'b0}};
    exp_valid_pipe[0] = 5'd0;
    exp_valid_pipe[1] = 5'd0;
    exp_valid_pipe[2] = 5'd0;
    drive_random();
    for (cycle = 0; cycle < 71; cycle = cycle + 1) begin
        @(negedge clk);
        check_outputs();
        if (cycle < 64) drive_random();
        else drive_zero();
    end
    if (failures == 0) begin
        $display("PASS depth=4 top=w4a4_overlap_depth4_hybrid trials=64");
    end
    $finish;
end

endmodule
