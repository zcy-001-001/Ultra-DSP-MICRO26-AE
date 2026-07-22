`timescale 1ns/1ps

module tb_FP8_E5M2_UltraDSP_Packing;
reg clk = 1'b0;
always #5 clk = ~clk;

reg [1:0] w0;
reg [1:0] w1;
reg [1:0] w2;
reg [1:0] a0;
reg [1:0] a1;
reg [1:0] a2;
reg [1:0] a3;
reg [1:0] a4;
reg [1:0] a5;
wire [3:0] mag0;
wire [3:0] mag1;
wire [3:0] mag2;
wire [3:0] mag3;
wire [3:0] mag4;
wire [3:0] mag5;
wire [3:0] mag6;
wire [3:0] mag7;
wire [3:0] mag8;
wire [3:0] mag9;
wire [3:0] mag10;
wire [3:0] mag11;
wire [3:0] mag12;
wire [3:0] mag13;
wire [3:0] mag14;
wire [3:0] mag15;
wire [3:0] mag16;
wire [3:0] mag17;
wire [7:0] valid_count;

FP8_E5M2_UltraDSP_Packing dut (
    .clk(clk),
    .w0(w0),
    .w1(w1),
    .w2(w2),
    .a0(a0),
    .a1(a1),
    .a2(a2),
    .a3(a3),
    .a4(a4),
    .a5(a5),
    .mag0(mag0),
    .mag1(mag1),
    .mag2(mag2),
    .mag3(mag3),
    .mag4(mag4),
    .mag5(mag5),
    .mag6(mag6),
    .mag7(mag7),
    .mag8(mag8),
    .mag9(mag9),
    .mag10(mag10),
    .mag11(mag11),
    .mag12(mag12),
    .mag13(mag13),
    .mag14(mag14),
    .mag15(mag15),
    .mag16(mag16),
    .mag17(mag17),
    .valid_count(valid_count)
);

integer seed;
integer trial;
integer failures;
integer rv;

function [1:0] rand_mag;
    input dummy;
    begin
        rv = $random(seed);
        if (rv < 0) rv = -rv;
        rand_mag = rv & 2'd3;
    end
endfunction

task set_zero;
begin
    w0 = 2'd0;
    w1 = 2'd0;
    w2 = 2'd0;
    a0 = 2'd0;
    a1 = 2'd0;
    a2 = 2'd0;
    a3 = 2'd0;
    a4 = 2'd0;
    a5 = 2'd0;
end
endtask

task set_max;
begin
    w0 = 2'd3;
    w1 = 2'd3;
    w2 = 2'd3;
    a0 = 2'd3;
    a1 = 2'd3;
    a2 = 2'd3;
    a3 = 2'd3;
    a4 = 2'd3;
    a5 = 2'd3;
end
endtask

task set_mixed;
begin
    w0 = 2'd0;
    w1 = 2'd1;
    w2 = 2'd2;
    a0 = 2'd1;
    a1 = 2'd0;
    a2 = 2'd3;
    a3 = 2'd2;
    a4 = 2'd1;
    a5 = 2'd0;
end
endtask

task set_random;
begin
    w0 = rand_mag(1'b0);
    w1 = rand_mag(1'b0);
    w2 = rand_mag(1'b0);
    a0 = rand_mag(1'b0);
    a1 = rand_mag(1'b0);
    a2 = rand_mag(1'b0);
    a3 = rand_mag(1'b0);
    a4 = rand_mag(1'b0);
    a5 = rand_mag(1'b0);
end
endtask

task check_current;
input [255:0] case_name;
begin
    repeat (5) @(posedge clk);
    #1;
    if (valid_count !== 8'd18) begin
        $display("FAIL %0s valid_count got=%0d", case_name, valid_count);
        failures = failures + 1;
    end
    if (mag0 !== (w0 * a0)) begin
        $display("FAIL %0s mag0 got=%0d exp=%0d", case_name, mag0, (w0 * a0));
        failures = failures + 1;
    end
    if (mag1 !== (w0 * a1)) begin
        $display("FAIL %0s mag1 got=%0d exp=%0d", case_name, mag1, (w0 * a1));
        failures = failures + 1;
    end
    if (mag2 !== (w0 * a2)) begin
        $display("FAIL %0s mag2 got=%0d exp=%0d", case_name, mag2, (w0 * a2));
        failures = failures + 1;
    end
    if (mag3 !== (w0 * a3)) begin
        $display("FAIL %0s mag3 got=%0d exp=%0d", case_name, mag3, (w0 * a3));
        failures = failures + 1;
    end
    if (mag4 !== (w0 * a4)) begin
        $display("FAIL %0s mag4 got=%0d exp=%0d", case_name, mag4, (w0 * a4));
        failures = failures + 1;
    end
    if (mag5 !== (w0 * a5)) begin
        $display("FAIL %0s mag5 got=%0d exp=%0d", case_name, mag5, (w0 * a5));
        failures = failures + 1;
    end
    if (mag6 !== (w1 * a0)) begin
        $display("FAIL %0s mag6 got=%0d exp=%0d", case_name, mag6, (w1 * a0));
        failures = failures + 1;
    end
    if (mag7 !== (w1 * a1)) begin
        $display("FAIL %0s mag7 got=%0d exp=%0d", case_name, mag7, (w1 * a1));
        failures = failures + 1;
    end
    if (mag8 !== (w1 * a2)) begin
        $display("FAIL %0s mag8 got=%0d exp=%0d", case_name, mag8, (w1 * a2));
        failures = failures + 1;
    end
    if (mag9 !== (w1 * a3)) begin
        $display("FAIL %0s mag9 got=%0d exp=%0d", case_name, mag9, (w1 * a3));
        failures = failures + 1;
    end
    if (mag10 !== (w1 * a4)) begin
        $display("FAIL %0s mag10 got=%0d exp=%0d", case_name, mag10, (w1 * a4));
        failures = failures + 1;
    end
    if (mag11 !== (w1 * a5)) begin
        $display("FAIL %0s mag11 got=%0d exp=%0d", case_name, mag11, (w1 * a5));
        failures = failures + 1;
    end
    if (mag12 !== (w2 * a0)) begin
        $display("FAIL %0s mag12 got=%0d exp=%0d", case_name, mag12, (w2 * a0));
        failures = failures + 1;
    end
    if (mag13 !== (w2 * a1)) begin
        $display("FAIL %0s mag13 got=%0d exp=%0d", case_name, mag13, (w2 * a1));
        failures = failures + 1;
    end
    if (mag14 !== (w2 * a2)) begin
        $display("FAIL %0s mag14 got=%0d exp=%0d", case_name, mag14, (w2 * a2));
        failures = failures + 1;
    end
    if (mag15 !== (w2 * a3)) begin
        $display("FAIL %0s mag15 got=%0d exp=%0d", case_name, mag15, (w2 * a3));
        failures = failures + 1;
    end
    if (mag16 !== (w2 * a4)) begin
        $display("FAIL %0s mag16 got=%0d exp=%0d", case_name, mag16, (w2 * a4));
        failures = failures + 1;
    end
    if (mag17 !== (w2 * a5)) begin
        $display("FAIL %0s mag17 got=%0d exp=%0d", case_name, mag17, (w2 * a5));
        failures = failures + 1;
    end
end
endtask

initial begin
    seed = 1305;
    failures = 0;
    set_zero(); check_current("zero");
    set_max(); check_current("max");
    set_mixed(); check_current("mixed");
    for (trial = 0; trial < 4096; trial = trial + 1) begin
        set_random();
        check_current("random");
    end
    if (failures == 0) begin
        $display("PASS FP8_E5M2_UltraDSP_Packing trials=4096 products=18");
    end else begin
        $display("FAIL FP8_E5M2_UltraDSP_Packing failures=%0d", failures);
    end
    $finish;
end

endmodule
