`timescale 1ns/1ps

module tb_FP8_E4M3_NoDSP_LUTParallel;
reg clk = 1'b0;
always #5 clk = ~clk;

reg [2:0] w0;
reg [2:0] w1;
reg [2:0] a0;
reg [2:0] a1;
reg [2:0] a2;
reg [2:0] a3;
reg [2:0] a4;
wire [5:0] mag0;
wire [5:0] mag1;
wire [5:0] mag2;
wire [5:0] mag3;
wire [5:0] mag4;
wire [5:0] mag5;
wire [5:0] mag6;
wire [5:0] mag7;
wire [5:0] mag8;
wire [5:0] mag9;
wire [7:0] valid_count;

FP8_E4M3_NoDSP_LUTParallel dut (
    .clk(clk),
    .w0(w0),
    .w1(w1),
    .a0(a0),
    .a1(a1),
    .a2(a2),
    .a3(a3),
    .a4(a4),
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
    .valid_count(valid_count)
);

integer seed;
integer trial;
integer failures;
integer rv;

function [2:0] rand_mag;
    input dummy;
    begin
        rv = $random(seed);
        if (rv < 0) rv = -rv;
        rand_mag = rv & 3'd7;
    end
endfunction

task set_zero;
begin
    w0 = 3'd0;
    w1 = 3'd0;
    a0 = 3'd0;
    a1 = 3'd0;
    a2 = 3'd0;
    a3 = 3'd0;
    a4 = 3'd0;
end
endtask

task set_max;
begin
    w0 = 3'd7;
    w1 = 3'd7;
    a0 = 3'd7;
    a1 = 3'd7;
    a2 = 3'd7;
    a3 = 3'd7;
    a4 = 3'd7;
end
endtask

task set_mixed;
begin
    w0 = 3'd0;
    w1 = 3'd1;
    a0 = 3'd1;
    a1 = 3'd4;
    a2 = 3'd7;
    a3 = 3'd2;
    a4 = 3'd5;
end
endtask

task set_random;
begin
    w0 = rand_mag(1'b0);
    w1 = rand_mag(1'b0);
    a0 = rand_mag(1'b0);
    a1 = rand_mag(1'b0);
    a2 = rand_mag(1'b0);
    a3 = rand_mag(1'b0);
    a4 = rand_mag(1'b0);
end
endtask

task check_current;
input [255:0] case_name;
begin
    repeat (5) @(posedge clk);
    #1;
    if (valid_count !== 8'd10) begin
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
    if (mag5 !== (w1 * a0)) begin
        $display("FAIL %0s mag5 got=%0d exp=%0d", case_name, mag5, (w1 * a0));
        failures = failures + 1;
    end
    if (mag6 !== (w1 * a1)) begin
        $display("FAIL %0s mag6 got=%0d exp=%0d", case_name, mag6, (w1 * a1));
        failures = failures + 1;
    end
    if (mag7 !== (w1 * a2)) begin
        $display("FAIL %0s mag7 got=%0d exp=%0d", case_name, mag7, (w1 * a2));
        failures = failures + 1;
    end
    if (mag8 !== (w1 * a3)) begin
        $display("FAIL %0s mag8 got=%0d exp=%0d", case_name, mag8, (w1 * a3));
        failures = failures + 1;
    end
    if (mag9 !== (w1 * a4)) begin
        $display("FAIL %0s mag9 got=%0d exp=%0d", case_name, mag9, (w1 * a4));
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
        $display("PASS FP8_E4M3_NoDSP_LUTParallel trials=4096 products=10");
    end else begin
        $display("FAIL FP8_E4M3_NoDSP_LUTParallel failures=%0d", failures);
    end
    $finish;
end

endmodule
