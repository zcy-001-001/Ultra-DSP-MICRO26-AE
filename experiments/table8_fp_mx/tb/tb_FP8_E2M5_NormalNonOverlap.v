`timescale 1ns/1ps

module tb_FP8_E2M5_NormalNonOverlap;
reg clk = 1'b0;
always #5 clk = ~clk;

reg [4:0] w0;
reg [4:0] w1;
reg [4:0] a0;
reg [4:0] a1;
wire [9:0] mag0;
wire [9:0] mag1;
wire [9:0] mag2;
wire [9:0] mag3;
wire [7:0] valid_count;

FP8_E2M5_NormalNonOverlap dut (
    .clk(clk),
    .w0(w0),
    .w1(w1),
    .a0(a0),
    .a1(a1),
    .mag0(mag0),
    .mag1(mag1),
    .mag2(mag2),
    .mag3(mag3),
    .valid_count(valid_count)
);

integer seed;
integer trial;
integer failures;
integer rv;

function [4:0] rand_mag;
    input dummy;
    begin
        rv = $random(seed);
        if (rv < 0) rv = -rv;
        rand_mag = rv & 5'd31;
    end
endfunction

task set_zero;
begin
    w0 = 5'd0;
    w1 = 5'd0;
    a0 = 5'd0;
    a1 = 5'd0;
end
endtask

task set_max;
begin
    w0 = 5'd31;
    w1 = 5'd31;
    a0 = 5'd31;
    a1 = 5'd31;
end
endtask

task set_mixed;
begin
    w0 = 5'd0;
    w1 = 5'd1;
    a0 = 5'd1;
    a1 = 5'd4;
end
endtask

task set_random;
begin
    w0 = rand_mag(1'b0);
    w1 = rand_mag(1'b0);
    a0 = rand_mag(1'b0);
    a1 = rand_mag(1'b0);
end
endtask

task check_current;
input [255:0] case_name;
begin
    repeat (5) @(posedge clk);
    #1;
    if (valid_count !== 8'd4) begin
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
    if (mag2 !== (w1 * a0)) begin
        $display("FAIL %0s mag2 got=%0d exp=%0d", case_name, mag2, (w1 * a0));
        failures = failures + 1;
    end
    if (mag3 !== (w1 * a1)) begin
        $display("FAIL %0s mag3 got=%0d exp=%0d", case_name, mag3, (w1 * a1));
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
        $display("PASS FP8_E2M5_NormalNonOverlap trials=4096 products=4");
    end else begin
        $display("FAIL FP8_E2M5_NormalNonOverlap failures=%0d", failures);
    end
    $finish;
end

endmodule
