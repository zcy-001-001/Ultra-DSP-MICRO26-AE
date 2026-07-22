`timescale 1ns/1ps

module tb_FP8_E4M3_NoPackingScalar;
reg clk = 1'b0;
always #5 clk = ~clk;

reg [2:0] w0;
reg [2:0] a0;
wire [5:0] mag0;
wire [7:0] valid_count;

FP8_E4M3_NoPackingScalar dut (
    .clk(clk),
    .w0(w0),
    .a0(a0),
    .mag0(mag0),
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
    a0 = 3'd0;
end
endtask

task set_max;
begin
    w0 = 3'd7;
    a0 = 3'd7;
end
endtask

task set_mixed;
begin
    w0 = 3'd0;
    a0 = 3'd1;
end
endtask

task set_random;
begin
    w0 = rand_mag(1'b0);
    a0 = rand_mag(1'b0);
end
endtask

task check_current;
input [255:0] case_name;
begin
    repeat (5) @(posedge clk);
    #1;
    if (valid_count !== 8'd1) begin
        $display("FAIL %0s valid_count got=%0d", case_name, valid_count);
        failures = failures + 1;
    end
    if (mag0 !== (w0 * a0)) begin
        $display("FAIL %0s mag0 got=%0d exp=%0d", case_name, mag0, (w0 * a0));
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
        $display("PASS FP8_E4M3_NoPackingScalar trials=4096 products=1");
    end else begin
        $display("FAIL FP8_E4M3_NoPackingScalar failures=%0d", failures);
    end
    $finish;
end

endmodule
