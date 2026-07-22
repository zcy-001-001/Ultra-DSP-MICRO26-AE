`timescale 1ns/1ps

module tb_MXFP4_E2M1_MAG1_UltraDSP_Packing;
reg clk = 1'b0;
always #5 clk = ~clk;

reg [0:0] w0;
reg [0:0] w1;
reg [0:0] w2;
reg [0:0] a0;
reg [0:0] a1;
reg [0:0] a2;
reg [0:0] a3;
reg [0:0] a4;
reg [0:0] a5;
reg [0:0] a6;
reg [0:0] a7;
reg [0:0] a8;
reg [0:0] a9;
reg [0:0] a10;
reg [0:0] a11;
wire [1:0] mag0;
wire [1:0] mag1;
wire [1:0] mag2;
wire [1:0] mag3;
wire [1:0] mag4;
wire [1:0] mag5;
wire [1:0] mag6;
wire [1:0] mag7;
wire [1:0] mag8;
wire [1:0] mag9;
wire [1:0] mag10;
wire [1:0] mag11;
wire [1:0] mag12;
wire [1:0] mag13;
wire [1:0] mag14;
wire [1:0] mag15;
wire [1:0] mag16;
wire [1:0] mag17;
wire [1:0] mag18;
wire [1:0] mag19;
wire [1:0] mag20;
wire [1:0] mag21;
wire [1:0] mag22;
wire [1:0] mag23;
wire [1:0] mag24;
wire [1:0] mag25;
wire [1:0] mag26;
wire [1:0] mag27;
wire [1:0] mag28;
wire [1:0] mag29;
wire [1:0] mag30;
wire [1:0] mag31;
wire [1:0] mag32;
wire [1:0] mag33;
wire [1:0] mag34;
wire [1:0] mag35;
wire [7:0] valid_count;

MXFP4_E2M1_MAG1_UltraDSP_Packing dut (
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
    .a6(a6),
    .a7(a7),
    .a8(a8),
    .a9(a9),
    .a10(a10),
    .a11(a11),
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
    .mag18(mag18),
    .mag19(mag19),
    .mag20(mag20),
    .mag21(mag21),
    .mag22(mag22),
    .mag23(mag23),
    .mag24(mag24),
    .mag25(mag25),
    .mag26(mag26),
    .mag27(mag27),
    .mag28(mag28),
    .mag29(mag29),
    .mag30(mag30),
    .mag31(mag31),
    .mag32(mag32),
    .mag33(mag33),
    .mag34(mag34),
    .mag35(mag35),
    .valid_count(valid_count)
);

integer seed;
integer trial;
integer failures;
integer rv;

function [0:0] rand_mag;
    input dummy;
    begin
        rv = $random(seed);
        if (rv < 0) rv = -rv;
        rand_mag = rv & 1'd1;
    end
endfunction

task set_zero;
begin
    w0 = 1'd0;
    w1 = 1'd0;
    w2 = 1'd0;
    a0 = 1'd0;
    a1 = 1'd0;
    a2 = 1'd0;
    a3 = 1'd0;
    a4 = 1'd0;
    a5 = 1'd0;
    a6 = 1'd0;
    a7 = 1'd0;
    a8 = 1'd0;
    a9 = 1'd0;
    a10 = 1'd0;
    a11 = 1'd0;
end
endtask

task set_max;
begin
    w0 = 1'd1;
    w1 = 1'd1;
    w2 = 1'd1;
    a0 = 1'd1;
    a1 = 1'd1;
    a2 = 1'd1;
    a3 = 1'd1;
    a4 = 1'd1;
    a5 = 1'd1;
    a6 = 1'd1;
    a7 = 1'd1;
    a8 = 1'd1;
    a9 = 1'd1;
    a10 = 1'd1;
    a11 = 1'd1;
end
endtask

task set_mixed;
begin
    w0 = 1'd0;
    w1 = 1'd1;
    w2 = 1'd0;
    a0 = 1'd1;
    a1 = 1'd0;
    a2 = 1'd1;
    a3 = 1'd0;
    a4 = 1'd1;
    a5 = 1'd0;
    a6 = 1'd1;
    a7 = 1'd0;
    a8 = 1'd1;
    a9 = 1'd0;
    a10 = 1'd1;
    a11 = 1'd0;
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
    a6 = rand_mag(1'b0);
    a7 = rand_mag(1'b0);
    a8 = rand_mag(1'b0);
    a9 = rand_mag(1'b0);
    a10 = rand_mag(1'b0);
    a11 = rand_mag(1'b0);
end
endtask

task check_current;
input [255:0] case_name;
begin
    repeat (5) @(posedge clk);
    #1;
    if (valid_count !== 8'd36) begin
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
    if (mag6 !== (w0 * a6)) begin
        $display("FAIL %0s mag6 got=%0d exp=%0d", case_name, mag6, (w0 * a6));
        failures = failures + 1;
    end
    if (mag7 !== (w0 * a7)) begin
        $display("FAIL %0s mag7 got=%0d exp=%0d", case_name, mag7, (w0 * a7));
        failures = failures + 1;
    end
    if (mag8 !== (w0 * a8)) begin
        $display("FAIL %0s mag8 got=%0d exp=%0d", case_name, mag8, (w0 * a8));
        failures = failures + 1;
    end
    if (mag9 !== (w0 * a9)) begin
        $display("FAIL %0s mag9 got=%0d exp=%0d", case_name, mag9, (w0 * a9));
        failures = failures + 1;
    end
    if (mag10 !== (w0 * a10)) begin
        $display("FAIL %0s mag10 got=%0d exp=%0d", case_name, mag10, (w0 * a10));
        failures = failures + 1;
    end
    if (mag11 !== (w0 * a11)) begin
        $display("FAIL %0s mag11 got=%0d exp=%0d", case_name, mag11, (w0 * a11));
        failures = failures + 1;
    end
    if (mag12 !== (w1 * a0)) begin
        $display("FAIL %0s mag12 got=%0d exp=%0d", case_name, mag12, (w1 * a0));
        failures = failures + 1;
    end
    if (mag13 !== (w1 * a1)) begin
        $display("FAIL %0s mag13 got=%0d exp=%0d", case_name, mag13, (w1 * a1));
        failures = failures + 1;
    end
    if (mag14 !== (w1 * a2)) begin
        $display("FAIL %0s mag14 got=%0d exp=%0d", case_name, mag14, (w1 * a2));
        failures = failures + 1;
    end
    if (mag15 !== (w1 * a3)) begin
        $display("FAIL %0s mag15 got=%0d exp=%0d", case_name, mag15, (w1 * a3));
        failures = failures + 1;
    end
    if (mag16 !== (w1 * a4)) begin
        $display("FAIL %0s mag16 got=%0d exp=%0d", case_name, mag16, (w1 * a4));
        failures = failures + 1;
    end
    if (mag17 !== (w1 * a5)) begin
        $display("FAIL %0s mag17 got=%0d exp=%0d", case_name, mag17, (w1 * a5));
        failures = failures + 1;
    end
    if (mag18 !== (w1 * a6)) begin
        $display("FAIL %0s mag18 got=%0d exp=%0d", case_name, mag18, (w1 * a6));
        failures = failures + 1;
    end
    if (mag19 !== (w1 * a7)) begin
        $display("FAIL %0s mag19 got=%0d exp=%0d", case_name, mag19, (w1 * a7));
        failures = failures + 1;
    end
    if (mag20 !== (w1 * a8)) begin
        $display("FAIL %0s mag20 got=%0d exp=%0d", case_name, mag20, (w1 * a8));
        failures = failures + 1;
    end
    if (mag21 !== (w1 * a9)) begin
        $display("FAIL %0s mag21 got=%0d exp=%0d", case_name, mag21, (w1 * a9));
        failures = failures + 1;
    end
    if (mag22 !== (w1 * a10)) begin
        $display("FAIL %0s mag22 got=%0d exp=%0d", case_name, mag22, (w1 * a10));
        failures = failures + 1;
    end
    if (mag23 !== (w1 * a11)) begin
        $display("FAIL %0s mag23 got=%0d exp=%0d", case_name, mag23, (w1 * a11));
        failures = failures + 1;
    end
    if (mag24 !== (w2 * a0)) begin
        $display("FAIL %0s mag24 got=%0d exp=%0d", case_name, mag24, (w2 * a0));
        failures = failures + 1;
    end
    if (mag25 !== (w2 * a1)) begin
        $display("FAIL %0s mag25 got=%0d exp=%0d", case_name, mag25, (w2 * a1));
        failures = failures + 1;
    end
    if (mag26 !== (w2 * a2)) begin
        $display("FAIL %0s mag26 got=%0d exp=%0d", case_name, mag26, (w2 * a2));
        failures = failures + 1;
    end
    if (mag27 !== (w2 * a3)) begin
        $display("FAIL %0s mag27 got=%0d exp=%0d", case_name, mag27, (w2 * a3));
        failures = failures + 1;
    end
    if (mag28 !== (w2 * a4)) begin
        $display("FAIL %0s mag28 got=%0d exp=%0d", case_name, mag28, (w2 * a4));
        failures = failures + 1;
    end
    if (mag29 !== (w2 * a5)) begin
        $display("FAIL %0s mag29 got=%0d exp=%0d", case_name, mag29, (w2 * a5));
        failures = failures + 1;
    end
    if (mag30 !== (w2 * a6)) begin
        $display("FAIL %0s mag30 got=%0d exp=%0d", case_name, mag30, (w2 * a6));
        failures = failures + 1;
    end
    if (mag31 !== (w2 * a7)) begin
        $display("FAIL %0s mag31 got=%0d exp=%0d", case_name, mag31, (w2 * a7));
        failures = failures + 1;
    end
    if (mag32 !== (w2 * a8)) begin
        $display("FAIL %0s mag32 got=%0d exp=%0d", case_name, mag32, (w2 * a8));
        failures = failures + 1;
    end
    if (mag33 !== (w2 * a9)) begin
        $display("FAIL %0s mag33 got=%0d exp=%0d", case_name, mag33, (w2 * a9));
        failures = failures + 1;
    end
    if (mag34 !== (w2 * a10)) begin
        $display("FAIL %0s mag34 got=%0d exp=%0d", case_name, mag34, (w2 * a10));
        failures = failures + 1;
    end
    if (mag35 !== (w2 * a11)) begin
        $display("FAIL %0s mag35 got=%0d exp=%0d", case_name, mag35, (w2 * a11));
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
        $display("PASS MXFP4_E2M1_MAG1_UltraDSP_Packing trials=4096 products=36");
    end else begin
        $display("FAIL MXFP4_E2M1_MAG1_UltraDSP_Packing failures=%0d", failures);
    end
    $finish;
end

endmodule
