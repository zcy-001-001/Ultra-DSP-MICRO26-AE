`timescale 1ns/1ps

module test_simd_dsp_add;
    localparam integer LANES = 4;

    reg clk = 1'b0;
    reg [LANES * 8 - 1:0] lhs8 = 0;
    reg [LANES * 8 - 1:0] rhs8 = 0;
    reg [LANES * 9 - 1:0] lhs9 = 0;
    reg [LANES * 9 - 1:0] rhs9 = 0;
    reg [LANES * 10 - 1:0] lhs10 = 0;
    reg [LANES * 10 - 1:0] rhs10 = 0;
    wire [LANES * 9 - 1:0] sum8;
    wire [LANES * 10 - 1:0] sum9;
    wire [LANES * 11 - 1:0] sum10;

    integer expected8[0:LANES-1];
    integer expected9[0:LANES-1];
    integer expected10[0:LANES-1];
    integer seed;
    integer vector_index;
    integer lane;
    integer lhs_value;
    integer rhs_value;
    integer actual;

    always #2.5 clk = ~clk;

    w4a4_packed_dsp_add #(.INPUT_WIDTH(8), .LANES(LANES)) add8 (
        .clk(clk), .ce(1'b1), .lhs(lhs8), .rhs(rhs8), .value(sum8)
    );
    w4a4_packed_dsp_add #(.INPUT_WIDTH(9), .LANES(LANES)) add9 (
        .clk(clk), .ce(1'b1), .lhs(lhs9), .rhs(rhs9), .value(sum9)
    );
    w4a4_packed_dsp_add #(.INPUT_WIDTH(10), .LANES(LANES)) add10 (
        .clk(clk), .ce(1'b1), .lhs(lhs10), .rhs(rhs10), .value(sum10)
    );

    task automatic drive_vector;
        input integer index;
        begin
            @(negedge clk);
            for (lane = 0; lane < LANES; lane = lane + 1) begin
                if (index == 0) begin
                    lhs_value = (lane[0]) ? 127 : -128;
                    rhs_value = (lane[1]) ? 127 : -128;
                end else begin
                    lhs_value = ($random(seed) & 255) - 128;
                    rhs_value = ($random(seed) & 255) - 128;
                end
                lhs8[lane * 8 +: 8] = lhs_value;
                rhs8[lane * 8 +: 8] = rhs_value;
                expected8[lane] = lhs_value + rhs_value;

                if (index == 0) begin
                    lhs_value = (lane[0]) ? 255 : -256;
                    rhs_value = (lane[1]) ? 255 : -256;
                end else begin
                    lhs_value = ($random(seed) & 511) - 256;
                    rhs_value = ($random(seed) & 511) - 256;
                end
                lhs9[lane * 9 +: 9] = lhs_value;
                rhs9[lane * 9 +: 9] = rhs_value;
                expected9[lane] = lhs_value + rhs_value;

                if (index == 0) begin
                    lhs_value = (lane[0]) ? 511 : -512;
                    rhs_value = (lane[1]) ? 511 : -512;
                end else begin
                    lhs_value = ($random(seed) & 1023) - 512;
                    rhs_value = ($random(seed) & 1023) - 512;
                end
                lhs10[lane * 10 +: 10] = lhs_value;
                rhs10[lane * 10 +: 10] = rhs_value;
                expected10[lane] = lhs_value + rhs_value;
            end

            @(posedge clk);
            #1;
            for (lane = 0; lane < LANES; lane = lane + 1) begin
                actual = $signed(sum8[lane * 9 +: 9]);
                if (actual != expected8[lane]) begin
                    $fatal(1, "8-bit SIMD lane %0d: got %0d expected %0d",
                           lane, actual, expected8[lane]);
                end
                actual = $signed(sum9[lane * 10 +: 10]);
                if (actual != expected9[lane]) begin
                    $fatal(1, "9-bit SIMD lane %0d: got %0d expected %0d",
                           lane, actual, expected9[lane]);
                end
                actual = $signed(sum10[lane * 11 +: 11]);
                if (actual != expected10[lane]) begin
                    $fatal(1, "10-bit SIMD lane %0d: got %0d expected %0d",
                           lane, actual, expected10[lane]);
                end
            end
        end
    endtask

    initial begin
        seed = 32'h48e2_4a4;
        // UNISIM's global set/reset is active for the first 100 ns.
        #105;
        for (vector_index = 0; vector_index < 2001;
             vector_index = vector_index + 1) begin
            drive_vector(vector_index);
        end
        $display("PASS: DSP48E2 FOUR12 boundary and 2000 consecutive random vectors");
        $finish;
    end
endmodule
