`timescale 1ns / 1ps

module W3A4_PD_tb();

reg clk;
reg mode;

reg [2:0] w1, w2, w3, w4, w5, w6, w7, w8, w9;
reg [3:0] a1, a2;

wire [5:0] result1, result2, result3, result4, result5, result6;
wire [5:0] result7, result8, result9, result10, result11, result12;

integer test_count;
integer pass_count;
integer fail_count;
integer group_count;
integer group_pass;
integer warmup_tests;
reg compare_enable;

W3A4_PD dut (
    .clk(clk),
    .mode(mode),
    .w1(w1), .w2(w2), .w3(w3), .w4(w4), .w5(w5), .w6(w6), .w7(w7), .w8(w8), .w9(w9),
    .a1(a1), .a2(a2),
    .result1(result1), .result2(result2), .result3(result3), .result4(result4),
    .result5(result5), .result6(result6), .result7(result7), .result8(result8),
    .result9(result9), .result10(result10), .result11(result11), .result12(result12)
);

initial begin
    clk = 0;
    forever #5 clk = ~clk;
end

function signed [2:0] sign_mag_to_2comp_w3;
    input [2:0] sign_mag;
    reg sign_bit;
    reg [1:0] magnitude;
    begin
        sign_bit = sign_mag[2];
        magnitude = sign_mag[1:0];
        if (sign_bit == 0)
            sign_mag_to_2comp_w3 = {1'b0, magnitude};
        else
            sign_mag_to_2comp_w3 = -{1'b0, magnitude};
    end
endfunction

function signed [3:0] process_a4;
    input signed [3:0] a_val;
    reg signed [3:0] tmp;
    begin
        tmp = a_val;
        if (a_val == 4'b1000)
            tmp = -4'sd7;
        process_a4 = tmp;
    end
endfunction

task run_test;
    input        mode_in;
    input [2:0]  w1_in, w2_in, w3_in, w4_in, w5_in, w6_in, w7_in, w8_in, w9_in;
    input [3:0]  a1_in, a2_in;
    input [80*8-1:0] test_name;

    reg signed [2:0] w1s, w2s, w3s, w4s, w5s, w6s, w7s, w8s, w9s;
    reg signed [3:0] a1s, a2s;

    reg signed [5:0] exp1, exp2, exp3, exp4, exp5, exp6;
    reg signed [5:0] exp7, exp8, exp9, exp10, exp11, exp12;

    integer i;
    reg test_pass;
    begin
        test_count  = test_count + 1;

        mode = mode_in;
        w1 = w1_in; w2 = w2_in; w3 = w3_in;
        w4 = w4_in; w5 = w5_in; w6 = w6_in;
        w7 = w7_in; w8 = w8_in; w9 = w9_in;
        a1 = a1_in; a2 = a2_in;

        w1s = sign_mag_to_2comp_w3(w1_in);
        w2s = sign_mag_to_2comp_w3(w2_in);
        w3s = sign_mag_to_2comp_w3(w3_in);
        w4s = sign_mag_to_2comp_w3(w4_in);
        w5s = sign_mag_to_2comp_w3(w5_in);
        w6s = sign_mag_to_2comp_w3(w6_in);
        w7s = sign_mag_to_2comp_w3(w7_in);
        w8s = sign_mag_to_2comp_w3(w8_in);
        w9s = sign_mag_to_2comp_w3(w9_in);

        a1s = process_a4(a1_in);
        a2s = process_a4(a2_in);

        if (mode_in == 1'b0) begin
            exp1  = w1s * a1s;
            exp2  = w2s * a1s;
            exp3  = w3s * a1s;
            exp4  = w4s * a1s;
            exp5  = w5s * a1s;
            exp6  = w6s * a1s;
            exp7  = w1s * a2s;
            exp8  = w2s * a2s;
            exp9  = w3s * a2s;
            exp10 = w4s * a2s;
            exp11 = w5s * a2s;
            exp12 = w6s * a2s;
        end else begin
            exp1  = w1s * a1s;
            exp2  = w2s * a1s;
            exp3  = w3s * a1s;
            exp4  = w4s * a1s;
            exp5  = w5s * a1s;
            exp6  = w6s * a1s;
            exp7  = w7s * a1s;
            exp8  = w8s * a1s;
            exp9  = w9s * a1s;
            exp10 = 6'sd0;
            exp11 = 6'sd0;
            exp12 = 6'sd0;
        end

        repeat(3) @(posedge clk);
        #1;

        if (!compare_enable) begin
            warmup_tests = warmup_tests + 1;
            $display("[WARMUP] Test %0d: %s (mode=%0d) - skipping comparison", test_count, test_name, mode_in);
        end else begin
            test_pass = 1'b1;

            if ($signed(result1)  !== exp1)  test_pass = 1'b0;
            if ($signed(result2)  !== exp2)  test_pass = 1'b0;
            if ($signed(result3)  !== exp3)  test_pass = 1'b0;
            if ($signed(result4)  !== exp4)  test_pass = 1'b0;
            if ($signed(result5)  !== exp5)  test_pass = 1'b0;
            if ($signed(result6)  !== exp6)  test_pass = 1'b0;
            if ($signed(result7)  !== exp7)  test_pass = 1'b0;
            if ($signed(result8)  !== exp8)  test_pass = 1'b0;
            if ($signed(result9)  !== exp9)  test_pass = 1'b0;
            if (mode_in == 1'b0) begin
                if ($signed(result10) !== exp10) test_pass = 1'b0;
                if ($signed(result11) !== exp11) test_pass = 1'b0;
                if ($signed(result12) !== exp12) test_pass = 1'b0;
            end else begin
                if ($signed(result10) !== 6'sd0) test_pass = 1'b0;
                if ($signed(result11) !== 6'sd0) test_pass = 1'b0;
                if ($signed(result12) !== 6'sd0) test_pass = 1'b0;
            end

            group_count = group_count + 1;

            if (test_pass) begin
                pass_count = pass_count + 1;
                group_pass = group_pass + 1;
                $display("[PASS] Test %0d: %s (mode=%0d)", test_count, test_name, mode_in);
            end else begin
                fail_count = fail_count + 1;
                $display("========================================");
                $display("[FAIL] Test %0d: %s (mode=%0d)", test_count, test_name, mode_in);
                $display("w = {%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d}, a1=%0d, a2=%0d",
                         w1s,w2s,w3s,w4s,w5s,w6s,w7s,w8s,w9s,a1s,a2s);
                $display("Got   : %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d",
                         $signed(result1), $signed(result2), $signed(result3), $signed(result4),
                         $signed(result5), $signed(result6), $signed(result7), $signed(result8),
                         $signed(result9), $signed(result10), $signed(result11), $signed(result12));
                $display("Expect: %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d",
                         exp1,exp2,exp3,exp4,exp5,exp6,exp7,exp8,exp9,exp10,exp11,exp12);
                $display("========================================");
            end
        end
    end
endtask

task display_group_summary;
    input [80*8-1:0] group_name;
    begin
        $display("  >> %s: %0d/%0d passed", group_name, group_pass, group_count);
        group_count = 0;
        group_pass  = 0;
    end
endtask

initial begin
    test_count  = 0;
    pass_count  = 0;
    fail_count  = 0;
    group_count = 0;
    group_pass  = 0;
    warmup_tests = 0;
    compare_enable = 0;

    mode = 0;
    w1 = 0; w2 = 0; w3 = 0; w4 = 0; w5 = 0; w6 = 0; w7 = 0; w8 = 0; w9 = 0;
    a1 = 0; a2 = 0;

    $display("========================================");
    $display("  W3A4_PD (Hybrid P/D) Testbench");
    $display("========================================");
    $display("Start Time: %0t", $time);
    $display("");

    repeat(5) @(posedge clk);

    $display("\n=== Group 1: Warmup ===");
    run_test(1'b0,
             3'b000,3'b000,3'b000,3'b000,3'b000,3'b000,3'b000,3'b000,3'b000,
             4'b0000,4'b0000,
             "Warmup all zeros");
    display_group_summary("Group 1 Warmup Summary");

    $display("\n=== Group 2: Warmup ===");
    run_test(1'b0,
             3'b001,3'b010,3'b011,3'b001,3'b010,3'b011,3'b000,3'b000,3'b000,
             4'b0001,4'b0010,
             "Warmup basic pos pattern");
    display_group_summary("Group 2 Warmup Summary");

    compare_enable = 1;

    $display("\n=== Group 3: Packaging mode (mode=0) ===");
    run_test(1'b0,
             3'b001,3'b010,3'b011,3'b001,3'b010,3'b011,3'b000,3'b000,3'b000,
             4'b0001,4'b0010,
             "P: small pos");
    run_test(1'b0,
             3'b011,3'b011,3'b011,3'b010,3'b001,3'b101,3'b000,3'b000,3'b000,
             4'b0111,4'b1001,
             "P: boundary mix");
    display_group_summary("Group 3 Summary");

    $display("\n=== Group 4: Packaging neg/zero ===");
    run_test(1'b0,
             3'b101,3'b110,3'b111,3'b101,3'b110,3'b111,3'b000,3'b000,3'b000,
             4'b1111,4'b1110,
             "P: neg w, neg a");
    run_test(1'b0,
             3'b000,3'b000,3'b000,3'b000,3'b000,3'b000,3'b000,3'b000,3'b000,
             4'b0111,4'b1111,
             "P: all w=0");
    display_group_summary("Group 4 Summary");

    $display("\n=== Group 5: Depackaging mode (mode=1) ===");
    run_test(1'b1,
             3'b001,3'b010,3'b011,3'b001,3'b010,3'b011,3'b001,3'b010,3'b011,
             4'b0001,4'b0000,
             "D: pos w, a1=+1");
    run_test(1'b1,
             3'b101,3'b110,3'b111,3'b101,3'b110,3'b111,3'b101,3'b110,3'b111,
             4'b1111,4'b0000,
             "D: neg w, a1=-1");
    run_test(1'b1,
             3'b011,3'b011,3'b011,3'b011,3'b011,3'b011,3'b011,3'b011,3'b011,
             4'b1000,4'b0000,
             "D: a1=-8 clamp");
    display_group_summary("Group 5 Summary");

    repeat(3) @(posedge clk);

    $display("\n========================================");
    $display("  FINAL TEST SUMMARY");
    $display("========================================");
    $display("Total Tests : %0d", test_count - warmup_tests);
    $display("Passed      : %0d", pass_count);
    $display("Failed      : %0d", fail_count);
    if ((test_count - warmup_tests) > 0)
        $display("Pass Rate   : %0.2f%%", (pass_count * 100.0) / (test_count - warmup_tests));
    $display("========================================");

    if (fail_count == 0)
        $display("\n*** ALL TESTS PASSED! ***\n");
    else
        $display("\n*** %0d TEST(S) FAILED! ***\n", fail_count);

    $display("End Time: %0t", $time);
    $display("========================================\n");

    $finish;
end

endmodule
