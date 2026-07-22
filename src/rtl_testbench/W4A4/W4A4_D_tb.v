`timescale 1ns / 1ps

module INT4_INT4_D_tb();

reg clk;

reg [3:0] w1, w2, w3, w4, w5, w6, w7;
reg [3:0] a1;

wire [6:0] result1, result2, result3, result4, result5, result6, result7;

integer test_count;
integer pass_count;
integer fail_count;
integer group_count;
integer group_pass;
integer warmup_tests;
reg compare_enable;

INT4_INT4_D dut (
    .clk(clk),
    .w1(w1), .w2(w2), .w3(w3), .w4(w4), .w5(w5), .w6(w6), .w7(w7),
    .a1(a1),
    .result1(result1), .result2(result2), .result3(result3),
    .result4(result4), .result5(result5), .result6(result6), .result7(result7)
);

initial begin
    clk = 0;
    forever #5 clk = ~clk;
end

function signed [3:0] sign_mag_to_2comp_w4;
    input [3:0] sign_mag;
    reg sign_bit;
    reg [2:0] magnitude;
    begin
        sign_bit  = sign_mag[3];
        magnitude = sign_mag[2:0];
        if (sign_bit == 0)
            sign_mag_to_2comp_w4 = {1'b0, magnitude};
        else
            sign_mag_to_2comp_w4 = -{1'b0, magnitude};
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
    input [3:0] w1_in, w2_in, w3_in, w4_in, w5_in, w6_in, w7_in;
    input [3:0] a1_in;
    input [80*8-1:0] test_name;

    reg signed [3:0] w1s, w2s, w3s, w4s, w5s, w6s, w7s;
    reg signed [3:0] a1s;

    reg signed [6:0] exp1, exp2, exp3, exp4, exp5, exp6, exp7;
    reg test_pass;
    begin
        test_count = test_count + 1;

        w1 = w1_in; w2 = w2_in; w3 = w3_in;
        w4 = w4_in; w5 = w5_in; w6 = w6_in; w7 = w7_in;
        a1 = a1_in;

        w1s = sign_mag_to_2comp_w4(w1_in);
        w2s = sign_mag_to_2comp_w4(w2_in);
        w3s = sign_mag_to_2comp_w4(w3_in);
        w4s = sign_mag_to_2comp_w4(w4_in);
        w5s = sign_mag_to_2comp_w4(w5_in);
        w6s = sign_mag_to_2comp_w4(w6_in);
        w7s = sign_mag_to_2comp_w4(w7_in);

        a1s = process_a4(a1_in);

        exp1 = w1s * a1s;
        exp2 = w2s * a1s;
        exp3 = w3s * a1s;
        exp4 = w4s * a1s;
        exp5 = w5s * a1s;
        exp6 = w6s * a1s;
        exp7 = w7s * a1s;

        repeat(3) @(posedge clk);
        #1;

        if (!compare_enable) begin
            warmup_tests = warmup_tests + 1;
            $display("[WARMUP] Test %0d: %s - skipping comparison", test_count, test_name);
        end else begin
            test_pass = 1'b1;
            if ($signed(result1) !== exp1) test_pass = 1'b0;
            if ($signed(result2) !== exp2) test_pass = 1'b0;
            if ($signed(result3) !== exp3) test_pass = 1'b0;
            if ($signed(result4) !== exp4) test_pass = 1'b0;
            if ($signed(result5) !== exp5) test_pass = 1'b0;
            if ($signed(result6) !== exp6) test_pass = 1'b0;
            if ($signed(result7) !== exp7) test_pass = 1'b0;

            group_count = group_count + 1;

            if (test_pass) begin
                pass_count = pass_count + 1;
                group_pass = group_pass + 1;
                $display("[PASS] Test %0d: %s", test_count, test_name);
            end else begin
                fail_count = fail_count + 1;
                $display("========================================");
                $display("[FAIL] Test %0d: %s", test_count, test_name);
                $display("----------------------------------------");
                $display("w = {%0d,%0d,%0d,%0d,%0d,%0d,%0d}, a1=%0d",
                         w1s, w2s, w3s, w4s, w5s, w6s, w7s, a1s);
                $display("Got   : %0d %0d %0d %0d %0d %0d %0d",
                         $signed(result1), $signed(result2), $signed(result3),
                         $signed(result4), $signed(result5), $signed(result6), $signed(result7));
                $display("Expect: %0d %0d %0d %0d %0d %0d %0d",
                         exp1, exp2, exp3, exp4, exp5, exp6, exp7);
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
        group_pass = 0;
    end
endtask

initial begin
    test_count = 0;
    pass_count = 0;
    fail_count = 0;
    group_count = 0;
    group_pass = 0;
    warmup_tests = 0;
    compare_enable = 0;

    w1 = 0; w2 = 0; w3 = 0; w4 = 0; w5 = 0; w6 = 0; w7 = 0;
    a1 = 0;

    $display("========================================");
    $display("  INT4_INT4_D Testbench");
    $display("========================================");
    $display("Start Time: %0t", $time);
    $display("");

    repeat(5) @(posedge clk);

    $display("\n=== Group 1: Warmup ===");
    run_test(4'b0000,4'b0000,4'b0000,4'b0000,4'b0000,4'b0000,4'b0000,
             4'b0000,
             "Warmup all zeros");
    display_group_summary("Group 1 Warmup Summary");

    $display("\n=== Group 2: Warmup ===");
    run_test(4'b0001,4'b0010,4'b0011,4'b0100,4'b0101,4'b0110,4'b0111,
             4'b0001,
             "Warmup basic pos pattern");
    display_group_summary("Group 2 Warmup Summary");

    compare_enable = 1;

    $display("\n=== Group 3: Pos x Pos basic ===");
    run_test(4'b0001,4'b0010,4'b0011,4'b0100,4'b0101,4'b0110,4'b0111,
             4'b0001,
             "Pos x Pos 1..7 x +1");
    run_test(4'b0011,4'b0011,4'b0011,4'b0011,4'b0011,4'b0011,4'b0011,
             4'b0011,
             "All +3 x +3");
    display_group_summary("Group 3 Summary");

    $display("\n=== Group 4: Pos x Neg ===");
    run_test(4'b0001,4'b0010,4'b0011,4'b0100,4'b0101,4'b0110,4'b0111,
             4'b1111,
             "1..7 x -1");
    run_test(4'b0011,4'b0010,4'b0001,4'b0011,4'b0010,4'b0001,4'b0011,
             4'b1101,
             "Pos mix x -3");
    display_group_summary("Group 4 Summary");

    $display("\n=== Group 5: Neg x Pos ===");
    run_test(4'b1001,4'b1010,4'b1011,4'b1100,4'b1101,4'b1110,4'b1111,
             4'b0011,
             "-1..-7 x +3");
    run_test(4'b1011,4'b1011,4'b1011,4'b1011,4'b1011,4'b1011,4'b1011,
             4'b0111,
             "All -3 x +7");
    display_group_summary("Group 5 Summary");

    $display("\n=== Group 6: Neg x Neg ===");
    run_test(4'b1001,4'b1010,4'b1011,4'b1100,4'b1101,4'b1110,4'b1111,
             4'b1101,
             "-1..-7 x -3");
    run_test(4'b1111,4'b1111,4'b1111,4'b1111,4'b1111,4'b1111,4'b1111,
             4'b1111,
             "All -1 x -1");
    display_group_summary("Group 6 Summary");

    $display("\n=== Group 7: Zero and boundary ===");
    run_test(4'b0000,4'b0000,4'b0000,4'b0000,4'b0000,4'b0000,4'b0000,
             4'b0111,
             "All w=0");
    run_test(4'b0001,4'b0010,4'b0011,4'b0100,4'b0101,4'b0110,4'b0111,
             4'b0000,
             "All a=0");
    run_test(4'b0111,4'b0111,4'b0111,4'b0111,4'b0111,4'b0111,4'b0111,
             4'b0111,
             "Max positive 7x7");
    run_test(4'b1001,4'b1001,4'b1001,4'b1001,4'b1001,4'b1001,4'b1001,
             4'b1001,
             "Min negative -7x-7");
    display_group_summary("Group 7 Summary");

    $display("\n=== Group 8: a = -8 clamp ===");
    run_test(4'b0001,4'b0010,4'b0011,4'b0100,4'b0101,4'b0110,4'b0111,
             4'b1000,
             "Pos w x a=-8");
    run_test(4'b1001,4'b1010,4'b1011,4'b1100,4'b1101,4'b1110,4'b1111,
             4'b1000,
             "Neg w x a=-8");
    display_group_summary("Group 8 Summary");

    $display("\n=== Group 9: Mixed/random-like ===");
    run_test(4'b0001,4'b1011,4'b0010,4'b1101,4'b0011,4'b1110,4'b0100,
             4'b0010,
             "Random mix 1");
    run_test(4'b0111,4'b1001,4'b0110,4'b1010,4'b0101,4'b1011,4'b0100,
             4'b1110,
             "Random mix 2");
    display_group_summary("Group 9 Summary");

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
