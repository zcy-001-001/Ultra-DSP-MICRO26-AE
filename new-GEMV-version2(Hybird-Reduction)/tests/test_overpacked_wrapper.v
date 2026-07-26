`timescale 1ns/1ps

module test_overpacked_wrapper;
    localparam integer ROWS = 64;
    localparam integer COLS = 64;
    localparam integer NUM_PE = ROWS * COLS;
    localparam integer GROUP_BITS = (NUM_PE / 4) * 4;
    localparam integer CHANNELS = 9;
    localparam integer CHANNEL_SUM_BITS = COLS * 16;
    localparam integer SUM_BUS_BITS = CHANNELS * COLS * 16;

    reg clk = 1'b0;
    reg rst = 1'b1;
    reg input_valid = 1'b0;
    reg outputs_ready = 1'b1;
    reg [NUM_PE * 4 - 1:0] weights1 = 0;
    reg [NUM_PE * 4 - 1:0] weights2 = 0;
    reg [NUM_PE * 4 - 1:0] weights3 = 0;
    reg [ROWS * 4 - 1:0] acts1 = 0;
    reg [ROWS * 4 - 1:0] acts2 = 0;
    reg [ROWS * 4 - 1:0] acts3 = 0;
    reg [SUM_BUS_BITS-1:0] expected0 = 0;
    reg [SUM_BUS_BITS-1:0] expected1 = 0;
    wire [CHANNEL_SUM_BITS-1:0] sum0;
    wire [CHANNEL_SUM_BITS-1:0] sum1;
    wire [CHANNEL_SUM_BITS-1:0] sum2;
    wire [CHANNEL_SUM_BITS-1:0] sum3;
    wire [CHANNEL_SUM_BITS-1:0] sum4;
    wire [CHANNEL_SUM_BITS-1:0] sum5;
    wire [CHANNEL_SUM_BITS-1:0] sum6;
    wire [CHANNEL_SUM_BITS-1:0] sum7;
    wire [CHANNEL_SUM_BITS-1:0] sum8;
    wire sum0_ap_vld, sum1_ap_vld, sum2_ap_vld;
    wire sum3_ap_vld, sum4_ap_vld, sum5_ap_vld;
    wire sum6_ap_vld, sum7_ap_vld, sum8_ap_vld;
    wire ap_done, ap_ready, ap_idle;
    wire [SUM_BUS_BITS-1:0] sums =
        {sum8, sum7, sum6, sum5, sum4, sum3, sum2, sum1, sum0};
    wire inputs_read = ap_ready;
    wire sums_write = ap_done & outputs_ready;

    integer received;
    integer errors;

    always #2.5 clk = ~clk;

    W4A4_P_chain dut (
        .ap_clk(clk), .ap_rst(rst), .ap_ce(1'b1),
        .ap_start(input_valid), .ap_continue(outputs_ready),
        .ap_done(ap_done), .ap_ready(ap_ready), .ap_idle(ap_idle),
        .w1g0(weights1[0 * GROUP_BITS +: GROUP_BITS]),
        .w1g1(weights1[1 * GROUP_BITS +: GROUP_BITS]),
        .w1g2(weights1[2 * GROUP_BITS +: GROUP_BITS]),
        .w1g3(weights1[3 * GROUP_BITS +: GROUP_BITS]),
        .w2g0(weights2[0 * GROUP_BITS +: GROUP_BITS]),
        .w2g1(weights2[1 * GROUP_BITS +: GROUP_BITS]),
        .w2g2(weights2[2 * GROUP_BITS +: GROUP_BITS]),
        .w2g3(weights2[3 * GROUP_BITS +: GROUP_BITS]),
        .w3g0(weights3[0 * GROUP_BITS +: GROUP_BITS]),
        .w3g1(weights3[1 * GROUP_BITS +: GROUP_BITS]),
        .w3g2(weights3[2 * GROUP_BITS +: GROUP_BITS]),
        .w3g3(weights3[3 * GROUP_BITS +: GROUP_BITS]),
        .a1(acts1), .a2(acts2), .a3(acts3),
        .sum0(sum0), .sum1(sum1), .sum2(sum2),
        .sum3(sum3), .sum4(sum4), .sum5(sum5),
        .sum6(sum6), .sum7(sum7), .sum8(sum8),
        .sum0_ap_vld(sum0_ap_vld), .sum1_ap_vld(sum1_ap_vld),
        .sum2_ap_vld(sum2_ap_vld), .sum3_ap_vld(sum3_ap_vld),
        .sum4_ap_vld(sum4_ap_vld), .sum5_ap_vld(sum5_ap_vld),
        .sum6_ap_vld(sum6_ap_vld), .sum7_ap_vld(sum7_ap_vld),
        .sum8_ap_vld(sum8_ap_vld)
    );

    function automatic integer decode_weight;
        input [3:0] raw;
        integer magnitude;
        begin
            magnitude = raw[2:0];
            decode_weight = raw[3] ? -magnitude : magnitude;
        end
    endfunction

    function automatic integer decode_activation;
        input [3:0] raw;
        integer signed_value;
        begin
            signed_value = raw[3] ? raw - 16 : raw;
            decode_activation = (signed_value == -8) ? -7 : signed_value;
        end
    endfunction

    task automatic build_vector;
        input integer seed;
        output reg [SUM_BUS_BITS-1:0] expected;
        integer col;
        integer row;
        integer lane;
        integer group;
        integer pe;
        integer output_index;
        integer sum;
        reg [3:0] weight_raw;
        reg [3:0] act_raw;
        begin
            weights1 = 0;
            weights2 = 0;
            weights3 = 0;
            acts1 = 0;
            acts2 = 0;
            acts3 = 0;
            expected = 0;

            for (group = 0; group < 3; group = group + 1) begin
                for (row = 0; row < ROWS; row = row + 1) begin
                    act_raw = (seed + 5 * group + 3 * row) & 15;
                    case (group)
                    0: acts1[row * 4 +: 4] = act_raw;
                    1: acts2[row * 4 +: 4] = act_raw;
                    default: acts3[row * 4 +: 4] = act_raw;
                    endcase
                end
            end

            for (col = 0; col < COLS; col = col + 1) begin
                for (row = 0; row < ROWS; row = row + 1) begin
                    pe = col * ROWS + row;
                    for (lane = 0; lane < 3; lane = lane + 1) begin
                        weight_raw =
                            (((seed + col + 2 * row + lane) & 1) << 3) |
                            ((seed + 3 * col + row + 5 * lane) & 7);
                        case (lane)
                        0: weights1[pe * 4 +: 4] = weight_raw;
                        1: weights2[pe * 4 +: 4] = weight_raw;
                        default: weights3[pe * 4 +: 4] = weight_raw;
                        endcase
                    end
                end
            end

            for (lane = 0; lane < 3; lane = lane + 1) begin
            for (group = 0; group < 3; group = group + 1) begin
                for (col = 0; col < COLS; col = col + 1) begin
                    sum = 0;
                    for (row = 0; row < ROWS; row = row + 1) begin
                        pe = col * ROWS + row;
                        case (lane)
                        0: weight_raw = weights1[pe * 4 +: 4];
                        1: weight_raw = weights2[pe * 4 +: 4];
                        default: weight_raw = weights3[pe * 4 +: 4];
                        endcase
                        case (group)
                        0: act_raw = acts1[row * 4 +: 4];
                        1: act_raw = acts2[row * 4 +: 4];
                        default: act_raw = acts3[row * 4 +: 4];
                        endcase
                        sum = sum + decode_weight(weight_raw) *
                                    decode_activation(act_raw);
                    end
                    output_index = (lane * 3 + group) * COLS + col;
                    expected[output_index * 16 +: 16] = sum;
                end
            end
            end
        end
    endtask

    always @(posedge clk) begin
        integer debug_index;
        if (!rst && inputs_read && !input_valid) begin
            $fatal(1, "ap_ready asserted without ap_start");
        end
        if (!rst &&
            {sum8_ap_vld, sum7_ap_vld, sum6_ap_vld, sum5_ap_vld,
             sum4_ap_vld, sum3_ap_vld, sum2_ap_vld, sum1_ap_vld,
             sum0_ap_vld} != {9{ap_done}}) begin
            $fatal(1, "output ap_vld signals diverged from ap_done");
        end
        if (!rst && sums_write) begin
            if (received == 0 && sums !== expected0) begin
                $display("FAIL: first pipelined result mismatch actual0=%0d expected0=%0d",
                         $signed(sums[15:0]), $signed(expected0[15:0]));
                for (debug_index = 0; debug_index < CHANNELS * COLS;
                     debug_index = debug_index + 1) begin
                    if (sums[debug_index * 16 +: 16] !==
                        expected0[debug_index * 16 +: 16]) begin
                        $display("  first mismatch index=%0d actual=%0d expected=%0d",
                                 debug_index,
                                 $signed(sums[debug_index * 16 +: 16]),
                                 $signed(expected0[debug_index * 16 +: 16]));
                        debug_index = CHANNELS * COLS;
                    end
                end
                errors = errors + 1;
            end
            if (received == 1 && sums !== expected1) begin
                $display("FAIL: second pipelined result mismatch actual0=%0d expected0=%0d",
                         $signed(sums[15:0]), $signed(expected1[15:0]));
                for (debug_index = 0; debug_index < CHANNELS * COLS;
                     debug_index = debug_index + 1) begin
                    if (sums[debug_index * 16 +: 16] !==
                        expected1[debug_index * 16 +: 16]) begin
                        $display("  first mismatch index=%0d actual=%0d expected=%0d",
                                 debug_index,
                                 $signed(sums[debug_index * 16 +: 16]),
                                 $signed(expected1[debug_index * 16 +: 16]));
                        debug_index = CHANNELS * COLS;
                    end
                end
                errors = errors + 1;
            end
            received = received + 1;
        end
    end

    initial begin
        received = 0;
        errors = 0;
        // Xilinx glbl holds the DSP48E2 global reset active for 100 ns.
        #105;
        @(negedge clk);
        rst = 1'b0;

        build_vector(3, expected0);
        input_valid = 1'b1;
        @(negedge clk);
        build_vector(11, expected1);
        @(negedge clk);
        input_valid = 1'b0;

        // Hold both in-flight results at the output and then release them.
        outputs_ready = 1'b0;
        repeat (10) @(negedge clk);
        outputs_ready = 1'b1;

        repeat (20) @(posedge clk);
        if (received != 2) begin
            $display("FAIL: expected 2 outputs, received %0d", received);
            errors = errors + 1;
        end
        if (errors != 0) begin
            $fatal(1, "FAIL: %0d wrapper errors", errors);
        end
        $display("PASS: full 64x64x9 array accepts consecutive GEMVs and stalls cleanly");
        $finish;
    end
endmodule
