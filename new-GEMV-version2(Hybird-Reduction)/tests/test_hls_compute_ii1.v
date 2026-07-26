`timescale 1ns/1ps

module test_hls_compute_ii1;
    localparam integer CHANNELS = 9;
    localparam integer COLS = 64;
    localparam integer HLS_CHANNEL_BITS = 1021;

    reg clk = 1'b0;
    reg rst = 1'b1;
    reg ap_start = 1'b0;
    reg [4095:0] weights [0:11];
    reg [255:0] acts [0:2];
    wire ap_done;
    wire ap_idle;
    wire ap_ready;
    wire [1020:0] result [0:8];
    wire [CHANNELS * HLS_CHANNEL_BITS - 1:0] packed_results =
        {result[8], result[7], result[6], result[5], result[4],
         result[3], result[2], result[1], result[0]};

    integer accepted;
    integer received;
    integer errors;
    integer expected [0:2][0:8];

    always #2.5 clk = ~clk;

    gemv_kernel_compute_gemv dut (
        .ap_clk(clk),
        .ap_rst(rst),
        .ap_start(ap_start),
        .ap_done(ap_done),
        .ap_continue(1'b1),
        .ap_idle(ap_idle),
        .ap_ready(ap_ready),
        .ap_ce(1'b1),
        .p_read(weights[0]),
        .p_read1(weights[1]),
        .p_read2(weights[2]),
        .p_read3(weights[3]),
        .p_read4(weights[4]),
        .p_read5(weights[5]),
        .p_read6(weights[6]),
        .p_read7(weights[7]),
        .p_read8(weights[8]),
        .p_read9(weights[9]),
        .p_read10(weights[10]),
        .p_read11(weights[11]),
        .p_read12(acts[0]),
        .p_read13(acts[1]),
        .p_read14(acts[2]),
        .ap_return_0(result[0]),
        .ap_return_1(result[1]),
        .ap_return_2(result[2]),
        .ap_return_3(result[3]),
        .ap_return_4(result[4]),
        .ap_return_5(result[5]),
        .ap_return_6(result[6]),
        .ap_return_7(result[7]),
        .ap_return_8(result[8])
    );

    task automatic set_vector;
        input [3:0] weight0_raw;
        input [3:0] weight1_raw;
        input [3:0] weight2_raw;
        input [3:0] act0_raw;
        input [3:0] act1_raw;
        input [3:0] act2_raw;
        integer index;
        begin
            for (index = 0; index < 4; index = index + 1)
                weights[index] = {1024{weight0_raw}};
            for (index = 4; index < 8; index = index + 1)
                weights[index] = {1024{weight1_raw}};
            for (index = 8; index < 12; index = index + 1)
                weights[index] = {1024{weight2_raw}};
            acts[0] = {64{act0_raw}};
            acts[1] = {64{act1_raw}};
            acts[2] = {64{act2_raw}};
        end
    endtask

    always @(posedge clk) begin : check_pipeline
        integer channel;
        integer col;
        integer actual;

        if (!rst && ap_start) begin
            if (!ap_ready) begin
                $display("FAIL: generated compute core deasserted ap_ready on input %0d",
                         accepted);
                errors = errors + 1;
            end else begin
                accepted = accepted + 1;
            end
        end

        if (!rst && ap_done) begin
            if (received >= 3) begin
                $display("FAIL: unexpected extra compute result");
                errors = errors + 1;
            end else begin
                for (channel = 0; channel < CHANNELS;
                     channel = channel + 1) begin
                    for (col = 0; col < COLS; col = col + 1) begin
                        if (col == COLS - 1)
                            actual = $signed(packed_results[
                                channel * HLS_CHANNEL_BITS + col * 16 +: 13]);
                        else
                            actual = $signed(packed_results[
                                channel * HLS_CHANNEL_BITS + col * 16 +: 16]);
                        if (actual != expected[received][channel]) begin
                            $display("FAIL: result=%0d channel=%0d col=%0d actual=%0d expected=%0d",
                                     received, channel, col, actual,
                                     expected[received][channel]);
                            errors = errors + 1;
                        end
                    end
                end
            end
            received = received + 1;
        end
    end

    initial begin
        accepted = 0;
        received = 0;
        errors = 0;
        expected[0][0] = 64;
        expected[0][1] = 192;
        expected[0][2] = -128;
        expected[0][3] = -128;
        expected[0][4] = -384;
        expected[0][5] = 256;
        expected[0][6] = 192;
        expected[0][7] = 576;
        expected[0][8] = -384;
        expected[1][0] = -384;
        expected[1][1] = 256;
        expected[1][2] = -640;
        expected[1][3] = -768;
        expected[1][4] = 512;
        expected[1][5] = -1280;
        expected[1][6] = 960;
        expected[1][7] = -640;
        expected[1][8] = 1600;
        expected[2][0] = 3136;
        expected[2][1] = 1344;
        expected[2][2] = -3136;
        expected[2][3] = 896;
        expected[2][4] = 384;
        expected[2][5] = -896;
        expected[2][6] = 448;
        expected[2][7] = 192;
        expected[2][8] = -448;
        set_vector(4'h0, 4'h0, 4'h0, 4'h0, 4'h0, 4'h0);

        // Xilinx glbl holds the DSP48E2 global reset active for 100 ns.
        #105;
        @(negedge clk);
        rst = 1'b0;
        set_vector(4'h1, 4'ha, 4'h3, 4'h1, 4'h3, 4'he);
        ap_start = 1'b1;

        @(negedge clk);
        set_vector(4'ha, 4'hc, 4'h5, 4'h3, 4'he, 4'h5);

        @(negedge clk);
        set_vector(4'h7, 4'h2, 4'h1, 4'h7, 4'h3, 4'h9);

        @(negedge clk);
        ap_start = 1'b0;
        set_vector(4'h0, 4'h0, 4'h0, 4'h0, 4'h0, 4'h0);

        repeat (24) @(posedge clk);
        if (accepted != 3) begin
            $display("FAIL: expected 3 accepted inputs, got %0d", accepted);
            errors = errors + 1;
        end
        if (received != 3) begin
            $display("FAIL: expected 3 outputs, got %0d", received);
            errors = errors + 1;
        end
        if (errors != 0)
            $fatal(1, "FAIL: %0d generated compute-core errors", errors);

        $display("PASS: generated HLS compute core accepts and returns 3 consecutive GEMVs at II=1");
        $finish;
    end
endmodule
