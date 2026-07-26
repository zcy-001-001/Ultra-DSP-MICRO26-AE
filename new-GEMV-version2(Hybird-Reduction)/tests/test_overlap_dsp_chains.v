`timescale 1ns/1ps

module test_overlap_dsp_chains;
    reg clk = 1'b0;
    reg ce = 1'b1;
    reg active = 1'b0;
    reg [4 * 8 * 8 - 1:0] inputs4 = 0;
    reg [4 * 4 * 12 - 1:0] inputs_lut = 0;
    wire [4 * 12 - 1:0] sums4;
    wire [4 * 13 - 1:0] sums_lut;
    integer expected4 [0:31][0:3];
    integer expected_lut [0:31][0:3];
    integer accepted = 0;
    integer checked4 = 0;
    integer checked_lut = 0;
    integer token;
    integer lane;
    integer term;
    integer value;
    integer errors = 0;

    always #2.5 clk = ~clk;

    w4a4_overlap_chain8_4x12 chain4 (
        .clk(clk), .ce(ce), .inputs(inputs4), .sums(sums4)
    );

    w4a4_lut_reduce4x12 lut_tree (
        .clk(clk), .ce(ce), .inputs(inputs_lut), .sums(sums_lut)
    );

    task automatic drive_token;
        input integer index;
        integer local_lane;
        integer local_term;
        integer local_value;
        begin
            inputs4 = 0;
            inputs_lut = 0;
            for (local_lane = 0; local_lane < 4;
                 local_lane = local_lane + 1) begin
                expected4[index][local_lane] = 0;
                for (local_term = 0; local_term < 8;
                     local_term = local_term + 1) begin
                    if (index == 0)
                        local_value = local_lane[0] ? -98 : 98;
                    else
                        local_value =
                            ((index * 17 + local_lane * 11 +
                              local_term * 7) % 197) - 98;
                    inputs4[(local_lane * 8 + local_term) * 8 +: 8] =
                        local_value;
                    expected4[index][local_lane] =
                        expected4[index][local_lane] + local_value;
                end
            end
            for (local_lane = 0; local_lane < 4;
                 local_lane = local_lane + 1) begin
                expected_lut[index][local_lane] = 0;
                for (local_term = 0; local_term < 4;
                     local_term = local_term + 1) begin
                    if (index == 0)
                        local_value = local_lane[0] ? -784 : 784;
                    else
                        local_value =
                            ((index * 71 + local_lane * 103 +
                              local_term * 137) % 1569) - 784;
                    inputs_lut[
                        (local_lane * 4 + local_term) * 12 +: 12] =
                        local_value;
                    expected_lut[index][local_lane] =
                        expected_lut[index][local_lane] + local_value;
                end
            end
        end
    endtask

    always @(posedge clk) begin
        if (active && ce) begin
            accepted = accepted + 1;
            #0.2;
            if (accepted >= 4 && checked4 < 24) begin
                for (lane = 0; lane < 4; lane = lane + 1)
                    if ($signed(sums4[lane * 12 +: 12]) !==
                        expected4[checked4][lane]) begin
                        $display("FAIL: FOUR12 token=%0d lane=%0d got=%0d expected=%0d",
                                 checked4, lane,
                                 $signed(sums4[lane * 12 +: 12]),
                                 expected4[checked4][lane]);
                        errors = errors + 1;
                    end
                checked4 = checked4 + 1;
            end
            if (accepted >= 2 && checked_lut < 24) begin
                for (lane = 0; lane < 4; lane = lane + 1)
                    if ($signed(sums_lut[lane * 13 +: 13]) !==
                        expected_lut[checked_lut][lane]) begin
                        $display("FAIL: LUT tree token=%0d lane=%0d got=%0d expected=%0d",
                                 checked_lut, lane,
                                 $signed(sums_lut[lane * 13 +: 13]),
                                 expected_lut[checked_lut][lane]);
                        errors = errors + 1;
                    end
                checked_lut = checked_lut + 1;
            end
        end
    end

    initial begin
        #105.2;
        drive_token(0);
        active = 1'b1;
        for (token = 1; token < 24; token = token + 1) begin
            @(negedge clk);
            drive_token(token);
            if (token == 11) begin
                ce = 1'b0;
                @(negedge clk);
                ce = 1'b1;
            end
        end
        while (checked4 < 24 || checked_lut < 24) begin
            @(negedge clk);
            inputs4 = 0;
            inputs_lut = 0;
        end
        if (errors != 0)
            $fatal(1, "FAIL: overlap DSP chain errors=%0d", errors);
        $display("PASS: FOUR12 chain latency=4, LUT tree latency=2, II=1");
        $finish;
    end

endmodule
