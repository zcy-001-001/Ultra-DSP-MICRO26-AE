`timescale 1ns/1ps

(* keep_hierarchy = "yes", use_dsp = "no" *)
module pair_add_current (
    input wire [5:0] magnitude0,
    input wire sign0,
    input wire [5:0] magnitude1,
    input wire sign1,
    output wire signed [7:0] value
);

wire signed [7:0] signed0 =
    sign0 ? -$signed({2'b0, magnitude0}) : $signed({2'b0, magnitude0});
wire signed [7:0] signed1 =
    sign1 ? -$signed({2'b0, magnitude1}) : $signed({2'b0, magnitude1});

assign value = signed0 + signed1;

endmodule

(* keep_hierarchy = "yes", use_dsp = "no" *)
module pair_add_registered (
    input wire clk,
    input wire ce,
    input wire [5:0] magnitude0,
    input wire sign0,
    input wire [5:0] magnitude1,
    input wire sign1,
    output reg signed [7:0] value
);

wire signed [7:0] signed0 =
    sign0 ? -$signed({2'b0, magnitude0}) : $signed({2'b0, magnitude0});
wire signed [7:0] signed1 =
    sign1 ? -$signed({2'b0, magnitude1}) : $signed({2'b0, magnitude1});

always @(posedge clk) begin
    if (ce)
        value <= signed0 + signed1;
end

endmodule

(* keep_hierarchy = "yes", use_dsp = "no" *)
module pair_add_explicit_carry (
    input wire [5:0] magnitude0,
    input wire sign0,
    input wire [5:0] magnitude1,
    input wire sign1,
    output wire signed [7:0] value
);

wire [7:0] carry_di;
wire [7:0] carry_s;
wire [7:0] carry_co;

assign carry_di[0] = sign0;
assign carry_di[7] = 1'b0;

(* HLUTNM = "pair_add_lutpair0" *)
LUT4 #(.INIT(16'h4EE4)) di_bit1 (
    .I0(magnitude1[0]), .I1(sign1),
    .I2(magnitude0[0]), .I3(sign0), .O(carry_di[1])
);
(* HLUTNM = "pair_add_lutpair1" *)
LUT4 #(.INIT(16'h0660)) di_bit2 (
    .I0(magnitude1[1]), .I1(sign1),
    .I2(magnitude0[1]), .I3(sign0), .O(carry_di[2])
);
(* HLUTNM = "pair_add_lutpair2" *)
LUT4 #(.INIT(16'h0660)) di_bit3 (
    .I0(magnitude1[2]), .I1(sign1),
    .I2(magnitude0[2]), .I3(sign0), .O(carry_di[3])
);
(* HLUTNM = "pair_add_lutpair3" *)
LUT4 #(.INIT(16'h0660)) di_bit4 (
    .I0(magnitude1[3]), .I1(sign1),
    .I2(magnitude0[3]), .I3(sign0), .O(carry_di[4])
);
(* HLUTNM = "pair_add_lutpair4" *)
LUT4 #(.INIT(16'h0660)) di_bit5 (
    .I0(magnitude1[4]), .I1(sign1),
    .I2(magnitude0[4]), .I3(sign0), .O(carry_di[5])
);
(* HLUTNM = "pair_add_lutpair5" *)
LUT4 #(.INIT(16'h0660)) di_bit6 (
    .I0(magnitude1[5]), .I1(sign1),
    .I2(magnitude0[5]), .I3(sign0), .O(carry_di[6])
);

(* HLUTNM = "pair_add_lutpair0" *)
LUT2 #(.INIT(4'h6)) sum_bit0 (
    .I0(magnitude1[0]), .I1(magnitude0[0]), .O(carry_s[0])
);
(* HLUTNM = "pair_add_lutpair1" *)
LUT5 #(.INIT(32'h96696996)) sum_bit1 (
    .I0(magnitude1[1]), .I1(sign1),
    .I2(magnitude0[1]), .I3(sign0),
    .I4(carry_di[1]), .O(carry_s[1])
);
(* HLUTNM = "pair_add_lutpair2" *)
LUT5 #(.INIT(32'h96696996)) sum_bit2 (
    .I0(magnitude1[2]), .I1(sign1),
    .I2(magnitude0[2]), .I3(sign0),
    .I4(carry_di[2]), .O(carry_s[2])
);
(* HLUTNM = "pair_add_lutpair3" *)
LUT5 #(.INIT(32'h96696996)) sum_bit3 (
    .I0(magnitude1[3]), .I1(sign1),
    .I2(magnitude0[3]), .I3(sign0),
    .I4(carry_di[3]), .O(carry_s[3])
);
(* HLUTNM = "pair_add_lutpair4" *)
LUT5 #(.INIT(32'h96696996)) sum_bit4 (
    .I0(magnitude1[4]), .I1(sign1),
    .I2(magnitude0[4]), .I3(sign0),
    .I4(carry_di[4]), .O(carry_s[4])
);
(* HLUTNM = "pair_add_lutpair5" *)
LUT5 #(.INIT(32'h96696996)) sum_bit5 (
    .I0(carry_di[5]), .I1(sign1),
    .I2(magnitude1[5]), .I3(sign0),
    .I4(magnitude0[5]), .O(carry_s[5])
);
LUT4 #(.INIT(16'h1BD8)) sum_bit6 (
    .I0(magnitude0[5]), .I1(magnitude1[5]),
    .I2(sign1), .I3(sign0), .O(carry_s[6])
);
LUT2 #(.INIT(4'hE)) sum_bit7 (
    .I0(sign1), .I1(sign0), .O(carry_s[7])
);

CARRY8 pair_carry (
    .CI(1'b0), .CI_TOP(1'b0),
    .DI(carry_di), .S(carry_s),
    .CO(carry_co), .O(value)
);

endmodule

(* keep_hierarchy = "yes", use_dsp = "no" *)
module pair_add_xor_correction (
    input wire [5:0] magnitude0,
    input wire sign0,
    input wire [5:0] magnitude1,
    input wire sign1,
    output wire signed [7:0] value
);

wire [7:0] operand0 = {2'b0, magnitude0} ^ {8{sign0}};
wire [7:0] operand1 = {2'b0, magnitude1} ^ {8{sign1}};

assign value = operand0 + operand1 + sign0 + sign1;

endmodule

(* keep_hierarchy = "yes", use_dsp = "no" *)
module pair_add_lut6_2 (
    input wire [5:0] magnitude0,
    input wire sign0,
    input wire [5:0] magnitude1,
    input wire sign1,
    output wire signed [7:0] value
);

wire [7:0] carry_di;
wire [7:0] carry_s;
wire [7:0] carry_co;

assign carry_di[0] = sign0;
assign carry_di[7] = 1'b0;

LUT6_2 #(.INIT(64'h5A5A5A5A4EE44EE4)) bit0_and_di1 (
    .I0(magnitude1[0]), .I1(sign1),
    .I2(magnitude0[0]), .I3(sign0),
    .I4(1'b0), .I5(1'b1),
    .O5(carry_di[1]), .O6(carry_s[0])
);

genvar pair_bit;
generate
    for (pair_bit = 1; pair_bit < 6;
         pair_bit = pair_bit + 1) begin : bit_gen
        LUT6_2 #(.INIT(64'h9669699606600660)) bit_and_next_di (
            .I0(magnitude1[pair_bit]), .I1(sign1),
            .I2(magnitude0[pair_bit]), .I3(sign0),
            .I4(carry_di[pair_bit]), .I5(1'b1),
            .O5(carry_di[pair_bit + 1]), .O6(carry_s[pair_bit])
        );
    end
endgenerate

LUT4 #(.INIT(16'h1BD8)) sum_bit6 (
    .I0(magnitude0[5]), .I1(magnitude1[5]),
    .I2(sign1), .I3(sign0), .O(carry_s[6])
);
LUT2 #(.INIT(4'hE)) sum_bit7 (
    .I0(sign1), .I1(sign0), .O(carry_s[7])
);

CARRY8 pair_carry (
    .CI(1'b0), .CI_TOP(1'b0),
    .DI(carry_di), .S(carry_s),
    .CO(carry_co), .O(value)
);

endmodule

(* keep_hierarchy = "yes", use_dsp = "no" *)
module pair_add_same_sign (
    input wire [5:0] magnitude0,
    input wire sign0,
    input wire [5:0] magnitude1,
    input wire sign1,
    output reg signed [7:0] value
);

wire signed [7:0] magnitude_sum =
    $signed({2'b0, magnitude0}) + $signed({2'b0, magnitude1});
wire signed [7:0] magnitude0_minus_1 =
    $signed({2'b0, magnitude0}) - $signed({2'b0, magnitude1});
wire signed [7:0] magnitude1_minus_0 =
    $signed({2'b0, magnitude1}) - $signed({2'b0, magnitude0});

always @* begin
    case ({sign0, sign1})
        2'b00: value = magnitude_sum;
        2'b01: value = magnitude0_minus_1;
        2'b10: value = magnitude1_minus_0;
        default: value = -magnitude_sum;
    endcase
end

endmodule

(* keep_hierarchy = "yes", use_dsp = "no" *)
module pair_add_weighted_sign (
    input wire [5:0] magnitude0,
    input wire sign0,
    input wire [5:0] magnitude1,
    input wire sign1,
    output wire signed [7:0] value
);

wire signed [7:0] magnitude_sum =
    $signed({2'b0, magnitude0}) + $signed({2'b0, magnitude1});
wire signed [7:0] negative_correction0 =
    sign0 ? $signed({1'b0, magnitude0, 1'b0}) : 8'sd0;
wire signed [7:0] negative_correction1 =
    sign1 ? $signed({1'b0, magnitude1, 1'b0}) : 8'sd0;

assign value =
    magnitude_sum - negative_correction0 - negative_correction1;

endmodule

module test_pair_add_variants;
    reg [5:0] magnitude0;
    reg sign0;
    reg [5:0] magnitude1;
    reg sign1;
    wire signed [7:0] current_value;
    wire signed [7:0] explicit_carry_value;
    wire signed [7:0] lut6_2_value;
    wire signed [7:0] xor_value;
    wire signed [7:0] same_sign_value;
    wire signed [7:0] weighted_value;
    integer m0;
    integer m1;
    integer s0;
    integer s1;
    integer expected;

    pair_add_current current_dut (
        .magnitude0(magnitude0), .sign0(sign0),
        .magnitude1(magnitude1), .sign1(sign1), .value(current_value)
    );
    pair_add_explicit_carry explicit_carry_dut (
        .magnitude0(magnitude0), .sign0(sign0),
        .magnitude1(magnitude1), .sign1(sign1),
        .value(explicit_carry_value)
    );
    pair_add_lut6_2 lut6_2_dut (
        .magnitude0(magnitude0), .sign0(sign0),
        .magnitude1(magnitude1), .sign1(sign1), .value(lut6_2_value)
    );
    pair_add_xor_correction xor_dut (
        .magnitude0(magnitude0), .sign0(sign0),
        .magnitude1(magnitude1), .sign1(sign1), .value(xor_value)
    );
    pair_add_same_sign same_sign_dut (
        .magnitude0(magnitude0), .sign0(sign0),
        .magnitude1(magnitude1), .sign1(sign1), .value(same_sign_value)
    );
    pair_add_weighted_sign weighted_dut (
        .magnitude0(magnitude0), .sign0(sign0),
        .magnitude1(magnitude1), .sign1(sign1), .value(weighted_value)
    );

    initial begin
        for (s0 = 0; s0 < 2; s0 = s0 + 1)
            for (s1 = 0; s1 < 2; s1 = s1 + 1)
                for (m0 = 0; m0 < 50; m0 = m0 + 1)
                    for (m1 = 0; m1 < 50; m1 = m1 + 1) begin
                        magnitude0 = m0;
                        magnitude1 = m1;
                        sign0 = s0;
                        sign1 = s1;
                        expected = (s0 ? -m0 : m0) + (s1 ? -m1 : m1);
                        #1;
                        if ($signed(current_value) !== expected ||
                            $signed(explicit_carry_value) !== expected ||
                            $signed(lut6_2_value) !== expected ||
                            $signed(xor_value) !== expected ||
                            $signed(same_sign_value) !== expected ||
                            $signed(weighted_value) !== expected)
                            $fatal(1,
                                "FAIL m0=%0d s0=%0d m1=%0d s1=%0d expected=%0d values=%0d/%0d/%0d/%0d/%0d/%0d",
                                m0, s0, m1, s1, expected,
                                $signed(current_value),
                                $signed(explicit_carry_value),
                                $signed(lut6_2_value),
                                $signed(xor_value),
                                $signed(same_sign_value),
                                $signed(weighted_value));
                    end
        $display("PASS: pair-add variants are equivalent");
        $finish;
    end
endmodule
