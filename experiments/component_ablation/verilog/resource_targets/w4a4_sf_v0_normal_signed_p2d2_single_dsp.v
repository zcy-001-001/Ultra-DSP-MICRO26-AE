// V0 resource target: normal signed two's-complement packing.
//
// This is the corrected Normal Signed Packing baseline.  It does not split
// operands into sign and magnitude.  The DSP sees two's-complement signed
// weights and a two's-complement signed activation value.
//
// P mode (mode=0):
//   B  = a1 + a2, using a small fabric pre-adder before the DSP B port.
//   A  = sign-extended w1 at offset 0.
//   D  = sign-extended w2 shifted by 8 bits.
//   Outputs are w1*(a1+a2) and w2*(a1+a2).  These are two physical
//   output lanes but four effective MAC contributions.
//
// D mode (mode=1):
//   B  = a1.
//   A/D are unchanged, so the DSP returns w1*a1 and w2*a1.
//
// The high lane needs one bit of correction when the low lane is negative,
// because an arithmetic packed product sign-extends the low 8-bit field into
// the next field.  This is the normal signed anti-pollution logic; it is not
// sign-magnitude decoupling.
module w4a4_sf_v0_normal_signed_p2d2_single_dsp(
    input  wire        clk,
    input  wire        mode,
    input  wire [3:0]  w1, w2, w3, w4, w5, w6, w7,
    input  wire [3:0]  a1, a2, a3,
    output wire signed [15:0] result1,
    output wire signed [15:0] result2,
    output wire signed [15:0] result3,
    output wire signed [15:0] result4,
    output wire signed [15:0] result5,
    output wire signed [15:0] result6,
    output wire signed [15:0] result7,
    output wire signed [15:0] result8,
    output wire signed [15:0] result9,
    output wire signed [15:0] result10,
    output wire [3:0]  valid_count
);

wire signed [26:0] w1_ext = {{23{w1[3]}}, w1};
wire signed [26:0] w2_ext = {{23{w2[3]}}, w2};
wire signed [26:0] w2_shifted = w2_ext <<< 8;

wire signed [4:0] a1_ext = {a1[3], a1};
wire signed [4:0] a2_ext = {a2[3], a2};
wire signed [5:0] a_sum_prefill = a1_ext + a2_ext;

wire signed [17:0] b_prefill = {{12{a_sum_prefill[5]}}, a_sum_prefill};
wire signed [17:0] b_decode  = {{14{a1[3]}}, a1};
wire signed [17:0] dsp_b_signed = mode ? b_decode : b_prefill;

wire [29:0] dsp_A = {{3{w1_ext[26]}}, w1_ext};
wire [26:0] dsp_D = w2_shifted;
wire [17:0] dsp_B = dsp_b_signed;
wire [47:0] dsp_P;

reg [3:0] valid_count_r1, valid_count_r2, valid_count_r3;

always @(posedge clk) begin
    valid_count_r1 <= 4'd2;
    valid_count_r2 <= valid_count_r1;
    valid_count_r3 <= valid_count_r2;
end

DSP48E2 #(
    .AMULTSEL("AD"),
    .A_INPUT("DIRECT"),
    .BMULTSEL("B"),
    .B_INPUT("DIRECT"),
    .PREADDINSEL("A"),
    .RND(48'h000000000000),
    .USE_MULT("MULTIPLY"),
    .USE_SIMD("ONE48"),
    .USE_WIDEXOR("FALSE"),
    .XORSIMD("XOR24_48_96"),
    .AUTORESET_PATDET("NO_RESET"),
    .AUTORESET_PRIORITY("RESET"),
    .MASK(48'h3fffffffffff),
    .PATTERN(48'h000000000000),
    .SEL_MASK("MASK"),
    .SEL_PATTERN("PATTERN"),
    .USE_PATTERN_DETECT("NO_PATDET"),
    .IS_ALUMODE_INVERTED(4'b0000),
    .IS_CARRYIN_INVERTED(1'b0),
    .IS_CLK_INVERTED(1'b0),
    .IS_INMODE_INVERTED(5'b00000),
    .IS_OPMODE_INVERTED(9'b000000000),
    .IS_RSTALLCARRYIN_INVERTED(1'b0),
    .IS_RSTALUMODE_INVERTED(1'b0),
    .IS_RSTA_INVERTED(1'b0),
    .IS_RSTB_INVERTED(1'b0),
    .IS_RSTCTRL_INVERTED(1'b0),
    .IS_RSTC_INVERTED(1'b0),
    .IS_RSTD_INVERTED(1'b0),
    .IS_RSTINMODE_INVERTED(1'b0),
    .IS_RSTM_INVERTED(1'b0),
    .IS_RSTP_INVERTED(1'b0),
    .ACASCREG(0),
    .ADREG(1),
    .ALUMODEREG(1),
    .AREG(1),
    .BCASCREG(0),
    .BREG(1),
    .CARRYINREG(1),
    .CARRYINSELREG(1),
    .CREG(0),
    .DREG(1),
    .INMODEREG(1),
    .MREG(1),
    .OPMODEREG(1),
    .PREG(1)
) dsp_inst (
    .ACOUT(),
    .BCOUT(),
    .CARRYCASCOUT(),
    .MULTSIGNOUT(),
    .PCOUT(),
    .OVERFLOW(),
    .PATTERNBDETECT(),
    .PATTERNDETECT(),
    .UNDERFLOW(),
    .CARRYOUT(),
    .P(dsp_P),
    .XOROUT(),
    .ACIN(30'b0),
    .BCIN(18'b0),
    .CARRYCASCIN(1'b0),
    .MULTSIGNIN(1'b0),
    .PCIN(48'b0),
    .ALUMODE(4'b0000),
    .CARRYINSEL(3'b000),
    .CLK(clk),
    .INMODE(5'b00100),
    .OPMODE(9'b000000101),
    .A(dsp_A),
    .B(dsp_B),
    .C(48'b0),
    .CARRYIN(1'b0),
    .D(dsp_D),
    .CEA1(1'b0),
    .CEA2(1'b1),
    .CEAD(1'b1),
    .CEALUMODE(1'b1),
    .CEB1(1'b0),
    .CEB2(1'b1),
    .CEC(1'b0),
    .CECARRYIN(1'b1),
    .CECTRL(1'b1),
    .CED(1'b1),
    .CEINMODE(1'b1),
    .CEM(1'b1),
    .CEP(1'b1),
    .RSTA(1'b0),
    .RSTALLCARRYIN(1'b0),
    .RSTALUMODE(1'b0),
    .RSTB(1'b0),
    .RSTC(1'b0),
    .RSTCTRL(1'b0),
    .RSTD(1'b0),
    .RSTINMODE(1'b0),
    .RSTM(1'b0),
    .RSTP(1'b0)
);

wire signed [7:0] lane0 = dsp_P[7:0];
wire signed [7:0] lane1_polluted = dsp_P[15:8];
wire signed [8:0] lane0_sign_correction = lane0[7] ? 9'sd1 : 9'sd0;
wire signed [8:0] lane1_corrected = {lane1_polluted[7], lane1_polluted} + lane0_sign_correction;

assign result1 = {{8{lane0[7]}}, lane0};
assign result2 = {{8{lane1_corrected[7]}}, lane1_corrected[7:0]};
assign result3 = 16'sd0;
assign result4 = 16'sd0;
assign result5 = 16'sd0;
assign result6 = 16'sd0;
assign result7 = 16'sd0;
assign result8 = 16'sd0;
assign result9 = 16'sd0;
assign result10 = 16'sd0;
assign valid_count = valid_count_r3;

endmodule
