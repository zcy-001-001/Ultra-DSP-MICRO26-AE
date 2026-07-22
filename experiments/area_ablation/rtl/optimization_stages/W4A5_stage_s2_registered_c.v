/*
 * Auto-generated for the Ultra-DSP rebuttal area ablation.
 * Top module: W4A5_stage_s2_registered_c
 * Precision: W4A5
 * Stage: s2_registered_c
 * Source: Ultra-DSP-main/DSP_Packaging_Verilog/W4A5/W4A5_Hybrid.v
 * Toggle summary: offline sign-magnitude weights; registered C-port; per-result two's-complement sign recovery
 */

module W4A5_stage_s2_registered_c(
    input wire clk,
    input wire mode,
    input wire [3:0] w1, w2, w3, w4, w5, w6,
    input wire [4:0] a1, a2,
    output wire [7:0] result1, result2, result3, result4, result5, result6, result7, result8,
    output wire sign1, sign2, sign3, sign4, sign5, sign6, sign7, sign8
);

wire [47:0] dsp_P;




wire w1_sign = w1[3], w2_sign = w2[3], w3_sign = w3[3], w4_sign = w4[3], w5_sign = w5[3], w6_sign = w6[3];
wire [2:0] w1_mag = w1[2:0], w2_mag = w2[2:0], w3_mag = w3[2:0], w4_mag = w4[2:0], w5_mag = w5[2:0], w6_mag = w6[2:0];




wire a1_sign = a1[4], a2_sign = a2[4];
wire [4:0] a1_tmp = a1[4] ? (~a1[4:0] + 1) : a1[4:0];
wire [4:0] a2_tmp = a2[4] ? (~a2[4:0] + 1) : a2[4:0];
wire [3:0] a1_mag = (a1_tmp == 5'b10000) ? 4'b1111 : a1_tmp[3:0];
wire [3:0] a2_mag = (a2_tmp == 5'b10000) ? 4'b1111 : a2_tmp[3:0];




wire [1:0] LSB_Res3, LSB_Res4, LSB_Res5, LSB_Res7, LSB_Res8;
wire [2:0] LSB_Res2, LSB_Res6;


wire c1_res2 = (w2_mag[0] & a1_mag[1]) & (w2_mag[1] & a1_mag[0]);
assign LSB_Res2 = {
    ((w2_mag[0] & a1_mag[2]) ^ (w2_mag[1] & a1_mag[1]) ^ (w2_mag[2] & a1_mag[0]) ^ c1_res2),
    ((w2_mag[0] & a1_mag[1]) ^ (w2_mag[1] & a1_mag[0])),
    (w2_mag[0] & a1_mag[0])
};


assign LSB_Res3 = {
    (w3_mag[0] & a1_mag[1]) ^ (w3_mag[1] & a1_mag[0]),
    w3_mag[0] & a1_mag[0]
};


assign LSB_Res4 = {
    (w4_mag[0] & a1_mag[1]) ^ (w4_mag[1] & a1_mag[0]),
    w4_mag[0] & a1_mag[0]
};


assign LSB_Res5 = (mode == 0) ? 2'b0 : {
    (w5_mag[0] & a1_mag[1]) ^ (w5_mag[1] & a1_mag[0]),
    w5_mag[0] & a1_mag[0]
};


wire c1_res6_P = (w2_mag[0] & a2_mag[1]) & (w2_mag[1] & a2_mag[0]);
wire c1_res6_D = (w6_mag[0] & a1_mag[1]) & (w6_mag[1] & a1_mag[0]);
assign LSB_Res6 = (mode == 0) ? {
    ((w2_mag[0] & a2_mag[2]) ^ (w2_mag[1] & a2_mag[1]) ^ (w2_mag[2] & a2_mag[0]) ^ c1_res6_P),
    ((w2_mag[0] & a2_mag[1]) ^ (w2_mag[1] & a2_mag[0])),
    (w2_mag[0] & a2_mag[0])
} : {
    ((w6_mag[0] & a1_mag[2]) ^ (w6_mag[1] & a1_mag[1]) ^ (w6_mag[2] & a1_mag[0]) ^ c1_res6_D),
    ((w6_mag[0] & a1_mag[1]) ^ (w6_mag[1] & a1_mag[0])),
    (w6_mag[0] & a1_mag[0])
};


assign LSB_Res7 = {
    (w3_mag[0] & a2_mag[1]) ^ (w3_mag[1] & a2_mag[0]),
    w3_mag[0] & a2_mag[0]
};


assign LSB_Res8 = {
    (w4_mag[0] & a2_mag[1]) ^ (w4_mag[1] & a2_mag[0]),
    w4_mag[0] & a2_mag[0]
};




wire [7:0] signs_comb = (mode == 0) ? {
    (w4_sign ^ a2_sign), (w3_sign ^ a2_sign), (w2_sign ^ a2_sign), (w1_sign ^ a2_sign),
    (w4_sign ^ a1_sign), (w3_sign ^ a1_sign), (w2_sign ^ a1_sign), (w1_sign ^ a1_sign)
} : {
    2'b0,
    (w6_sign ^ a1_sign), (w5_sign ^ a1_sign), (w4_sign ^ a1_sign),
    (w3_sign ^ a1_sign), (w2_sign ^ a1_sign), (w1_sign ^ a1_sign)
};




wire [26:0] dsp_A = (mode == 0) ?
    {2'b0, a2_mag, 17'b0, a1_mag} :
    {1'b0, w6_mag, 1'b0, w5_mag, 2'b0, w4_mag, 2'b0, w3_mag, 2'b0, w2_mag, 1'b0, w1_mag};

wire [17:0] dsp_B = (mode == 0) ?
    {1'b0, w4_mag, 2'b0, w3_mag, 2'b0, w2_mag, 1'b0, w1_mag} :
    {14'b0, a1_mag};




reg mode_r1, mode_r2, mode_r3;
reg [7:0] signs_r1, signs_r2, signs_r3;
reg [1:0] LSB_Res3_r1, LSB_Res4_r1, LSB_Res5_r1, LSB_Res7_r1, LSB_Res8_r1;
reg [2:0] LSB_Res2_r1, LSB_Res6_r1;
reg [1:0] LSB_Res3_r2, LSB_Res4_r2, LSB_Res5_r2, LSB_Res7_r2, LSB_Res8_r2;
reg [2:0] LSB_Res2_r2, LSB_Res6_r2;
reg [1:0] LSB_Res3_r3, LSB_Res4_r3, LSB_Res5_r3, LSB_Res7_r3, LSB_Res8_r3;
reg [2:0] LSB_Res2_r3, LSB_Res6_r3;

wire [47:0] dsp_C = (mode_r1 == 0) ? {
    11'b0, LSB_Res8_r1, 3'b0, LSB_Res7_r1, 2'b0, LSB_Res6_r1, 9'b0,
    LSB_Res4_r1, 3'b0, LSB_Res3_r1, 2'b0, LSB_Res2_r1, 4'b0
} : {
    22'b0, LSB_Res6_r1, 2'b0, LSB_Res5_r1, 3'b0, LSB_Res4_r1,
    3'b0, LSB_Res3_r1, 2'b0, LSB_Res2_r1, 4'b0
};

always @(posedge clk) begin
    mode_r1 <= mode;
    signs_r1 <= signs_comb;
    LSB_Res2_r1 <= LSB_Res2;
    LSB_Res3_r1 <= LSB_Res3;
    LSB_Res4_r1 <= LSB_Res4;
    LSB_Res5_r1 <= LSB_Res5;
    LSB_Res6_r1 <= LSB_Res6;
    LSB_Res7_r1 <= LSB_Res7;
    LSB_Res8_r1 <= LSB_Res8;
end

always @(posedge clk) begin
    mode_r2 <= mode_r1;
    signs_r2 <= signs_r1;
    LSB_Res2_r2 <= LSB_Res2_r1;
    LSB_Res3_r2 <= LSB_Res3_r1;
    LSB_Res4_r2 <= LSB_Res4_r1;
    LSB_Res5_r2 <= LSB_Res5_r1;
    LSB_Res6_r2 <= LSB_Res6_r1;
    LSB_Res7_r2 <= LSB_Res7_r1;
    LSB_Res8_r2 <= LSB_Res8_r1;
end

always @(posedge clk) begin
    mode_r3 <= mode_r2;
    signs_r3 <= signs_r2;
    LSB_Res2_r3 <= LSB_Res2_r2;
    LSB_Res3_r3 <= LSB_Res3_r2;
    LSB_Res4_r3 <= LSB_Res4_r2;
    LSB_Res5_r3 <= LSB_Res5_r2;
    LSB_Res6_r3 <= LSB_Res6_r2;
    LSB_Res7_r3 <= LSB_Res7_r2;
    LSB_Res8_r3 <= LSB_Res8_r2;
end




DSP48E2 #(
    .AMULTSEL("A"), .A_INPUT("DIRECT"), .BMULTSEL("B"), .B_INPUT("DIRECT"), .PREADDINSEL("A"),
    .RND(48'h000000000000), .USE_MULT("MULTIPLY"), .USE_SIMD("ONE48"), .USE_WIDEXOR("FALSE"), .XORSIMD("XOR24_48_96"),
    .AUTORESET_PATDET("NO_RESET"), .AUTORESET_PRIORITY("RESET"), .MASK(48'h3fffffffffff), .PATTERN(48'h000000000000),
    .SEL_MASK("MASK"), .SEL_PATTERN("PATTERN"), .USE_PATTERN_DETECT("NO_PATDET"),
    .IS_ALUMODE_INVERTED(4'b0000), .IS_CARRYIN_INVERTED(1'b0), .IS_CLK_INVERTED(1'b0), .IS_INMODE_INVERTED(5'b00000),
    .IS_OPMODE_INVERTED(9'b000000000), .IS_RSTALLCARRYIN_INVERTED(1'b0), .IS_RSTALUMODE_INVERTED(1'b0),
    .IS_RSTA_INVERTED(1'b0), .IS_RSTB_INVERTED(1'b0), .IS_RSTCTRL_INVERTED(1'b0), .IS_RSTC_INVERTED(1'b0),
    .IS_RSTD_INVERTED(1'b0), .IS_RSTINMODE_INVERTED(1'b0), .IS_RSTM_INVERTED(1'b0), .IS_RSTP_INVERTED(1'b0),
    .ACASCREG(1), .ADREG(0), .ALUMODEREG(1), .AREG(1), .BCASCREG(1), .BREG(1), .CARRYINREG(1),
    .CARRYINSELREG(1), .CREG(1), .DREG(0), .INMODEREG(1), .MREG(1), .OPMODEREG(1), .PREG(1)
) dsp_inst (
    .ACOUT(), .BCOUT(), .CARRYCASCOUT(), .MULTSIGNOUT(), .PCOUT(), .OVERFLOW(), .PATTERNBDETECT(), .PATTERNDETECT(), .UNDERFLOW(),
    .CARRYOUT(), .P(dsp_P), .XOROUT(),
    .ACIN(30'b0), .BCIN(18'b0), .CARRYCASCIN(1'b0), .MULTSIGNIN(1'b0), .PCIN(48'b0),
    .ALUMODE(4'b0001), .CARRYINSEL(3'b000), .CLK(clk), .INMODE(5'b00000), .OPMODE(9'b000110101),
    .A(dsp_A), .B(dsp_B), .C(dsp_C), .CARRYIN(1'b1), .D(27'b0),
    .CEA1(1'b0), .CEA2(1'b1), .CEAD(1'b0), .CEALUMODE(1'b1), .CEB1(1'b0), .CEB2(1'b1), .CEC(1'b1),
    .CECARRYIN(1'b1), .CECTRL(1'b1), .CED(1'b0), .CEINMODE(1'b1), .CEM(1'b1), .CEP(1'b1),
    .RSTA(1'b0), .RSTALLCARRYIN(1'b0), .RSTALUMODE(1'b0), .RSTB(1'b0), .RSTC(1'b0), .RSTCTRL(1'b0),
    .RSTD(1'b0), .RSTINMODE(1'b0), .RSTM(1'b0), .RSTP(1'b0)
);




wire [6:0] res1_mag, res2_mag, res3_mag, res4_mag, res5_mag, res6_mag, res7_mag, res8_mag;

assign res1_mag = dsp_P[6:0];
assign res2_mag = {dsp_P[10:7], LSB_Res2_r3};
assign res3_mag = {dsp_P[15:11], LSB_Res3_r3};
assign res4_mag = {dsp_P[20:16], LSB_Res4_r3};
assign res5_mag = (mode_r3 == 0) ? dsp_P[27:21] : {dsp_P[25:21], LSB_Res5_r3};
assign res6_mag = (mode_r3 == 0) ?{dsp_P[31:28], LSB_Res6_r3} : {dsp_P[29:26], LSB_Res6_r3};
assign res7_mag = (mode_r3 == 0) ? {dsp_P[36:32], LSB_Res7_r3} : 7'b0;
assign res8_mag = (mode_r3 == 0) ? {dsp_P[41:37], LSB_Res8_r3} : 7'b0;




assign result1 = (signs_r3[0] ? (~{1'b0, res1_mag} + 8'b00000001) : {1'b0, res1_mag});
assign result2 = (signs_r3[1] ? (~{1'b0, res2_mag} + 8'b00000001) : {1'b0, res2_mag});
assign result3 = (signs_r3[2] ? (~{1'b0, res3_mag} + 8'b00000001) : {1'b0, res3_mag});
assign result4 = (signs_r3[3] ? (~{1'b0, res4_mag} + 8'b00000001) : {1'b0, res4_mag});
assign result5 = (signs_r3[4] ? (~{1'b0, res5_mag} + 8'b00000001) : {1'b0, res5_mag});
assign result6 = (signs_r3[5] ? (~{1'b0, res6_mag} + 8'b00000001) : {1'b0, res6_mag});
assign result7 = (mode_r3 == 0) ? ((signs_r3[6] ? (~{1'b0, res7_mag} + 8'b00000001) : {1'b0, res7_mag})) : 8'b0;
assign result8 = (mode_r3 == 0) ? ((signs_r3[7] ? (~{1'b0, res8_mag} + 8'b00000001) : {1'b0, res8_mag})) : 8'b0;

assign sign1 = signs_r3[0];
assign sign2 = signs_r3[1];
assign sign3 = signs_r3[2];
assign sign4 = signs_r3[3];
assign sign5 = signs_r3[4];
assign sign6 = signs_r3[5];
assign sign7 = signs_r3[6];
assign sign8 = signs_r3[7];

endmodule
