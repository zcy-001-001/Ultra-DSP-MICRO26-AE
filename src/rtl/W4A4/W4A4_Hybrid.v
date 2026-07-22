
module Hybrid_INT4_INT4_PD(
    
    input wire        clk,
    input wire        mode,
    
    
    input wire [3:0]  w1, w2, w3, w4, w5, w6, w7,
    
    
    input wire [3:0]  a1, a2, a3,

    
    output wire [6:0] result1, result2, result3, result4, result5, result6, result7, result8, result9,
    
    
    output wire       sign1, sign2, sign3, sign4, sign5, sign6, sign7, sign8, sign9
);

wire [47:0] dsp_P;




wire w1_sign, w2_sign, w3_sign, w4_sign, w5_sign, w6_sign, w7_sign;
wire [2:0] w1_mag, w2_mag, w3_mag, w4_mag, w5_mag, w6_mag, w7_mag;

assign w1_sign = w1[3]; assign w1_mag = w1[2:0];
assign w2_sign = w2[3]; assign w2_mag = w2[2:0];
assign w3_sign = w3[3]; assign w3_mag = w3[2:0];
assign w4_sign = w4[3]; assign w4_mag = w4[2:0];
assign w5_sign = w5[3]; assign w5_mag = w5[2:0];
assign w6_sign = w6[3]; assign w6_mag = w6[2:0];
assign w7_sign = w7[3]; assign w7_mag = w7[2:0];

wire a1_sign, a2_sign, a3_sign;
wire [2:0] a1_mag, a2_mag, a3_mag;
wire [3:0] a1_temp_mag, a2_temp_mag, a3_temp_mag;

assign a1_sign = a1[3];
assign a1_temp_mag = a1[3] ? (~a1[3:0] + 1'b1) : a1[3:0];
assign a1_mag = (a1_temp_mag == 4'b1000) ? 3'b111 : a1_temp_mag[2:0];

assign a2_sign = a2[3];
assign a2_temp_mag = a2[3] ? (~a2[3:0] + 1'b1) : a2[3:0];
assign a2_mag = (a2_temp_mag == 4'b1000) ? 3'b111 : a2_temp_mag[2:0];

assign a3_sign = a3[3];
assign a3_temp_mag = a3[3] ? (~a3[3:0] + 1'b1) : a3[3:0];
assign a3_mag = (a3_temp_mag == 4'b1000) ? 3'b111 : a3_temp_mag[2:0];




wire [1:0] LSB_Res2, LSB_Res3, LSB_Res5, LSB_Res6, LSB_Res7, LSB_Res8, LSB_Res9;
wire [2:0] LSB_Res4_P, LSB_Res4_D;


assign LSB_Res2[0] = (mode == 0) ? (w1_mag[0] & a2_mag[0]) : (w2_mag[0] & a1_mag[0]);
assign LSB_Res2[1] = (mode == 0) ? ((w1_mag[0] & a2_mag[1]) ^ (w1_mag[1] & a2_mag[0])) : 
                                    ((w2_mag[0] & a1_mag[1]) ^ (w2_mag[1] & a1_mag[0]));


assign LSB_Res3[0] = (mode == 0) ? (w1_mag[0] & a3_mag[0]) : (w3_mag[0] & a1_mag[0]);
assign LSB_Res3[1] = (mode == 0) ? ((w1_mag[0] & a3_mag[1]) ^ (w1_mag[1] & a3_mag[0])) : 
                                    ((w3_mag[0] & a1_mag[1]) ^ (w3_mag[1] & a1_mag[0]));


wire c1_lsb4_P, c1_lsb4_D;
assign LSB_Res4_P[0] = w2_mag[0] & a1_mag[0];
assign LSB_Res4_P[1] = (w2_mag[0] & a1_mag[1]) ^ (w2_mag[1] & a1_mag[0]);
assign c1_lsb4_P = (w2_mag[0] & a1_mag[1]) & (w2_mag[1] & a1_mag[0]);
assign LSB_Res4_P[2] = (w2_mag[0] & a1_mag[2]) ^ (w2_mag[1] & a1_mag[1]) ^ (w2_mag[2] & a1_mag[0]) ^ c1_lsb4_P;

assign LSB_Res4_D[0] = w4_mag[0] & a1_mag[0];
assign LSB_Res4_D[1] = (w4_mag[0] & a1_mag[1]) ^ (w4_mag[1] & a1_mag[0]);
assign c1_lsb4_D = (w4_mag[0] & a1_mag[1]) & (w4_mag[1] & a1_mag[0]);
assign LSB_Res4_D[2] = (w4_mag[0] & a1_mag[2]) ^ (w4_mag[1] & a1_mag[1]) ^ (w4_mag[2] & a1_mag[0]) ^ c1_lsb4_D;


assign LSB_Res5[0] = (mode == 0) ? (w2_mag[0] & a2_mag[0]) : (w5_mag[0] & a1_mag[0]);
assign LSB_Res5[1] = (mode == 0) ? ((w2_mag[0] & a2_mag[1]) ^ (w2_mag[1] & a2_mag[0])) : 
                                    ((w5_mag[0] & a1_mag[1]) ^ (w5_mag[1] & a1_mag[0]));


assign LSB_Res6[0] = (mode == 0) ? (w2_mag[0] & a3_mag[0]) : (w6_mag[0] & a1_mag[0]);
assign LSB_Res6[1] = (mode == 0) ? ((w2_mag[0] & a3_mag[1]) ^ (w2_mag[1] & a3_mag[0])) : 
                                    ((w6_mag[0] & a1_mag[1]) ^ (w6_mag[1] & a1_mag[0]));


assign LSB_Res7[0] = (mode == 0) ? (w3_mag[0] & a1_mag[0]) : (w7_mag[0] & a1_mag[0]);
assign LSB_Res7[1] = (mode == 0) ? ((w3_mag[0] & a1_mag[1]) ^ (w3_mag[1] & a1_mag[0])) : 
                                    ((w7_mag[0] & a1_mag[1]) ^ (w7_mag[1] & a1_mag[0]));


assign LSB_Res8[0] = w3_mag[0] & a2_mag[0];
assign LSB_Res8[1] = (w3_mag[0] & a2_mag[1]) ^ (w3_mag[1] & a2_mag[0]);


assign LSB_Res9[0] = w3_mag[0] & a3_mag[0];
assign LSB_Res9[1] = (w3_mag[0] & a3_mag[1]) ^ (w3_mag[1] & a3_mag[0]);




wire [8:0] signs_comb = (mode == 0) ? {
    (w3_sign ^ a3_sign), (w3_sign ^ a2_sign), (w3_sign ^ a1_sign),
    (w2_sign ^ a3_sign), (w2_sign ^ a2_sign), (w2_sign ^ a1_sign),
    (w1_sign ^ a3_sign), (w1_sign ^ a2_sign), (w1_sign ^ a1_sign)
} : {
    2'b0,
    (w7_sign ^ a1_sign), (w6_sign ^ a1_sign), (w5_sign ^ a1_sign),
    (w4_sign ^ a1_sign), (w3_sign ^ a1_sign), (w2_sign ^ a1_sign), (w1_sign ^ a1_sign)
};




wire [26:0] dsp_A = (mode == 0) ? 
    {1'b0, w3_mag, 9'b0, w2_mag,8'b0, w1_mag} :
    {1'b0, w7_mag, 1'b0, w6_mag, 1'b0, w5_mag, 1'b0, w4_mag, w3_mag, 1'b0, w2_mag, 1'b0, w1_mag};

wire [17:0] dsp_B = (mode == 0) ?
    {7'b0, a3_mag, 1'b0, a2_mag, 1'b0, a1_mag} :
    {15'b0, a1_mag};




reg mode_r1, mode_r2, mode_r3;
reg [8:0] signs_r1, signs_r2, signs_r3;
reg [1:0] LSB_Res2_r1, LSB_Res3_r1, LSB_Res5_r1, LSB_Res6_r1, LSB_Res7_r1, LSB_Res8_r1, LSB_Res9_r1;
reg [2:0] LSB_Res4_r1;
reg [1:0] LSB_Res2_r2, LSB_Res3_r2, LSB_Res5_r2, LSB_Res6_r2, LSB_Res7_r2, LSB_Res8_r2, LSB_Res9_r2;
reg [2:0] LSB_Res4_r2;
reg [1:0] LSB_Res2_r3, LSB_Res3_r3, LSB_Res5_r3, LSB_Res6_r3, LSB_Res7_r3, LSB_Res8_r3, LSB_Res9_r3;
reg [2:0] LSB_Res4_r3;

wire [47:0] dsp_C = (mode_r1 == 0) ? {
    
    15'b0, LSB_Res9_r1, 2'b0, LSB_Res8_r1, 2'b0, LSB_Res7_r1, 2'b0,
    LSB_Res6_r1, 2'b0, LSB_Res5_r1, 1'b0, LSB_Res4_r1, 1'b0,
    LSB_Res3_r1, 2'b0, LSB_Res2_r1, 4'b0
} : {
    
    23'b0, LSB_Res7_r1, 2'b0, LSB_Res6_r1, 2'b0, LSB_Res5_r1, 1'b0,
    LSB_Res4_r1, 1'b0, LSB_Res3_r1, 2'b0, LSB_Res2_r1, 4'b0
};

always @(posedge clk) begin
    mode_r1 <= mode; signs_r1 <= signs_comb;
    LSB_Res2_r1 <= LSB_Res2; LSB_Res3_r1 <= LSB_Res3;
    LSB_Res4_r1 <= (mode == 0) ? LSB_Res4_P : LSB_Res4_D;
    LSB_Res5_r1 <= LSB_Res5; LSB_Res6_r1 <= LSB_Res6;
    LSB_Res7_r1 <= LSB_Res7; LSB_Res8_r1 <= LSB_Res8; LSB_Res9_r1 <= LSB_Res9;
end

always @(posedge clk) begin
    mode_r2 <= mode_r1; signs_r2 <= signs_r1;
    LSB_Res2_r2 <= LSB_Res2_r1; LSB_Res3_r2 <= LSB_Res3_r1; LSB_Res4_r2 <= LSB_Res4_r1;
    LSB_Res5_r2 <= LSB_Res5_r1; LSB_Res6_r2 <= LSB_Res6_r1; LSB_Res7_r2 <= LSB_Res7_r1;
    LSB_Res8_r2 <= LSB_Res8_r1; LSB_Res9_r2 <= LSB_Res9_r1;
end

always @(posedge clk) begin
    mode_r3 <= mode_r2; signs_r3 <= signs_r2;
    LSB_Res2_r3 <= LSB_Res2_r2; LSB_Res3_r3 <= LSB_Res3_r2; LSB_Res4_r3 <= LSB_Res4_r2;
    LSB_Res5_r3 <= LSB_Res5_r2; LSB_Res6_r3 <= LSB_Res6_r2; LSB_Res7_r3 <= LSB_Res7_r2;
    LSB_Res8_r3 <= LSB_Res8_r2; LSB_Res9_r3 <= LSB_Res9_r2;
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




wire [5:0] res1_mag, res2_mag, res3_mag, res4_mag, res5_mag, res6_mag, res7_mag, res8_mag, res9_mag;

assign res1_mag = dsp_P[5:0];
assign res2_mag = {dsp_P[9:6], LSB_Res2_r3};
assign res3_mag = {dsp_P[13:10], LSB_Res3_r3};
assign res4_mag = {dsp_P[16:14], LSB_Res4_r3};
assign res5_mag = {dsp_P[20:17], LSB_Res5_r3};
assign res6_mag = {dsp_P[24:21], LSB_Res6_r3};
assign res7_mag = {dsp_P[28:25], LSB_Res7_r3};
assign res8_mag = (mode_r3 == 0) ? {dsp_P[32:29], LSB_Res8_r3} : 6'b0;
assign res9_mag = (mode_r3 == 0) ? {dsp_P[36:33], LSB_Res9_r3} : 6'b0;




assign result1 = {1'b0, res1_mag} ^ {7{signs_r3[0]}};
assign result2 = {1'b0, res2_mag} ^ {7{signs_r3[1]}};
assign result3 = {1'b0, res3_mag} ^ {7{signs_r3[2]}};
assign result4 = {1'b0, res4_mag} ^ {7{signs_r3[3]}};
assign result5 = {1'b0, res5_mag} ^ {7{signs_r3[4]}};
assign result6 = {1'b0, res6_mag} ^ {7{signs_r3[5]}};
assign result7 = {1'b0, res7_mag} ^ {7{signs_r3[6]}};
assign result8 = (mode_r3 == 0) ? ({1'b0, res8_mag} ^ {7{signs_r3[7]}}) : 7'b0;
assign result9 = (mode_r3 == 0) ? ({1'b0, res9_mag} ^ {7{signs_r3[8]}}) : 7'b0;

assign {sign9, sign8, sign7, sign6, sign5, sign4, sign3, sign2, sign1} = signs_r3;

endmodule
