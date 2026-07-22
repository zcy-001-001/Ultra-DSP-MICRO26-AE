/*
 * Auto-generated for the Ultra-DSP rebuttal area ablation.
 * Top module: W3A5_stage_s2_registered_c
 * Precision: W3A5
 * Stage: s2_registered_c
 * Source: Ultra-DSP-main/DSP_Packaging_Verilog/W3A5/W3A5_Hybrid.v
 * Toggle summary: offline sign-magnitude weights; registered C-port; per-result two's-complement sign recovery
 */

module W3A5_stage_s2_registered_c(
    input wire        clk,
    input wire        mode,
    input wire [2:0]  w1, w2, w3, w4, w5, w6, w7,
    input wire [4:0]  a1, a2,
    output wire [6:0] result1, result2, result3, result4, result5, result6, 
                      result7, result8, result9, result10,
    output wire       sign1, sign2, sign3, sign4, sign5, sign6,
                      sign7, sign8, sign9, sign10
);

wire [47:0] dsp_P;


wire w1_sign, w2_sign, w3_sign, w4_sign, w5_sign, w6_sign, w7_sign;
wire [1:0] w1_mag, w2_mag, w3_mag, w4_mag, w5_mag, w6_mag, w7_mag;
assign w1_sign = w1[2]; assign w1_mag = w1[1:0];
assign w2_sign = w2[2]; assign w2_mag = w2[1:0];
assign w3_sign = w3[2]; assign w3_mag = w3[1:0];
assign w4_sign = w4[2]; assign w4_mag = w4[1:0];
assign w5_sign = w5[2]; assign w5_mag = w5[1:0];
assign w6_sign = w6[2]; assign w6_mag = w6[1:0];
assign w7_sign = w7[2]; assign w7_mag = w7[1:0];


wire a1_sign, a2_sign;
wire [3:0] a1_mag, a2_mag;
wire [4:0] a1_temp_mag, a2_temp_mag;
assign a1_sign = a1[4];
assign a1_temp_mag = a1[4] ? (~a1[4:0] + 1'b1) : a1[4:0];
assign a1_mag = (a1_temp_mag == 5'b10000) ? 4'b1111 : a1_temp_mag[3:0];
assign a2_sign = a2[4];
assign a2_temp_mag = a2[4] ? (~a2[4:0] + 1'b1) : a2[4:0];
assign a2_mag = (a2_temp_mag == 5'b10000) ? 4'b1111 : a2_temp_mag[3:0];



wire [1:0] LSB_Res2_P = {(w2_mag[0] & a1_mag[1]) ^ (w2_mag[1] & a1_mag[0]), w2_mag[0] & a1_mag[0]};

wire [1:0] LSB_Res3_P = {(w3_mag[0] & a1_mag[1]) ^ (w3_mag[1] & a1_mag[0]), w3_mag[0] & a1_mag[0]};

wire [1:0] LSB_Res4_P = {(w4_mag[0] & a1_mag[1]) ^ (w4_mag[1] & a1_mag[0]), w4_mag[0] & a1_mag[0]};

wire c1_lsb5_P = (w5_mag[0] & a1_mag[1]) & (w5_mag[1] & a1_mag[0]);
wire [2:0] LSB_Res5_P = {(w5_mag[0] & a1_mag[2]) ^ (w5_mag[1] & a1_mag[1]) ^ c1_lsb5_P, 
                         (w5_mag[0] & a1_mag[1]) ^ (w5_mag[1] & a1_mag[0]), 
                         w5_mag[0] & a1_mag[0]};

wire [1:0] LSB_Res7_P = {(w2_mag[0] & a2_mag[1]) ^ (w2_mag[1] & a2_mag[0]), w2_mag[0] & a2_mag[0]};

wire [1:0] LSB_Res8_P = {(w3_mag[0] & a2_mag[1]) ^ (w3_mag[1] & a2_mag[0]), w3_mag[0] & a2_mag[0]};

wire [1:0] LSB_Res9_P = {(w4_mag[0] & a2_mag[1]) ^ (w4_mag[1] & a2_mag[0]), w4_mag[0] & a2_mag[0]};

wire c1_lsb10_P = (w5_mag[0] & a2_mag[1]) & (w5_mag[1] & a2_mag[0]);
wire [2:0] LSB_Res10_P = {(w5_mag[0] & a2_mag[2]) ^ (w5_mag[1] & a2_mag[1]) ^ c1_lsb10_P, 
                          (w5_mag[0] & a2_mag[1]) ^ (w5_mag[1] & a2_mag[0]), 
                          w5_mag[0] & a2_mag[0]};


wire [1:0] LSB_Res2_D = {(w2_mag[0] & a1_mag[1]) ^ (w2_mag[1] & a1_mag[0]), w2_mag[0] & a1_mag[0]};
wire [1:0] LSB_Res3_D = {(w3_mag[0] & a1_mag[1]) ^ (w3_mag[1] & a1_mag[0]), w3_mag[0] & a1_mag[0]};
wire [1:0] LSB_Res4_D = {(w4_mag[0] & a1_mag[1]) ^ (w4_mag[1] & a1_mag[0]), w4_mag[0] & a1_mag[0]};
wire [1:0] LSB_Res5_D = {(w5_mag[0] & a1_mag[1]) ^ (w5_mag[1] & a1_mag[0]), w5_mag[0] & a1_mag[0]};
wire [1:0] LSB_Res6_D = {(w6_mag[0] & a1_mag[1]) ^ (w6_mag[1] & a1_mag[0]), w6_mag[0] & a1_mag[0]};
wire [1:0] LSB_Res7_D = {(w7_mag[0] & a1_mag[1]) ^ (w7_mag[1] & a1_mag[0]), w7_mag[0] & a1_mag[0]};


wire [1:0] LSB_Res2 = (mode == 0) ? LSB_Res2_P : LSB_Res2_D;
wire [1:0] LSB_Res3 = (mode == 0) ? LSB_Res3_P : LSB_Res3_D;
wire [1:0] LSB_Res4 = (mode == 0) ? LSB_Res4_P : LSB_Res4_D;
wire [2:0] LSB_Res5 = (mode == 0) ? LSB_Res5_P : {1'b0, LSB_Res5_D};  
wire [1:0] LSB_Res6_2bit = LSB_Res6_D;
wire [1:0] LSB_Res7 = (mode == 0) ? LSB_Res7_P : LSB_Res7_D;
wire [1:0] LSB_Res8 = LSB_Res8_P;
wire [1:0] LSB_Res9 = LSB_Res9_P;  
wire [2:0] LSB_Res10 = LSB_Res10_P;  


wire [9:0] signs_comb = (mode == 0) ? {
    (w5_sign ^ a2_sign), (w4_sign ^ a2_sign), (w3_sign ^ a2_sign), (w2_sign ^ a2_sign), (w1_sign ^ a2_sign),
    (w5_sign ^ a1_sign), (w4_sign ^ a1_sign), (w3_sign ^ a1_sign), (w2_sign ^ a1_sign), (w1_sign ^ a1_sign)
} : {
    3'b0, (w7_sign ^ a1_sign), (w6_sign ^ a1_sign), (w5_sign ^ a1_sign), (w4_sign ^ a1_sign),
    (w3_sign ^ a1_sign), (w2_sign ^ a1_sign), (w1_sign ^ a1_sign)
};


wire [26:0] dsp_A = (mode == 0) ? {2'b0, a2_mag, 17'b0, a1_mag} : 
    {1'b0, w7_mag, 2'b0, w6_mag, 2'b0, w5_mag, 2'b0, w4_mag, 2'b0, w3_mag, 2'b0, w2_mag, 2'b0, w1_mag};
wire [17:0] dsp_B = (mode == 0) ? {1'b0, w5_mag, 1'b0, w4_mag, 2'b0, w3_mag, 2'b0, w2_mag, 2'b0, w1_mag} : {14'b0, a1_mag};


reg mode_r1, mode_r2, mode_r3;
reg [9:0] signs_r1, signs_r2, signs_r3;
reg [1:0] LSB_Res2_r1, LSB_Res3_r1, LSB_Res4_r1, LSB_Res6_2bit_r1, LSB_Res7_r1, LSB_Res8_r1, LSB_Res9_r1;
reg [2:0] LSB_Res5_r1, LSB_Res10_r1;
reg [1:0] LSB_Res2_r2, LSB_Res3_r2, LSB_Res4_r2, LSB_Res6_2bit_r2, LSB_Res7_r2, LSB_Res8_r2, LSB_Res9_r2;
reg [2:0] LSB_Res5_r2, LSB_Res10_r2;
reg [1:0] LSB_Res2_r3, LSB_Res3_r3, LSB_Res4_r3, LSB_Res6_2bit_r3, LSB_Res7_r3, LSB_Res8_r3, LSB_Res9_r3;
reg [2:0] LSB_Res5_r3, LSB_Res10_r3;

wire [47:0] dsp_C = (mode_r1 == 0) ? {
    
    9'b0, LSB_Res10_r1, 1'b0, LSB_Res9_r1, 2'b0, LSB_Res8_r1, 2'b0, LSB_Res7_r1, 7'b0, LSB_Res5_r1, 
    1'b0, LSB_Res4_r1, 2'b0, LSB_Res3_r1, 2'b0, LSB_Res2_r1, 4'b0
} : {
    
    22'b0, LSB_Res7_r1, 2'b0, LSB_Res6_2bit_r1, 2'b0, LSB_Res5_r1[1:0], 2'b0, LSB_Res4_r1, 2'b0, LSB_Res3_r1, 2'b0, LSB_Res2_r1, 4'b0
};

always @(posedge clk) begin
    mode_r1 <= mode; signs_r1 <= signs_comb;
    LSB_Res2_r1 <= LSB_Res2; LSB_Res3_r1 <= LSB_Res3; LSB_Res4_r1 <= LSB_Res4; LSB_Res5_r1 <= LSB_Res5;
    LSB_Res6_2bit_r1 <= LSB_Res6_2bit; LSB_Res7_r1 <= LSB_Res7; LSB_Res8_r1 <= LSB_Res8;
    LSB_Res9_r1 <= LSB_Res9; LSB_Res10_r1 <= LSB_Res10;
end

always @(posedge clk) begin
    mode_r2 <= mode_r1; signs_r2 <= signs_r1;
    LSB_Res2_r2 <= LSB_Res2_r1; LSB_Res3_r2 <= LSB_Res3_r1; LSB_Res4_r2 <= LSB_Res4_r1; LSB_Res5_r2 <= LSB_Res5_r1;
    LSB_Res6_2bit_r2 <= LSB_Res6_2bit_r1; LSB_Res7_r2 <= LSB_Res7_r1; LSB_Res8_r2 <= LSB_Res8_r1;
    LSB_Res9_r2 <= LSB_Res9_r1; LSB_Res10_r2 <= LSB_Res10_r1;
end

always @(posedge clk) begin
    mode_r3 <= mode_r2; signs_r3 <= signs_r2;
    LSB_Res2_r3 <= LSB_Res2_r2; LSB_Res3_r3 <= LSB_Res3_r2; LSB_Res4_r3 <= LSB_Res4_r2; LSB_Res5_r3 <= LSB_Res5_r2;
    LSB_Res6_2bit_r3 <= LSB_Res6_2bit_r2; LSB_Res7_r3 <= LSB_Res7_r2; LSB_Res8_r3 <= LSB_Res8_r2;
    LSB_Res9_r3 <= LSB_Res9_r2; LSB_Res10_r3 <= LSB_Res10_r2;
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


wire [5:0] res1_mag, res2_mag, res3_mag, res4_mag, res5_mag, res6_mag, res7_mag, res8_mag, res9_mag, res10_mag;
assign res1_mag  = dsp_P[5:0];  
assign res2_mag  = {dsp_P[9:6], LSB_Res2_r3};  
assign res3_mag  = {dsp_P[13:10], LSB_Res3_r3};  
assign res4_mag  = {dsp_P[17:14], LSB_Res4_r3};  
assign res5_mag  = (mode_r3 == 0) ? {dsp_P[20:18], LSB_Res5_r3} : {dsp_P[21:18], LSB_Res5_r3[1:0]};  
assign res6_mag  = (mode_r3 == 0) ? dsp_P[26:21] : {dsp_P[25:22], LSB_Res6_2bit_r3};  
assign res7_mag  = (mode_r3 == 0) ? {dsp_P[30:27], LSB_Res7_r3} : {dsp_P[29:26], LSB_Res7_r3};  
assign res8_mag  = (mode_r3 == 0) ? {dsp_P[34:31], LSB_Res8_r3} : 6'b0;  
assign res9_mag  = (mode_r3 == 0) ? {dsp_P[38:35], LSB_Res9_r3} : 6'b0;  
assign res10_mag = (mode_r3 == 0) ? {dsp_P[41:39], LSB_Res10_r3} : 6'b0;  

assign result1  = (signs_r3[0] ? (~{1'b0, res1_mag} + 7'b0000001) : {1'b0, res1_mag});
assign result2  = (signs_r3[1] ? (~{1'b0, res2_mag} + 7'b0000001) : {1'b0, res2_mag});
assign result3  = (signs_r3[2] ? (~{1'b0, res3_mag} + 7'b0000001) : {1'b0, res3_mag});
assign result4  = (signs_r3[3] ? (~{1'b0, res4_mag} + 7'b0000001) : {1'b0, res4_mag});
assign result5  = (signs_r3[4] ? (~{1'b0, res5_mag} + 7'b0000001) : {1'b0, res5_mag});
assign result6  = (signs_r3[5] ? (~{1'b0, res6_mag} + 7'b0000001) : {1'b0, res6_mag});
assign result7  = (signs_r3[6] ? (~{1'b0, res7_mag} + 7'b0000001) : {1'b0, res7_mag});
assign result8  = (mode_r3 == 0) ? ((signs_r3[7] ? (~{1'b0, res8_mag} + 7'b0000001) : {1'b0, res8_mag})) : 7'b0;
assign result9  = (mode_r3 == 0) ? ((signs_r3[8] ? (~{1'b0, res9_mag} + 7'b0000001) : {1'b0, res9_mag})) : 7'b0;
assign result10 = (mode_r3 == 0) ? ((signs_r3[9] ? (~{1'b0, res10_mag} + 7'b0000001) : {1'b0, res10_mag})) : 7'b0;

assign {sign10, sign9, sign8, sign7, sign6, sign5, sign4, sign3, sign2, sign1} = signs_r3;

endmodule
