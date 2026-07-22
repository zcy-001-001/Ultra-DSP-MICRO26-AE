/*
 * Auto-generated for the Ultra-DSP rebuttal area ablation.
 * Top module: W5A5_stage_s1_weight_offline
 * Precision: W5A5
 * Stage: s1_weight_offline
 * Source: Ultra-DSP-main/DSP_Packaging_Verilog/W5A5/W5A5_Hybrid.v
 * Toggle summary: offline sign-magnitude weights; combinational C-port; per-result two's-complement sign recovery
 */

module W5A5_stage_s1_weight_offline(
    input wire        clk,
    input wire        mode,
    input wire [4:0]  w1, w2, w3, w4, w5,
    input wire [4:0]  a1, a2, a3,
    output wire [7:0] result1, result2, result3, result4, result5, result6,
    output wire       sign1, sign2, sign3, sign4, sign5, sign6
);

wire [47:0] dsp_P;




wire w1_sign, w2_sign, w3_sign, w4_sign, w5_sign;
wire [3:0] w1_mag, w2_mag, w3_mag, w4_mag, w5_mag;

assign w1_sign = w1[4]; assign w1_mag = w1[3:0];
assign w2_sign = w2[4]; assign w2_mag = w2[3:0];
assign w3_sign = w3[4]; assign w3_mag = w3[3:0];
assign w4_sign = w4[4]; assign w4_mag = w4[3:0];
assign w5_sign = w5[4]; assign w5_mag = w5[3:0];




wire a1_sign, a2_sign, a3_sign;
wire [3:0] a1_mag, a2_mag, a3_mag;
wire [4:0] a1_temp_mag, a2_temp_mag, a3_temp_mag;

assign a1_sign = a1[4];
assign a1_temp_mag = a1[4] ? (~a1[4:0] + 1'b1) : a1[4:0];
assign a1_mag = (a1_temp_mag == 5'b10000) ? 4'b1111 : a1_temp_mag[3:0];

assign a2_sign = a2[4];
assign a2_temp_mag = a2[4] ? (~a2[4:0] + 1'b1) : a2[4:0];
assign a2_mag = (a2_temp_mag == 5'b10000) ? 4'b1111 : a2_temp_mag[3:0];

assign a3_sign = a3[4];
assign a3_temp_mag = a3[4] ? (~a3[4:0] + 1'b1) : a3[4:0];
assign a3_mag = (a3_temp_mag == 5'b10000) ? 4'b1111 : a3_temp_mag[3:0];





wire [0:0] LSB_Res2_P = w1_mag[0] & a2_mag[0];
wire [1:0] LSB_Res3_P = {(w1_mag[0] & a3_mag[1]) ^ (w1_mag[1] & a3_mag[0]), w1_mag[0] & a3_mag[0]};
wire [0:0] LSB_Res5_P = w2_mag[0] & a2_mag[0];
wire [1:0] LSB_Res6_P = {(w2_mag[0] & a3_mag[1]) ^ (w2_mag[1] & a3_mag[0]), w2_mag[0] & a3_mag[0]};


wire [1:0] LSB_Res2_D = {(w2_mag[0] & a1_mag[1]) ^ (w2_mag[1] & a1_mag[0]), w2_mag[0] & a1_mag[0]};
wire [1:0] LSB_Res3_D = {(w3_mag[0] & a1_mag[1]) ^ (w3_mag[1] & a1_mag[0]), w3_mag[0] & a1_mag[0]};

wire c1_res4_D = (w4_mag[0] & a1_mag[1]) & (w4_mag[1] & a1_mag[0]);
wire [2:0] LSB_Res4_D = {
    (w4_mag[0] & a1_mag[2]) ^ (w4_mag[1] & a1_mag[1]) ^ (w4_mag[2] & a1_mag[0]) ^ c1_res4_D,
    (w4_mag[0] & a1_mag[1]) ^ (w4_mag[1] & a1_mag[0]),
    w4_mag[0] & a1_mag[0]
};

wire c1_res5_D = (w5_mag[0] & a1_mag[1]) & (w5_mag[1] & a1_mag[0]);
wire [2:0] LSB_Res5_D = {
    (w5_mag[0] & a1_mag[2]) ^ (w5_mag[1] & a1_mag[1]) ^ (w5_mag[2] & a1_mag[0]) ^ c1_res5_D,
    (w5_mag[0] & a1_mag[1]) ^ (w5_mag[1] & a1_mag[0]),
    w5_mag[0] & a1_mag[0]
};


wire [0:0] LSB_Res2_1bit = (mode == 0) ? LSB_Res2_P : LSB_Res2_D[0];
wire [1:0] LSB_Res2_2bit = LSB_Res2_D;
wire [1:0] LSB_Res3 = (mode == 0) ? LSB_Res3_P : LSB_Res3_D;
wire [2:0] LSB_Res4 = LSB_Res4_D;
wire [0:0] LSB_Res5_1bit = LSB_Res5_P;
wire [2:0] LSB_Res5_3bit = LSB_Res5_D;
wire [1:0] LSB_Res6 = LSB_Res6_P;




wire [5:0] signs_comb = (mode == 0) ? {
    (w2_sign ^ a3_sign),  
    (w2_sign ^ a2_sign),  
    (w2_sign ^ a1_sign),  
    (w1_sign ^ a3_sign),  
    (w1_sign ^ a2_sign),  
    (w1_sign ^ a1_sign)   
} : {
    (w5_sign ^ a1_sign),  
    (w4_sign ^ a1_sign),  
    (w3_sign ^ a1_sign),  
    (w2_sign ^ a1_sign),  
    (w1_sign ^ a1_sign)   
};




wire [26:0] dsp_A = (mode == 0) ? 
    {2'b0, w2_mag, 17'b0, w1_mag} :
    {1'b0, w5_mag, 1'b0, w4_mag, 1'b0, w3_mag, 2'b0, w2_mag, 2'b0, w1_mag};

wire [17:0] dsp_B = (mode == 0) ? 
    {1'b0, a3_mag, 2'b0, a2_mag, 3'b0, a1_mag} :
    {14'b0, a1_mag};




reg mode_r1, mode_r2, mode_r3;
reg [10:0] signs_r1, signs_r2, signs_r3;
reg [0:0] LSB_Res2_1bit_r1, LSB_Res5_1bit_r1;
reg [1:0] LSB_Res2_2bit_r1, LSB_Res3_r1, LSB_Res6_r1;
reg [2:0] LSB_Res4_r1, LSB_Res5_3bit_r1;
reg [0:0] LSB_Res2_1bit_r2, LSB_Res5_1bit_r2;
reg [1:0] LSB_Res2_2bit_r2, LSB_Res3_r2, LSB_Res6_r2;
reg [2:0] LSB_Res4_r2, LSB_Res5_3bit_r2;
reg [0:0] LSB_Res2_1bit_r3, LSB_Res5_1bit_r3;
reg [1:0] LSB_Res2_2bit_r3, LSB_Res3_r3, LSB_Res6_r3;
reg [2:0] LSB_Res4_r3, LSB_Res5_3bit_r3;

wire [47:0] dsp_C = (mode == 0) ? {
    12'b0, LSB_Res6, 5'b0, LSB_Res5_1bit, 13'b0,
    LSB_Res3, 5'b0, LSB_Res2_1bit, 7'b0
} : {
    23'b0, LSB_Res5_3bit, 2'b0, LSB_Res4, 3'b0,
    LSB_Res3, 4'b0, LSB_Res2_2bit, 6'b0
};

always @(posedge clk) begin
    mode_r1 <= mode;
    signs_r1 <= signs_comb;
    LSB_Res2_1bit_r1 <= LSB_Res2_1bit;
    LSB_Res2_2bit_r1 <= LSB_Res2_2bit;
    LSB_Res3_r1 <= LSB_Res3;
    LSB_Res4_r1 <= LSB_Res4;
    LSB_Res5_1bit_r1 <= LSB_Res5_1bit;
    LSB_Res5_3bit_r1 <= LSB_Res5_3bit;
    LSB_Res6_r1 <= LSB_Res6;
end

always @(posedge clk) begin
    mode_r2 <= mode_r1;
    signs_r2 <= signs_r1;
    LSB_Res2_1bit_r2 <= LSB_Res2_1bit_r1;
    LSB_Res2_2bit_r2 <= LSB_Res2_2bit_r1;
    LSB_Res3_r2 <= LSB_Res3_r1;
    LSB_Res4_r2 <= LSB_Res4_r1;
    LSB_Res5_1bit_r2 <= LSB_Res5_1bit_r1;
    LSB_Res5_3bit_r2 <= LSB_Res5_3bit_r1;
    LSB_Res6_r2 <= LSB_Res6_r1;
end

always @(posedge clk) begin
    mode_r3 <= mode_r2;
    signs_r3 <= signs_r2;
    LSB_Res2_1bit_r3 <= LSB_Res2_1bit_r2;
    LSB_Res2_2bit_r3 <= LSB_Res2_2bit_r2;
    LSB_Res3_r3 <= LSB_Res3_r2;
    LSB_Res4_r3 <= LSB_Res4_r2;
    LSB_Res5_1bit_r3 <= LSB_Res5_1bit_r2;
    LSB_Res5_3bit_r3 <= LSB_Res5_3bit_r2;
    LSB_Res6_r3 <= LSB_Res6_r2;
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




wire [7:0] res1_mag, res2_mag, res3_mag, res4_mag, res5_mag, res6_mag;


wire [7:0] res1_mag_P = dsp_P[7:0];
wire [7:0] res2_mag_P = {dsp_P[14:8], LSB_Res2_1bit_r3};
wire [7:0] res3_mag_P = {dsp_P[20:15], LSB_Res3_r3};
wire [7:0] res4_mag_P = dsp_P[29:21];
wire [7:0] res5_mag_P = {dsp_P[35:30], LSB_Res5_1bit_r3};
wire [7:0] res6_mag_P = {dsp_P[41:36], LSB_Res6_r3};


wire [7:0] res1_mag_D = dsp_P[7:0];
wire [7:0] res2_mag_D = {dsp_P[13:8], LSB_Res2_2bit_r3};
wire [7:0] res3_mag_D = {dsp_P[19:14], LSB_Res3_r3};
wire [7:0] res4_mag_D = {dsp_P[24:20], LSB_Res4_r3};
wire [7:0] res5_mag_D = {dsp_P[29:25], LSB_Res5_3bit_r3};

assign res1_mag = (mode_r3 == 0) ? res1_mag_P : res1_mag_D;
assign res2_mag = (mode_r3 == 0) ? res2_mag_P : res2_mag_D;
assign res3_mag = (mode_r3 == 0) ? res3_mag_P : res3_mag_D;
assign res4_mag = (mode_r3 == 0) ? res4_mag_P : res4_mag_D;
assign res5_mag = (mode_r3 == 0) ? res5_mag_P : res5_mag_D;
assign res6_mag = res6_mag_P;




assign result1  = (signs_r3[0] ? (~res1_mag + 8'b00000001) : res1_mag);
assign result2  = (signs_r3[1] ? (~res2_mag + 8'b00000001) : res2_mag);
assign result3  = (signs_r3[2] ? (~res3_mag + 8'b00000001) : res3_mag);
assign result4  = (mode_r3 == 0) ? res4_mag : ((signs_r3[3] ? (~res4_mag + 8'b00000001) : res4_mag));
assign result5  = (mode_r3 == 0) ? ((signs_r3[4] ? (~res5_mag + 8'b00000001) : res5_mag)) : ((signs_r3[4] ? (~res5_mag + 8'b00000001) : res5_mag));
assign result6  = (mode_r3 == 0) ? ((signs_r3[5] ? (~res6_mag + 8'b00000001) : res6_mag)) : 8'b0;


assign sign1 = signs_r3[0];
assign sign2 = signs_r3[1];
assign sign3 = signs_r3[2];
assign sign4 = (mode_r3 == 0) ? 1'b0 : signs_r3[3];
assign sign5 = signs_r3[4];
assign sign6 = (mode_r3 == 0) ? signs_r3[5] : 1'b0;
endmodule
