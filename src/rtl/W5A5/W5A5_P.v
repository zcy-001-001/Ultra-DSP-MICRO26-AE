
module INT5_INT5_P(
    
    input wire        clk,        
    
    
    input wire [4:0]  w1,         
    input wire [4:0]  w2,         
    input wire [4:0]  a1,         
    input wire [4:0]  a2,         
    input wire [4:0]  a3,         

    
    output wire [7:0] result1,    
    output wire [7:0] result2,    
    output wire [7:0] result3,    
    output wire [7:0] result4,    
    output wire [7:0] result5,    
    output wire [7:0] result6,    
    
    
    output wire       sign1,      
    output wire       sign2,      
    output wire       sign3,      
    output wire       sign4,      
    output wire       sign5,      
    output wire       sign6       
    
);





reg [5:0]  signs_r1;
reg [0:0]  LSB_Res2_r1, LSB_Res5_r1;
reg [1:0]  LSB_Res3_r1, LSB_Res6_r1;


reg [5:0]  signs_r2;
reg [0:0]  LSB_Res2_r2, LSB_Res5_r2;
reg [1:0]  LSB_Res3_r2, LSB_Res6_r2;


reg [5:0]  signs_r3;
reg [0:0]  LSB_Res2_r3, LSB_Res5_r3;
reg [1:0]  LSB_Res3_r3, LSB_Res6_r3;




wire [47:0] dsp_P;   





wire w1_sign, w2_sign;
wire [3:0] w1_mag, w2_mag;

assign w1_sign = w1[4];
assign w1_mag  = w1[3:0];
assign w2_sign = w2[4];
assign w2_mag  = w2[3:0];


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





wire [0:0] LSB_Res2, LSB_Res5;  
wire [1:0] LSB_Res3, LSB_Res6;  


assign LSB_Res2 = w1_mag[0] & a2_mag[0];
assign LSB_Res3 = {(w1_mag[0] & a3_mag[1]) ^ (w1_mag[1] & a3_mag[0]), w1_mag[0] & a3_mag[0]};
assign LSB_Res5 = w2_mag[0] & a2_mag[0];
assign LSB_Res6 = {(w2_mag[0] & a3_mag[1]) ^ (w2_mag[1] & a3_mag[0]), w2_mag[0] & a3_mag[0]};








wire [5:0] signs_comb;
assign signs_comb = { 
    (w2_sign ^ a3_sign),  
    (w2_sign ^ a2_sign),  
    (w2_sign ^ a1_sign),  
    (w1_sign ^ a3_sign),  
    (w1_sign ^ a2_sign),  
    (w1_sign ^ a1_sign)   
};



wire [17:0] dsp_B;
assign dsp_B = { 1'b0, a3_mag, 2'b0, a2_mag, 3'b0, a1_mag};


wire [26:0] dsp_A;
assign dsp_A = {2'b0, w2_mag, 17'b0, w1_mag};


wire [47:0] dsp_C;
assign dsp_C = {
    12'b0,         
    LSB_Res6_r1,   
    5'b0,          
    LSB_Res5_r1,   
    13'b0,         
    LSB_Res3_r1,   
    5'b0,          
    LSB_Res2_r1,   
    7'b0           
};





always @(posedge clk) begin
    signs_r1 <= signs_comb;
    LSB_Res2_r1 <= LSB_Res2;
    LSB_Res3_r1 <= LSB_Res3;
    LSB_Res5_r1 <= LSB_Res5;
    LSB_Res6_r1 <= LSB_Res6;
end


always @(posedge clk) begin
    signs_r2 <= signs_r1;
    LSB_Res2_r2 <= LSB_Res2_r1;
    LSB_Res3_r2 <= LSB_Res3_r1;
    LSB_Res5_r2 <= LSB_Res5_r1;
    LSB_Res6_r2 <= LSB_Res6_r1;
end


always @(posedge clk) begin
    signs_r3 <= signs_r2;
    LSB_Res2_r3 <= LSB_Res2_r2;
    LSB_Res3_r3 <= LSB_Res3_r2;
    LSB_Res5_r3 <= LSB_Res5_r2;
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





wire [7:0] res1_mag_comb, res2_mag_comb, res3_mag_comb, res4_mag_comb;
wire [7:0] res5_mag_comb, res6_mag_comb;



assign res1_mag_comb = dsp_P[7:0];                      
assign res2_mag_comb = {dsp_P[14:8], LSB_Res2_r3};      
assign res3_mag_comb = {dsp_P[20:15], LSB_Res3_r3};     
assign res4_mag_comb = dsp_P[29:21];                    
assign res5_mag_comb = {dsp_P[35:30], LSB_Res5_r3};     
assign res6_mag_comb = {dsp_P[41:36], LSB_Res6_r3};     





assign result1 = res1_mag_comb ^ {8{signs_r3[0]}};
assign result2 = res2_mag_comb ^ {8{signs_r3[1]}};
assign result3 = res3_mag_comb ^ {8{signs_r3[2]}};
assign result4 = res4_mag_comb ^ {8{signs_r3[3]}};
assign result5 = res5_mag_comb ^ {8{signs_r3[4]}};
assign result6 = res6_mag_comb ^ {8{signs_r3[5]}};




assign sign1 = signs_r3[0];
assign sign2 = signs_r3[1];
assign sign3 = signs_r3[2];
assign sign4 = signs_r3[3];
assign sign5 = signs_r3[4];
assign sign6 = signs_r3[5];

endmodule
