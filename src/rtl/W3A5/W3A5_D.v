
module INT5_INT3_D(
    
    input wire        clk,        
    
    
    input wire [2:0]  w1,         
    input wire [2:0]  w2,         
    input wire [2:0]  w3,         
    input wire [2:0]  w4,         
    input wire [2:0]  w5,         
    input wire [2:0]  w6,         
    input wire [2:0]  w7,         
    
    
    input wire [4:0]  a1,         

    
    output wire [6:0] result1,    
    output wire [6:0] result2,    
    output wire [6:0] result3,    
    output wire [6:0] result4,    
    output wire [6:0] result5,    
    output wire [6:0] result6,    
    output wire [6:0] result7,    

    
    output wire       sign1,      
    output wire       sign2,      
    output wire       sign3,      
    output wire       sign4,      
    output wire       sign5,      
    output wire       sign6,      
    output wire       sign7       
);





reg [6:0] signs_r1;
reg [1:0]  LSB_Res2_r1, LSB_Res3_r1, LSB_Res4_r1, LSB_Res5_r1;
reg [1:0]  LSB_Res6_r1, LSB_Res7_r1;


reg [6:0] signs_r2;
reg [1:0]  LSB_Res2_r2, LSB_Res3_r2, LSB_Res4_r2, LSB_Res5_r2;
reg [1:0]  LSB_Res6_r2, LSB_Res7_r2;


reg [6:0] signs_r3;
reg [1:0]  LSB_Res2_r3, LSB_Res3_r3, LSB_Res4_r3, LSB_Res5_r3;
reg [1:0]  LSB_Res6_r3, LSB_Res7_r3;




wire [47:0] dsp_P;   





wire w1_sign, w2_sign, w3_sign, w4_sign, w5_sign, w6_sign, w7_sign;
wire [1:0] w1_mag, w2_mag, w3_mag, w4_mag, w5_mag, w6_mag, w7_mag;

assign w1_sign = w1[2];
assign w1_mag  = w1[1:0];
assign w2_sign = w2[2];
assign w2_mag  = w2[1:0];
assign w3_sign = w3[2];
assign w3_mag  = w3[1:0];
assign w4_sign = w4[2];
assign w4_mag  = w4[1:0];
assign w5_sign = w5[2];
assign w5_mag  = w5[1:0];
assign w6_sign = w6[2];
assign w6_mag  = w6[1:0];
assign w7_sign = w7[2];
assign w7_mag  = w7[1:0];


wire a1_sign;
wire [3:0] a1_mag;
wire [4:0] a1_temp_mag;


assign a1_sign = a1[4];
assign a1_temp_mag = a1[4] ? (~a1[4:0] + 1'b1) : a1[4:0];
assign a1_mag = (a1_temp_mag == 5'b10000) ? 4'b1111 : a1_temp_mag[3:0];





wire [1:0] LSB_Res2, LSB_Res3, LSB_Res4, LSB_Res5, LSB_Res6, LSB_Res7;


assign LSB_Res2[0] = w2_mag[0] & a1_mag[0];
assign LSB_Res2[1] = (w2_mag[0] & a1_mag[1]) ^ (w2_mag[1] & a1_mag[0]);


assign LSB_Res3[0] = w3_mag[0] & a1_mag[0];
assign LSB_Res3[1] = (w3_mag[0] & a1_mag[1]) ^ (w3_mag[1] & a1_mag[0]);


assign LSB_Res4[0] = w4_mag[0] & a1_mag[0];
assign LSB_Res4[1] = (w4_mag[0] & a1_mag[1]) ^ (w4_mag[1] & a1_mag[0]);


assign LSB_Res5[0] = w5_mag[0] & a1_mag[0];
assign LSB_Res5[1] = (w5_mag[0] & a1_mag[1]) ^ (w5_mag[1] & a1_mag[0]);


assign LSB_Res6[0] = w6_mag[0] & a1_mag[0];
assign LSB_Res6[1] = (w6_mag[0] & a1_mag[1]) ^ (w6_mag[1] & a1_mag[0]);


assign LSB_Res7[0] = w7_mag[0] & a1_mag[0];
assign LSB_Res7[1] = (w7_mag[0] & a1_mag[1]) ^ (w7_mag[1] & a1_mag[0]);





wire [6:0] signs_comb;
assign signs_comb = {
    (w7_sign ^ a1_sign),  
    (w6_sign ^ a1_sign),  
    (w5_sign ^ a1_sign),  
    (w4_sign ^ a1_sign),  
    (w3_sign ^ a1_sign),  
    (w2_sign ^ a1_sign),  
    (w1_sign ^ a1_sign)   
};



wire [17:0] dsp_B;
assign dsp_B = {14'b0, a1_mag};


wire [26:0] dsp_A;
assign dsp_A = {1'b0, w7_mag, 2'b0, w6_mag, 2'b0, w5_mag, 2'b0, w4_mag, 2'b0, w3_mag, 2'b0, w2_mag, 2'b0, w1_mag};



wire [47:0] dsp_C;
assign dsp_C = {
    22'b0,              
    LSB_Res7_r1,        
    2'b0,               
    LSB_Res6_r1,        
    2'b0,               
    LSB_Res5_r1,        
    2'b0,               
    LSB_Res4_r1,        
    2'b0,               
    LSB_Res3_r1,        
    2'b0,               
    LSB_Res2_r1,        
    4'b0                
};





always @(posedge clk) begin
    signs_r1 <= signs_comb;
    LSB_Res2_r1 <= LSB_Res2;
    LSB_Res3_r1 <= LSB_Res3;
    LSB_Res4_r1 <= LSB_Res4;
    LSB_Res5_r1 <= LSB_Res5;
    LSB_Res6_r1 <= LSB_Res6;
    LSB_Res7_r1 <= LSB_Res7;
end


always @(posedge clk) begin
    signs_r2 <= signs_r1;
    LSB_Res2_r2 <= LSB_Res2_r1;
    LSB_Res3_r2 <= LSB_Res3_r1;
    LSB_Res4_r2 <= LSB_Res4_r1;
    LSB_Res5_r2 <= LSB_Res5_r1;
    LSB_Res6_r2 <= LSB_Res6_r1;
    LSB_Res7_r2 <= LSB_Res7_r1;
end


always @(posedge clk) begin
    signs_r3 <= signs_r2;
    LSB_Res2_r3 <= LSB_Res2_r2;
    LSB_Res3_r3 <= LSB_Res3_r2;
    LSB_Res4_r3 <= LSB_Res4_r2;
    LSB_Res5_r3 <= LSB_Res5_r2;
    LSB_Res6_r3 <= LSB_Res6_r2;
    LSB_Res7_r3 <= LSB_Res7_r2;
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


wire [5:0] res1_mag_comb, res2_mag_comb, res3_mag_comb, res4_mag_comb;
wire [5:0] res5_mag_comb, res6_mag_comb, res7_mag_comb;



assign res1_mag_comb  = dsp_P[5:0];                      
assign res2_mag_comb  = {dsp_P[9:6],    LSB_Res2_r3};   
assign res3_mag_comb  = {dsp_P[13:10],  LSB_Res3_r3};   
assign res4_mag_comb  = {dsp_P[17:14],  LSB_Res4_r3};   
assign res5_mag_comb  = {dsp_P[21:18],  LSB_Res5_r3};   
assign res6_mag_comb  = {dsp_P[25:22],  LSB_Res6_r3};   
assign res7_mag_comb  = {dsp_P[29:26],  LSB_Res7_r3};   







assign result1 = {1'b0,res1_mag_comb} ^ {7{signs_r3[0]}};
assign result2 = {1'b0,res2_mag_comb} ^ {7{signs_r3[1]}};
assign result3 = {1'b0,res3_mag_comb} ^ {7{signs_r3[2]}};
assign result4 = {1'b0,res4_mag_comb} ^ {7{signs_r3[3]}};
assign result5 = {1'b0,res5_mag_comb} ^ {7{signs_r3[4]}};
assign result6 = {1'b0,res6_mag_comb} ^ {7{signs_r3[5]}};
assign result7 = {1'b0,res7_mag_comb} ^ {7{signs_r3[6]}};

assign sign1 = signs_r3[0];
assign sign2 = signs_r3[1];
assign sign3 = signs_r3[2];
assign sign4 = signs_r3[3];
assign sign5 = signs_r3[4];
assign sign6 = signs_r3[5];
assign sign7 = signs_r3[6];

endmodule
