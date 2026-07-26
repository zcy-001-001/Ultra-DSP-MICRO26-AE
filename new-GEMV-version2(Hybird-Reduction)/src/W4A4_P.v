`timescale 1ns/1ps

module INT4_INT4_P_MAG(
    
    input wire        clk,
    input wire        ce,
    
    
    input wire [2:0]  w1_magnitude,
    input wire [2:0]  w2_magnitude,
    input wire [2:0]  w3_magnitude,
    input wire [2:0]  a1_magnitude,
    input wire [2:0]  a2_magnitude,
    input wire [2:0]  a3_magnitude,

    
    output wire [5:0] magnitude1,
    output wire [5:0] magnitude2,
    output wire [5:0] magnitude3,
    output wire [5:0] magnitude4,
    output wire [5:0] magnitude5,
    output wire [5:0] magnitude6,
    output wire [5:0] magnitude7,
    output wire [5:0] magnitude8,
    output wire [5:0] magnitude9
    
);





reg [1:0]  LSB_Res2_r1, LSB_Res3_r1, LSB_Res7_r1, LSB_Res5_r1, LSB_Res6_r1, LSB_Res8_r1, LSB_Res9_r1;
reg [2:0]  LSB_Res4_r1;


reg [1:0]  LSB_Res2_r2, LSB_Res3_r2, LSB_Res7_r2, LSB_Res5_r2, LSB_Res6_r2, LSB_Res8_r2, LSB_Res9_r2;
reg [2:0]  LSB_Res4_r2;






wire [47:0] dsp_P;   





wire [2:0] w1_mag = w1_magnitude;
wire [2:0] w2_mag = w2_magnitude;
wire [2:0] w3_mag = w3_magnitude;

wire [2:0] a1_mag = a1_magnitude;
wire [2:0] a2_mag = a2_magnitude;
wire [2:0] a3_mag = a3_magnitude;





wire [1:0] LSB_Res2, LSB_Res3, LSB_Res7, LSB_Res5, LSB_Res6, LSB_Res8, LSB_Res9;
wire [2:0] LSB_Res4;

assign LSB_Res2[0] = w1_mag[0] & a2_mag[0];
assign LSB_Res2[1] = (w1_mag[0] & a2_mag[1]) ^ (w1_mag[1] & a2_mag[0]);
assign LSB_Res3[0] = w1_mag[0] & a3_mag[0];
assign LSB_Res3[1] = (w1_mag[0] & a3_mag[1]) ^ (w1_mag[1] & a3_mag[0]);

wire c1_for_lsb4;
assign LSB_Res4[0] = w2_mag[0] & a1_mag[0];
assign LSB_Res4[1] = (w2_mag[0] & a1_mag[1]) ^ (w2_mag[1] & a1_mag[0]);
assign c1_for_lsb4 = (w2_mag[0] & a1_mag[1]) & (w2_mag[1] & a1_mag[0]);
assign LSB_Res4[2] = (w2_mag[0] & a1_mag[2]) ^ (w2_mag[1] & a1_mag[1]) ^ (w2_mag[2] & a1_mag[0]) ^ c1_for_lsb4;

assign LSB_Res5[0] = w2_mag[0] & a2_mag[0];
assign LSB_Res5[1] = (w2_mag[0] & a2_mag[1]) ^ (w2_mag[1] & a2_mag[0]);
assign LSB_Res6[0] = w2_mag[0] & a3_mag[0];
assign LSB_Res6[1] = (w2_mag[0] & a3_mag[1]) ^ (w2_mag[1] & a3_mag[0]);
assign LSB_Res8[0] = w3_mag[0] & a2_mag[0];
assign LSB_Res8[1] = (w3_mag[0] & a2_mag[1]) ^ (w3_mag[1] & a2_mag[0]);
assign LSB_Res9[0] = w3_mag[0] & a3_mag[0];
assign LSB_Res9[1] = (w3_mag[0] & a3_mag[1]) ^ (w3_mag[1] & a3_mag[0]);



assign LSB_Res7[0] = w3_mag[0] & a1_mag[0];
assign LSB_Res7[1] = (w3_mag[0] & a1_mag[1]) ^ (w3_mag[1] & a1_mag[0]);







wire [17:0] dsp_B;
assign dsp_B = {7'b0, a3_mag, 1'b0, a2_mag, 1'b0, a1_mag};


wire [26:0] dsp_A;
assign dsp_A = {
    1'b0, w3_magnitude,
    9'b0, w2_magnitude,
    8'b0, w1_magnitude
};


wire [47:0] dsp_C;
assign dsp_C = {
    15'b0,
    LSB_Res9,
    2'b0,
    LSB_Res8,
    2'b0,
    LSB_Res7,
    2'b0,
    LSB_Res6,
    2'b0,
    LSB_Res5,
    1'b0,
    LSB_Res4,
    1'b0,
    LSB_Res3,
    2'b0,
    LSB_Res2,
    4'b0       
};





always @(posedge clk) begin
    if (ce) begin
        LSB_Res2_r1 <= LSB_Res2;
        LSB_Res3_r1 <= LSB_Res3;
        LSB_Res4_r1 <= LSB_Res4;
        LSB_Res5_r1 <= LSB_Res5;
        LSB_Res6_r1 <= LSB_Res6;
        LSB_Res7_r1 <= LSB_Res7;
        LSB_Res8_r1 <= LSB_Res8;
        LSB_Res9_r1 <= LSB_Res9;
    end
end


always @(posedge clk) begin
    if (ce) begin
        LSB_Res2_r2 <= LSB_Res2_r1;
        LSB_Res3_r2 <= LSB_Res3_r1;
        LSB_Res4_r2 <= LSB_Res4_r1;
        LSB_Res5_r2 <= LSB_Res5_r1;
        LSB_Res6_r2 <= LSB_Res6_r1;
        LSB_Res7_r2 <= LSB_Res7_r1;
        LSB_Res8_r2 <= LSB_Res8_r1;
        LSB_Res9_r2 <= LSB_Res9_r1;
    end
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
            .ACASCREG(0), .ADREG(0), .ALUMODEREG(0), .AREG(0), .BCASCREG(0), .BREG(0), .CARRYINREG(0),
            .CARRYINSELREG(0), .CREG(1), .DREG(0), .INMODEREG(0), .MREG(1), .OPMODEREG(0), .PREG(1)
        ) dsp_inst (
            .ACOUT(), .BCOUT(), .CARRYCASCOUT(), .MULTSIGNOUT(), .PCOUT(), .OVERFLOW(), .PATTERNBDETECT(), .PATTERNDETECT(), .UNDERFLOW(),
            .CARRYOUT(), .P(dsp_P), .XOROUT(),
            .ACIN(30'b0), .BCIN(18'b0), .CARRYCASCIN(1'b0), .MULTSIGNIN(1'b0), .PCIN(48'b0),
            .ALUMODE(4'b0001), .CARRYINSEL(3'b000), .CLK(clk), .INMODE(5'b00000), .OPMODE(9'b000110101),
            .A({3'b0, dsp_A}), .B(dsp_B), .C(dsp_C), .CARRYIN(1'b1), .D(27'b0),
            .CEA1(1'b0), .CEA2(1'b0), .CEAD(1'b0), .CEALUMODE(1'b0), .CEB1(1'b0), .CEB2(1'b0), .CEC(ce),
            .CECARRYIN(1'b0), .CECTRL(1'b0), .CED(1'b0), .CEINMODE(1'b0), .CEM(ce), .CEP(ce),
            .RSTA(1'b0), .RSTALLCARRYIN(1'b0), .RSTALUMODE(1'b0), .RSTB(1'b0), .RSTC(1'b0), .RSTCTRL(1'b0),
            .RSTD(1'b0), .RSTINMODE(1'b0), .RSTM(1'b0), .RSTP(1'b0)
        );





wire [5:0] res1_mag_comb, res2_mag_comb, res3_mag_comb;
wire [5:0] res4_mag_comb, res5_mag_comb, res6_mag_comb;
wire [5:0] res7_mag_comb, res8_mag_comb, res9_mag_comb;



assign res1_mag_comb = dsp_P[5:0];                     
assign res2_mag_comb = {dsp_P[9:6],   LSB_Res2_r2};
assign res3_mag_comb = {dsp_P[13:10], LSB_Res3_r2};
assign res4_mag_comb = {dsp_P[16:14], LSB_Res4_r2};
assign res5_mag_comb = {dsp_P[20:17], LSB_Res5_r2};
assign res6_mag_comb = {dsp_P[24:21], LSB_Res6_r2};
assign res7_mag_comb = {dsp_P[28:25], LSB_Res7_r2};
assign res8_mag_comb = {dsp_P[32:29], LSB_Res8_r2};
assign res9_mag_comb = {dsp_P[36:33], LSB_Res9_r2};




// Keep all nine overpacked products independent, but expose magnitude and sign
// separately. Applying two's-complement to every product here duplicates nine
// banks of XOR logic; the first accumulation stage folds the sign into its add.
assign magnitude1 = res1_mag_comb;
assign magnitude2 = res2_mag_comb;
assign magnitude3 = res3_mag_comb;
assign magnitude4 = res4_mag_comb;
assign magnitude5 = res5_mag_comb;
assign magnitude6 = res6_mag_comb;
assign magnitude7 = res7_mag_comb;
assign magnitude8 = res8_mag_comb;
assign magnitude9 = res9_mag_comb;
endmodule

// Compatibility shell for standalone users of the original signed-activation
// interface. The full array bypasses this shell and shares each conversion
// across all 64 columns of a row.
module INT4_INT4_P(
    input wire        clk,
    input wire        ce,
    input wire [2:0]  w1_magnitude,
    input wire [2:0]  w2_magnitude,
    input wire [2:0]  w3_magnitude,
    input wire [3:0]  a1,
    input wire [3:0]  a2,
    input wire [3:0]  a3,
    output wire [5:0] magnitude1,
    output wire [5:0] magnitude2,
    output wire [5:0] magnitude3,
    output wire [5:0] magnitude4,
    output wire [5:0] magnitude5,
    output wire [5:0] magnitude6,
    output wire [5:0] magnitude7,
    output wire [5:0] magnitude8,
    output wire [5:0] magnitude9
);

function automatic [2:0] activation_magnitude;
    input [3:0] value;
    reg [2:0] negative_magnitude;
    begin
        negative_magnitude[0] = value[0];
        negative_magnitude[1] = value[1] ^ value[0];
        negative_magnitude[2] = value[2] ^ (value[1] | value[0]);
        if (value[2:0] == 3'b000)
            negative_magnitude = 3'b111;
        activation_magnitude = value[3] ? negative_magnitude : value[2:0];
    end
endfunction

wire [2:0] a1_magnitude = activation_magnitude(a1);
wire [2:0] a2_magnitude = activation_magnitude(a2);
wire [2:0] a3_magnitude = activation_magnitude(a3);

INT4_INT4_P_MAG core (
    .clk(clk), .ce(ce),
    .w1_magnitude(w1_magnitude),
    .w2_magnitude(w2_magnitude),
    .w3_magnitude(w3_magnitude),
    .a1_magnitude(a1_magnitude),
    .a2_magnitude(a2_magnitude),
    .a3_magnitude(a3_magnitude),
    .magnitude1(magnitude1), .magnitude2(magnitude2),
    .magnitude3(magnitude3), .magnitude4(magnitude4),
    .magnitude5(magnitude5), .magnitude6(magnitude6),
    .magnitude7(magnitude7), .magnitude8(magnitude8),
    .magnitude9(magnitude9)
);

endmodule
