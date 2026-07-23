//============================================================
// DSP_Packing_new.v
// MR-Overpacking (delta = -2), wwidth={5,5}, awidth={4,4,4}
// signed W, unsigned A
//
// Compared with DSP_Packing.v, this version adds a C-port
// compensation term for the signed B-port interpretation issue:
// the MSB of the last activation lane a2[3] lands on B[17],
// which is the sign bit of DSP48E2's 18-bit multiplier input.
//
// Intended packed B:
//   B_u = a0 + (a1 << 7) + (a2 << 14)
//
// What DSP48E2 actually sees on signed B[17:0]:
//   B_s = B_u - a2[3] * 2^18
//
// Therefore the raw multiplier output misses:
//   (w0 + (w1 << 21)) * a2[3] * 2^18
// = a2[3] * ((w0 << 18) + (w1 << 39))
//
// Since this design keeps the DSP configured as P = M - C,
// we feed the negated compensation into C so that:
//   P = M - (-comp) = M + comp
//============================================================
`timescale 1ns/1ps

module FPL (
    input  wire        clk,
    input  wire signed [4:0] w0,
    input  wire signed [4:0] w1,
    input  wire [3:0]  a0,
    input  wire [3:0]  a1,
    input  wire [3:0]  a2,
    output wire signed [8:0] r00,
    output wire signed [8:0] r01,
    output wire signed [8:0] r02,
    output wire signed [8:0] r10,
    output wire signed [8:0] r11,
    output wire signed [8:0] r12
);

    //----------------------------------------------------------
    // Packing according to delta = -2 configuration
    // Offsets: woff={0,21}, aoff={0,7,14}
    //----------------------------------------------------------
    wire [26:0] dsp_A = {{22{w0[4]}}, w0};
    wire [26:0] dsp_D = {{w1[4]}, w1, 21'd0};
    wire [17:0] dsp_B = {a2, 3'b0, a1, 3'b0, a0};

    //----------------------------------------------------------
    // C-port compensation for B[17] sign-bit aliasing
    //
    // Here "a3" in the issue description corresponds to the
    // numeric bit a2[3], i.e. the MSB of the last 4-bit
    // activation lane packed at offset 14.
    //----------------------------------------------------------
    wire signed [47:0] corr_w0 = $signed({{43{w0[4]}}, w0}) <<< 18;
    wire signed [47:0] corr_w1 = $signed({{43{w1[4]}}, w1}) <<< 39;
    wire signed [47:0] corr_term =
        a2[3] ? (corr_w0 + corr_w1) : 48'sd0;

    // DSP is configured as P = M - C, so feed -corr_term.
    wire signed [47:0] dsp_C_signed = -corr_term;
    wire [47:0] dsp_C = dsp_C_signed;

    //----------------------------------------------------------
    // MR-Overpacking: calculate the 2 LSBs of the contaminating
    // neighbor results to restore the corrupted MSBs.
    //----------------------------------------------------------
    wire [1:0] LSB_Res2, LSB_Res3, LSB_Res4, LSB_Res5, LSB_Res6;

    assign LSB_Res2[0] = w0[0] & a1[0];
    assign LSB_Res2[1] = (w0[0] & a1[1]) ^ (w0[1] & a1[0]);
    assign LSB_Res3[0] = w0[0] & a2[0];
    assign LSB_Res3[1] = (w0[0] & a2[1]) ^ (w0[1] & a2[0]);
    assign LSB_Res4[0] = w1[0] & a0[0];
    assign LSB_Res4[1] = (w1[0] & a0[1]) ^ (w1[1] & a0[0]);
    assign LSB_Res5[0] = w1[0] & a1[0];
    assign LSB_Res5[1] = (w1[0] & a1[1]) ^ (w1[1] & a1[0]);
    assign LSB_Res6[0] = w1[0] & a2[0];
    assign LSB_Res6[1] = (w1[0] & a2[1]) ^ (w1[1] & a2[0]);

    reg [1:0] LSB_Res2_r1, LSB_Res3_r1, LSB_Res4_r1, LSB_Res5_r1, LSB_Res6_r1;
    reg [1:0] LSB_Res2_r2, LSB_Res3_r2, LSB_Res4_r2, LSB_Res5_r2, LSB_Res6_r2;
    reg [1:0] LSB_Res2_r3, LSB_Res3_r3, LSB_Res4_r3, LSB_Res5_r3, LSB_Res6_r3;
    reg [1:0] LSB_Res2_r4, LSB_Res3_r4, LSB_Res4_r4, LSB_Res5_r4, LSB_Res6_r4;

    always @(posedge clk) begin
        LSB_Res2_r1 <= LSB_Res2;
        LSB_Res3_r1 <= LSB_Res3;
        LSB_Res4_r1 <= LSB_Res4;
        LSB_Res5_r1 <= LSB_Res5;
        LSB_Res6_r1 <= LSB_Res6;
    end

    always @(posedge clk) begin
        LSB_Res2_r2 <= LSB_Res2_r1;
        LSB_Res3_r2 <= LSB_Res3_r1;
        LSB_Res4_r2 <= LSB_Res4_r1;
        LSB_Res5_r2 <= LSB_Res5_r1;
        LSB_Res6_r2 <= LSB_Res6_r1;
    end

    always @(posedge clk) begin
        LSB_Res2_r3 <= LSB_Res2_r2;
        LSB_Res3_r3 <= LSB_Res3_r2;
        LSB_Res4_r3 <= LSB_Res4_r2;
        LSB_Res5_r3 <= LSB_Res5_r2;
        LSB_Res6_r3 <= LSB_Res6_r2;
    end

    always @(posedge clk) begin
        LSB_Res2_r4 <= LSB_Res2_r3;
        LSB_Res3_r4 <= LSB_Res3_r3;
        LSB_Res4_r4 <= LSB_Res4_r3;
        LSB_Res5_r4 <= LSB_Res5_r3;
        LSB_Res6_r4 <= LSB_Res6_r3;
    end

    //----------------------------------------------------------
    // DSP48E2 primitive
    //----------------------------------------------------------
    wire [47:0] dsp_P;

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
       .IS_RSTALUMODE_INVERTED(1'b0),
       .IS_RSTA_INVERTED(1'b0),
       .IS_RSTB_INVERTED(1'b0),
       .IS_RSTCTRL_INVERTED(1'b0),
       .IS_RSTC_INVERTED(1'b0),
       .IS_RSTD_INVERTED(1'b0),
       .IS_RSTINMODE_INVERTED(1'b0),
       .IS_RSTM_INVERTED(1'b0),
       .IS_RSTP_INVERTED(1'b0),

       .ACASCREG(1),
       .ADREG(1),
       .ALUMODEREG(1),
       .AREG(1),
       .BCASCREG(1),
       .BREG(2),
       .CARRYINREG(1),
       .CARRYINSELREG(1),
       .CREG(1),
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

       .ALUMODE(4'b0001),
       .CARRYINSEL(3'b000),
       .CLK(clk),
       .INMODE(5'b00101),
       .OPMODE(9'b000110101),

       .A({{3{dsp_A[26]}}, dsp_A}),
       .B(dsp_B),
       .C(dsp_C),
       .CARRYIN(1'b1),
       .D(dsp_D),

       .CEA1(1'b1),
       .CEA2(1'b0),
       .CEAD(1'b1),
       .CEALUMODE(1'b1),
       .CEB1(1'b1),
       .CEB2(1'b1),
       .CEC(1'b1),
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

    //----------------------------------------------------------
    // Extract 6 results according to roff = {0,7,14,21,28,35}
    // Restore the 2 corrupted MSBs for MR-overpacking.
    //----------------------------------------------------------
    wire [1:0] MSB0, MSB1, MSB2, MSB3, MSB4;

    assign MSB0 = dsp_P[8:7]   - LSB_Res2_r4;
    assign MSB1 = dsp_P[15:14] - LSB_Res3_r4;
    assign MSB2 = dsp_P[22:21] - LSB_Res4_r4;
    assign MSB3 = dsp_P[29:28] - LSB_Res5_r4;
    assign MSB4 = dsp_P[36:35] - LSB_Res6_r4;

    assign r00 = {MSB0, dsp_P[6:0]};
    assign r01 = {MSB1, dsp_P[13:7]};
    assign r02 = {MSB2, dsp_P[20:14]};
    assign r10 = {MSB3, dsp_P[27:21]};
    assign r11 = {MSB4, dsp_P[34:28]};
    assign r12 = dsp_P[43:35];

endmodule
