`timescale 1ns / 1ps

// DuoQ INT4 x INT4 signed packing on a single DSP48E2.
//
// The original version in this folder used a fully pipelined DSP48E2
// configuration plus a 4-stage sideband metadata delay line to align the
// correction path. That implementation is functionally correct, but it also
// explains the resource gap against the paper:
//   1. `corr_meta_d1..d4` alone costs 4 * 9 = 36 fabric FFs.
//   2. The nibble-sliced external correction logic synthesizes into more LUTs
//      than the paper's Fig.5 style byte-level compensation path.
//
// This version keeps the same signed 4-way packing math, but moves back to a
// low-latency PE-style implementation:
//   * no sideband alignment pipeline in fabric
//   * direct byte-level correction/carry compensation from the current inputs
//   * one output register stage for stable timing at the module boundary
//
// Functionally it matches the `paper_style_model` in verify_duoq_pack.py.

module DuoQ (
    input  wire              clk,
    input  wire signed [3:0] a_in,
    input  wire signed [3:0] w0,
    input  wire signed [3:0] w1,
    input  wire signed [3:0] w2,
    input  wire signed [3:0] w3,
    output wire signed [7:0] p0,
    output wire signed [7:0] p1,
    output wire signed [7:0] p2,
    output wire signed [7:0] p3
);

// Pack the four INT4 weights into the DSP pre-adder inputs.
// A uses the lower 27 bits of the 30-bit port inside DSP48E2, so the top
// three bits are left as zero padding.
wire [29:0] dsp_A = {18'd0, w1, 4'b0, w0};
wire [26:0] dsp_D = {w3[2:0], 4'b0, w2, 16'b0};
wire [17:0] dsp_B = {{14{a_in[3]}}, a_in};

wire [47:0] dsp_P;

wire signed [7:0] raw_p0 = dsp_P[7:0];
wire signed [7:0] raw_p1 = dsp_P[15:8];
wire signed [7:0] raw_p2 = dsp_P[23:16];
wire signed [7:0] raw_p3 = dsp_P[31:24];

wire signed [7:0] a_ext    = {{4{a_in[3]}}, a_in};
wire signed [7:0] a_shift4 = a_ext <<< 4;
wire signed [7:0] a_shift3 = a_ext <<< 3;

// Paper Eq.(6)/(7): correction items for the four packed lanes.
wire signed [7:0] ct0 = w0[3] ? a_shift4 : 8'sd0;
wire signed [7:0] ct1 = w1[3] ? a_shift4 : 8'sd0;
wire signed [7:0] ct2 = w2[3] ? a_shift4 : 8'sd0;
wire signed [7:0] ct3 =
    (w3[3] ? a_shift3 : 8'sd0) -
    (w3[2] ? a_shift3 : 8'sd0);

// Signed-signed packing introduces carry into the adjacent byte lanes.
// Using the sign bit of the preceding raw byte is equivalent to the carry
// compensation used in the exhaustive Python model.
wire signed [8:0] p0_calc = $signed(raw_p0) - $signed(ct0);
wire signed [8:0] p1_calc = $signed(raw_p1) - $signed(ct1) +
                            (raw_p0[7] ? 9'sd1 : 9'sd0);
wire signed [8:0] p2_calc = $signed(raw_p2) - $signed(ct2) +
                            (raw_p1[7] ? 9'sd1 : 9'sd0);
wire signed [8:0] p3_calc = $signed(raw_p3) - $signed(ct3) +
                            (raw_p2[7] ? 9'sd1 : 9'sd0);

reg signed [7:0] p0_r = 8'sd0;
reg signed [7:0] p1_r = 8'sd0;
reg signed [7:0] p2_r = 8'sd0;
reg signed [7:0] p3_r = 8'sd0;

always @(posedge clk) begin
    p0_r <= p0_calc[7:0];
    p1_r <= p1_calc[7:0];
    p2_r <= p2_calc[7:0];
    p3_r <= p3_calc[7:0];
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

    // Use the DSP as a low-latency packed multiplier. The fabric output
    // register above keeps the module interface synchronous without carrying
    // a separate metadata pipeline through the PE.
    .ACASCREG(0),
    .ADREG(0),
    .ALUMODEREG(0),
    .AREG(0),
    .BCASCREG(0),
    .BREG(0),
    .CARRYINREG(0),
    .CARRYINSELREG(0),
    .CREG(0),
    .DREG(0),
    .INMODEREG(0),
    .MREG(0),
    .OPMODEREG(0),
    .PREG(0)
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
    .INMODE(5'b00101),
    .OPMODE(9'b000000101),

    .A(dsp_A),
    .B(dsp_B),
    .C(48'd0),
    .CARRYIN(1'b0),
    .D(dsp_D),

    .CEA1(1'b1),
    .CEA2(1'b1),
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

assign p0 = p0_r;
assign p1 = p1_r;
assign p2 = p2_r;
assign p3 = p3_r;

endmodule
