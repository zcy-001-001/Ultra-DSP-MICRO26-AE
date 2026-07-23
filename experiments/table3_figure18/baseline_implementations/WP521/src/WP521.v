//=====================================================================
// Xilinx INT4 Packing (no compensation)
// Based on WP521 
// Performs four parallel multiplications:
// r00 = a0*w0 , r01 = a0*w1 , r10 = a1*w0 , r11 = a1*w1
// Latency=4 II=1
//=====================================================================
`timescale 1ns/1ps

module WP521 (
    input  wire        clk,
    input  wire signed [3:0] w0,   // signed weights
    input  wire signed [3:0] w1,
    input  wire [3:0]  a0,         // unsigned activations
    input  wire [3:0]  a1,
    output wire signed [7:0] r00,  // a0*w0
    output wire signed [7:0] r01,  // a0*w1
    output wire signed [7:0] r10,  // a1*w0
    output wire signed [7:0] r11   // a1*w1
);

    //-----------------------------------------------------------------
    // Packaging INPUT Port
    //-----------------------------------------------------------------
    wire [26:0] dsp_A = {{23{w0[3]}}, w0};     // w0 at offset 0
    wire [26:0] dsp_D = {w1[3], w1,22'd0};     // w1 at offset 22
    wire [17:0] dsp_B = {3'd0, a1, 7'd0, a0};              // a0@0, a1@11
    wire [47:0] dsp_C = 48'd0;

    //-----------------------------------------------------------------
    // DSP48E2 Primitive: P = (A + D) * B + C
    //-----------------------------------------------------------------
    wire [47:0] dsp_P;
DSP48E2 #(
   // Feature Control Attributes: Data Path Selection
   // 关键修改：选择预加法器(AD)的输出作为乘法器的A输入
   .AMULTSEL("AD"),                 // Selects A input to multiplier (A, AD) 
   .A_INPUT("DIRECT"),           // Selects A input source, "DIRECT" (A port) or "CASCADE" (ACIN port)
   .BMULTSEL("B"),                 // Selects B input to multiplier (AD, B)
   .B_INPUT("DIRECT"),           // Selects B input source, "DIRECT" (B port) or "CASCADE" (BCIN port)
   .PREADDINSEL("A"),            // Selects input to pre-adder (A, B)
   .RND(48'h000000000000),         // Rounding Constant (未使用)
   .USE_MULT("MULTIPLY"),        // 必须使用乘法器 (MULTIPLY)
   .USE_SIMD("ONE48"),           // 使用单个48位加法器
   .USE_WIDEXOR("FALSE"),        // (未使用)
   .XORSIMD("XOR24_48_96"),      // (未使用)
   
   // Pattern Detector Attributes: (未使用)
   .AUTORESET_PATDET("NO_RESET"),
   .AUTORESET_PRIORITY("RESET"),
   .MASK(48'h3fffffffffff),
   .PATTERN(48'h000000000000),
   .SEL_MASK("MASK"),
   .SEL_PATTERN("PATTERN"),
   .USE_PATTERN_DETECT("NO_PATDET"),
   
   // Programmable Inversion Attributes: (全部不反向)
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
   
   // Register Control Attributes: Pipeline Register Configuration
   .ACASCREG(1),           // (未使用级联,但必须设置等于小于AREG)
   .ADREG(1),              // Pipeline stages for pre-adder (0-1)
   .ALUMODEREG(1),         // 控制信号寄存器
   .AREG(1),               // A 输入寄存器 (A pipline stage,一个在加法器前，一个在乘法器前，同时并行送入加法器和乘法器)
   .BCASCREG(1),           // (未使用级联,但必须B置等于小于BREG)
   .BREG(2),               // B 输入寄存器 (B pipline stage,一个在加法器前，一个在乘法器前，同时并行送入加法器和乘法器)
   .CARRYINREG(1),         // CARRYIN 寄存器，设为1
   .CARRYINSELREG(1),      // 控制信号寄存器，设为1
   .CREG(1),               // C 输入寄存器 
   .DREG(1),               // 
   .INMODEREG(1),          // 控制信号寄存器，设为1
   .MREG(1),               // 乘法器输出寄存器 (Tier 5)
   .OPMODEREG(1),          // 控制信号寄存器，设为1
   .PREG(1)                // P 输出寄存器 (Tier 6)
)
dsp_inst (
   // Cascade outputs: (未使用)
   .ACOUT(),
   .BCOUT(),
   .CARRYCASCOUT(),
   .MULTSIGNOUT(),
   .PCOUT(),
   
   // Control outputs: (未使用)
   .OVERFLOW(),
   .PATTERNBDETECT(),
   .PATTERNDETECT(),
   .UNDERFLOW(),
   
   // Data outputs:
   .CARRYOUT(),            // (未使用)
   .P(dsp_P),                // 48-bit output: P[47:0]
   .XOROUT(),              // (未使用)
   
   // Cascade inputs: (未使用, 必须接地)
   .ACIN(30'b0),
   .BCIN(18'b0),
   .CARRYCASCIN(1'b0),
   .MULTSIGNIN(1'b0),
   .PCIN(48'b0),
   
   // Control inputs:
   // ALUMODE 设为 '0000' (Z + (W+X+Y+CIN)) 
   .ALUMODE(4'b0000),
   .CARRYINSEL(3'b000),      // 核心：选择 CARRYIN 端口
   .CLK(clk),                  // 1-bit input: Clock
   // INMODE[3]=0 (加) , INMODE[2]=1 (用D) , INMODE[1]=0 (用A) , INMODE[0]=0 (用A2) 
   .INMODE(5'b00101),
   // OPMODE 保持不变，它正确地选择了 W=0 [cite: 873], X=M [cite: 876], Y=M [cite: 878], Z=C [cite: 880]
   .OPMODE(9'b000110101),
   
   // Data inputs:
   .A(dsp_A),         // 30-bit input: A[29:0]. dsp_A[26:0] 送入 A[26:0]
   .B(dsp_B),                // 18-bit input: B[17:0]
   .C(dsp_C),                // 48-bit input: C[47:0]
   // 关键修改：CARRYIN 设为 0，因为我们执行 M + C + 0
   .CARRYIN(1'b0),
   // 关键修改：连接D端口
   .D(dsp_D),                // 27-bit input: D data (假定您在外部定义了 dsp_D)
   
   // Reset/Clock Enable inputs:
   .CEA1(1'b1),//直接将A送入加法器
   .CEA2(1'b0),//直接讲A送入乘法器
   // 关键修改：启用D路径的时钟
   .CEAD(1'b1),
   .CEALUMODE(1'b1),
   .CEB1(1'b1),//B送入加法器
   .CEB2(1'b1),//B送入乘法器
   .CEC(1'b1),
   .CECARRYIN(1'b1),
   .CECTRL(1'b1),
   // 关键修改：启用D路径的时钟
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


    //-----------------------------------------------------------------
    // Result extraction (roff={0,11,22,33})
    //-----------------------------------------------------------------
    assign r00 = $signed(dsp_P[7:0]);       // bits [7:0]
    assign r10 = $signed(dsp_P[18:11]);     // bits [18:11]
    assign r01 = $signed(dsp_P[29:22]);     // bits [29:22]
    assign r11 = $signed(dsp_P[40:33]);     // bits [40:33]

endmodule
