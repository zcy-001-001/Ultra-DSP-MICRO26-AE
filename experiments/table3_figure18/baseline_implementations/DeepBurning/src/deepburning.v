module INT4_INT4_DEEPBURNING6(
    input  wire               clk,

    // Signed INT4 operands.
    input  wire signed [3:0]  w1,
    input  wire signed [3:0]  w2,
    input  wire signed [3:0]  a1,
    input  wire signed [3:0]  a2,
    input  wire signed [3:0]  a3,

    // Signed INT8 products.
    output wire signed [7:0]  result1,   // w1 * a1
    output wire signed [7:0]  result2,   // w1 * a2
    output wire signed [7:0]  result3,   // w1 * a3
    output wire signed [7:0]  result4,   // w2 * a1
    output wire signed [7:0]  result5,   // w2 * a2
    output wire signed [7:0]  result6    // w2 * a3
);

    // Strict paper-literal implementation of ICCAD23 1-bit overpacking:
    // - packed signed operands use direct sign extension
    // - LSB pollution is corrected by adding the recalculated B_H_LS
    // - high-position correction uses overlap_bit ^ B_H_LS
    //
    // Packing geometry:
    //   A-side: w1 @ bit 0,  w2 @ bit 21
    //   B-side: a1 @ bit 0,  a2 @ bit 7,  a3 @ bit 14
    //   Result windows: [7:0], [14:7], [21:14], [28:21], [35:28], [42:35]
    //
    // This file intentionally follows the correction flow described in
    // `Baseline/Deep-Burning/ICCAD23.pdf` Section IV-B-1 only. No extra
    // preprocess/postprocess or carry-specific compensation is added here.

    // -----------------------------
    // Packed signed operands
    // -----------------------------
    wire signed [26:0] dsp_A_pack;
    wire signed [17:0] dsp_B_pack;

    assign dsp_A_pack = {
        {2{w2[3]}}, w2,         // bits [26:21], signed w2
        {17{w1[3]}}, w1         // bits [20:0],  signed w1
    };

    assign dsp_B_pack = {
        a3,                     // bits [17:14], signed a3
        {3{a2[3]}}, a2,         // bits [13:7],  signed a2
        {3{a1[3]}}, a1          // bits [6:0],   signed a1
    };

    // -----------------------------
    // Recalculated B_H_LS for each high-position segment
    // -----------------------------
    wire bh_ls_r2_comb;
    wire bh_ls_r3_comb;
    wire bh_ls_r4_comb;
    wire bh_ls_r5_comb;
    wire bh_ls_r6_comb;

    assign bh_ls_r2_comb = w1[0] & a2[0];
    assign bh_ls_r3_comb = w1[0] & a3[0];
    assign bh_ls_r4_comb = w2[0] & a1[0];
    assign bh_ls_r5_comb = w2[0] & a2[0];
    assign bh_ls_r6_comb = w2[0] & a3[0];

    // -----------------------------
    // Single packed DSP multiply
    // -----------------------------
    (* use_dsp = "yes", keep = "true" *)
    reg signed [44:0] dsp_mult_r1;

    reg bh_ls_r2_r1;
    reg bh_ls_r3_r1;
    reg bh_ls_r4_r1;
    reg bh_ls_r5_r1;
    reg bh_ls_r6_r1;

    always @(posedge clk) begin
        dsp_mult_r1 <= $signed(dsp_A_pack) * $signed(dsp_B_pack);

        bh_ls_r2_r1 <= bh_ls_r2_comb;
        bh_ls_r3_r1 <= bh_ls_r3_comb;
        bh_ls_r4_r1 <= bh_ls_r4_comb;
        bh_ls_r5_r1 <= bh_ls_r5_comb;
        bh_ls_r6_r1 <= bh_ls_r6_comb;
    end

    wire signed [47:0] dsp_P_r1;
    assign dsp_P_r1 = {{3{dsp_mult_r1[44]}}, dsp_mult_r1};

    // -----------------------------
    // Raw overlapped INT8 windows
    // -----------------------------
    wire signed [7:0] r1_raw;
    wire signed [7:0] r2_raw;
    wire signed [7:0] r3_raw;
    wire signed [7:0] r4_raw;
    wire signed [7:0] r5_raw;
    wire signed [7:0] r6_raw;

    assign r1_raw = dsp_P_r1[7:0];
    assign r2_raw = dsp_P_r1[14:7];
    assign r3_raw = dsp_P_r1[21:14];
    assign r4_raw = dsp_P_r1[28:21];
    assign r5_raw = dsp_P_r1[35:28];
    assign r6_raw = dsp_P_r1[42:35];

    // -----------------------------
    // Overlapped physical bits
    // -----------------------------
    wire ov_r1_r2;
    wire ov_r2_r3;
    wire ov_r3_r4;
    wire ov_r4_r5;
    wire ov_r5_r6;

    assign ov_r1_r2 = dsp_P_r1[7];
    assign ov_r2_r3 = dsp_P_r1[14];
    assign ov_r3_r4 = dsp_P_r1[21];
    assign ov_r4_r5 = dsp_P_r1[28];
    assign ov_r5_r6 = dsp_P_r1[35];

    // -----------------------------
    // Paper-literal high-position correction:
    // overlap_bit ^ B_H_LS
    // -----------------------------
    wire msb_fix_r2;
    wire msb_fix_r3;
    wire msb_fix_r4;
    wire msb_fix_r5;
    wire msb_fix_r6;

    assign msb_fix_r2 = ov_r1_r2 ^ bh_ls_r2_r1;
    assign msb_fix_r3 = ov_r2_r3 ^ bh_ls_r3_r1;
    assign msb_fix_r4 = ov_r3_r4 ^ bh_ls_r4_r1;
    assign msb_fix_r5 = ov_r4_r5 ^ bh_ls_r5_r1;
    assign msb_fix_r6 = ov_r5_r6 ^ bh_ls_r6_r1;

    // -----------------------------
    // Paper-literal segment correction
    // - low-position segment: add B_H_LS to the polluted MSB
    // - high-position segment: add overlap_bit ^ B_H_LS
    // -----------------------------
    (* use_dsp = "no" *) wire signed [7:0] r1_corr;
    (* use_dsp = "no" *) wire signed [7:0] r2_corr;
    (* use_dsp = "no" *) wire signed [7:0] r3_corr;
    (* use_dsp = "no" *) wire signed [7:0] r4_corr;
    (* use_dsp = "no" *) wire signed [7:0] r5_corr;
    (* use_dsp = "no" *) wire signed [7:0] r6_corr;

    assign r1_corr = r1_raw
                   + {bh_ls_r2_r1, 7'b0};

    assign r2_corr = r2_raw
                   + {bh_ls_r3_r1, 7'b0}
                   + {7'b0, msb_fix_r2};

    assign r3_corr = r3_raw
                   + {bh_ls_r4_r1, 7'b0}
                   + {7'b0, msb_fix_r3};

    assign r4_corr = r4_raw
                   + {bh_ls_r5_r1, 7'b0}
                   + {7'b0, msb_fix_r4};

    assign r5_corr = r5_raw
                   + {bh_ls_r6_r1, 7'b0}
                   + {7'b0, msb_fix_r5};

    assign r6_corr = r6_raw
                   + {7'b0, msb_fix_r6};

    assign result1 = r1_corr;
    assign result2 = r2_corr;
    assign result3 = r3_corr;
    assign result4 = r4_corr;
    assign result5 = r5_corr;
    assign result6 = r6_corr;

endmodule
