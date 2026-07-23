//// LUT:45  FF:15
module W3A4(
    input  wire                     clock,
    input  wire                     reset,
    input  wire signed [3:0]   wArr_in_0,
    input  wire signed [3:0]   wArr_in_1,
    input  wire signed [3:0]   wArr_in_2,
    input  wire signed [2:0]   aArr_in_0,
    input  wire signed [2:0]   aArr_in_1,
    input  wire                     valid_in,
    output reg                      valid_out,
    output wire signed [6:0]   pArr_out_0,
    output wire signed [6:0]   pArr_out_1,
    output wire signed [6:0]   pArr_out_2,
    output wire signed [6:0]   pArr_out_3,
    output wire signed [6:0]   pArr_out_4,
    output wire signed [6:0]   pArr_out_5
);

    // reset is unused; kept for compatibility with Chisel-generated top ports.

    localparam [3:0] POST_BIAS = 4'd8;

    wire signed [3:0] neg_w_0 = -wArr_in_0;
    wire signed [3:0] neg_w_1 = -wArr_in_1;
    wire signed [3:0] neg_w_2 = -wArr_in_2;
    wire signed [2:0] neg_a_0 = -aArr_in_0;
    wire signed [2:0] neg_a_1 = -aArr_in_1;

    wire [5:0] packedElem_0 = {3'd0, wArr_in_0[2:0]};
    wire [5:0] packedElem_1 = {3'd0, wArr_in_1[2:0]};
    wire [5:0] packedElem_2 = {3'd0, wArr_in_2[2:0]};
    wire signed [17:0] packedportS = {packedElem_2, packedElem_1, packedElem_0};

    wire [17:0] packedL_0 = {15'd0, aArr_in_0[2:0]};
    wire [17:0] packedL_1 = {15'd0, aArr_in_1[2:0]};
    wire signed [35:0] packedportL = {packedL_1, packedL_0};

    wire [5:0] preprocess_0 = (wArr_in_0 < 0) ? {neg_a_0[2:0], 3'd0} : 6'd0;
    wire [5:0] preprocess_1 = (wArr_in_1 < 0) ? {neg_a_0[2:0], 3'd0} : 6'd0;
    wire [5:0] preprocess_2 = (wArr_in_2 < 0) ? {neg_a_0[2:0], 3'd0} : 6'd0;
    wire [5:0] preprocess_3 = (wArr_in_0 < 0) ? {neg_a_1[2:0], 3'd0} : 6'd0;
    wire [5:0] preprocess_4 = (wArr_in_1 < 0) ? {neg_a_1[2:0], 3'd0} : 6'd0;
    wire [5:0] preprocess_5 = (wArr_in_2 < 0) ? {neg_a_1[2:0], 3'd0} : 6'd0;
    wire [35:0] packedpreprocess = {preprocess_5, preprocess_4, preprocess_3, preprocess_2, preprocess_1, preprocess_0};

    reg  [3:0] postprocessArr_0;
    reg  [3:0] postprocessArr_1;
    reg  [3:0] postprocessArr_2;
    reg  signed [2:0] aArr_in_Reg_0;
    reg  signed [2:0] aArr_in_Reg_1;
    reg  signed [44:0] packedProduct;

    wire signed [17:0] packedportS_dsp = packedportS[17:0];
    wire signed [26:0] packedportL_dsp = packedportL[26:0];
    wire signed [44:0] packedpreprocess_dsp = $signed({{9{1'b0}}, packedpreprocess});

    always @(posedge clock) begin
        valid_out <= valid_in;
        if (wArr_in_0 < 0)
            postprocessArr_0 <= neg_w_0[3:0] - POST_BIAS;
        else
            postprocessArr_0 <= neg_w_0[3:0];
        if (wArr_in_1 < 0)
            postprocessArr_1 <= neg_w_1[3:0] - POST_BIAS;
        else
            postprocessArr_1 <= neg_w_1[3:0];
        if (wArr_in_2 < 0)
            postprocessArr_2 <= neg_w_2[3:0] - POST_BIAS;
        else
            postprocessArr_2 <= neg_w_2[3:0];
        aArr_in_Reg_0 <= aArr_in_0;
        aArr_in_Reg_1 <= aArr_in_1;
        packedProduct <= (packedportS_dsp * packedportL_dsp) + packedpreprocess_dsp;
    end

    wire signed [5:0] unpackedProduct_0 = packedProduct[5:0];
    wire [3:0] unpackedProductZext_0 = {1'b0, unpackedProduct_0[5:3]};
    wire [4:0] sum_0 = unpackedProductZext_0 + postprocessArr_0;
    assign pArr_out_0 = (aArr_in_Reg_0 < 0) ?
                          $signed({sum_0[3:0], unpackedProduct_0[2:0]}) :
                          $signed({unpackedProduct_0[5], unpackedProduct_0});

    wire signed [5:0] unpackedProduct_1 = packedProduct[11:6];
    wire [3:0] unpackedProductZext_1 = {1'b0, unpackedProduct_1[5:3]};
    wire [4:0] sum_1 = unpackedProductZext_1 + postprocessArr_1;
    assign pArr_out_1 = (aArr_in_Reg_0 < 0) ?
                          $signed({sum_1[3:0], unpackedProduct_1[2:0]}) :
                          $signed({unpackedProduct_1[5], unpackedProduct_1});

    wire signed [5:0] unpackedProduct_2 = packedProduct[17:12];
    wire [3:0] unpackedProductZext_2 = {1'b0, unpackedProduct_2[5:3]};
    wire [4:0] sum_2 = unpackedProductZext_2 + postprocessArr_2;
    assign pArr_out_2 = (aArr_in_Reg_0 < 0) ?
                          $signed({sum_2[3:0], unpackedProduct_2[2:0]}) :
                          $signed({unpackedProduct_2[5], unpackedProduct_2});

    wire signed [5:0] unpackedProduct_3 = packedProduct[23:18];
    wire [3:0] unpackedProductZext_3 = {1'b0, unpackedProduct_3[5:3]};
    wire [4:0] sum_3 = unpackedProductZext_3 + postprocessArr_0;
    assign pArr_out_3 = (aArr_in_Reg_1 < 0) ?
                          $signed({sum_3[3:0], unpackedProduct_3[2:0]}) :
                          $signed({unpackedProduct_3[5], unpackedProduct_3});

    wire signed [5:0] unpackedProduct_4 = packedProduct[29:24];
    wire [3:0] unpackedProductZext_4 = {1'b0, unpackedProduct_4[5:3]};
    wire [4:0] sum_4 = unpackedProductZext_4 + postprocessArr_1;
    assign pArr_out_4 = (aArr_in_Reg_1 < 0) ?
                          $signed({sum_4[3:0], unpackedProduct_4[2:0]}) :
                          $signed({unpackedProduct_4[5], unpackedProduct_4});

    wire signed [5:0] unpackedProduct_5 = packedProduct[35:30];
    wire [3:0] unpackedProductZext_5 = {1'b0, unpackedProduct_5[5:3]};
    wire [4:0] sum_5 = unpackedProductZext_5 + postprocessArr_2;
    assign pArr_out_5 = (aArr_in_Reg_1 < 0) ?
                          $signed({sum_5[3:0], unpackedProduct_5[2:0]}) :
                          $signed({unpackedProduct_5[5], unpackedProduct_5});

endmodule