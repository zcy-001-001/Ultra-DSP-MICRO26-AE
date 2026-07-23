// LUT:39  FF:12
module W3A5(
    input  wire                     clock,
    input  wire                     reset,
    input  wire signed [2:0]   wArr_in_0,
    input  wire signed [2:0]   wArr_in_1,
    input  wire signed [2:0]   wArr_in_2,
    input  wire signed [4:0]   aArr_in_0,
    input  wire signed [4:0]   aArr_in_1,
    input  wire                     valid_in,
    output reg                      valid_out,
    output wire signed [7:0]   pArr_out_0,
    output wire signed [7:0]   pArr_out_1,
    output wire signed [7:0]   pArr_out_2,
    output wire signed [7:0]   pArr_out_3,
    output wire signed [7:0]   pArr_out_4,
    output wire signed [7:0]   pArr_out_5
);

    // reset is unused; kept for compatibility with Chisel-generated top ports.

    localparam [2:0] POST_BIAS = 3'd4;

    wire signed [2:0] neg_w_0 = -wArr_in_0;
    wire signed [2:0] neg_w_1 = -wArr_in_1;
    wire signed [2:0] neg_w_2 = -wArr_in_2;
    wire signed [4:0] neg_a_0 = -aArr_in_0;
    wire signed [4:0] neg_a_1 = -aArr_in_1;

    wire [6:0] packedElem_0 = {5'd0, wArr_in_0[1:0]};
    wire [6:0] packedElem_1 = {5'd0, wArr_in_1[1:0]};
    wire [6:0] packedElem_2 = {5'd0, wArr_in_2[1:0]};
    wire signed [20:0] packedportS = {packedElem_2, packedElem_1, packedElem_0};

    wire [20:0] packedL_0 = {16'd0, aArr_in_0[4:0]};
    wire [20:0] packedL_1 = {16'd0, aArr_in_1[4:0]};
    wire signed [41:0] packedportL = {packedL_1, packedL_0};

    wire [6:0] preprocess_0 = (wArr_in_0 < 0) ? {neg_a_0[4:0], 2'd0} : 7'd0;
    wire [6:0] preprocess_1 = (wArr_in_1 < 0) ? {neg_a_0[4:0], 2'd0} : 7'd0;
    wire [6:0] preprocess_2 = (wArr_in_2 < 0) ? {neg_a_0[4:0], 2'd0} : 7'd0;
    wire [6:0] preprocess_3 = (wArr_in_0 < 0) ? {neg_a_1[4:0], 2'd0} : 7'd0;
    wire [6:0] preprocess_4 = (wArr_in_1 < 0) ? {neg_a_1[4:0], 2'd0} : 7'd0;
    wire [6:0] preprocess_5 = (wArr_in_2 < 0) ? {neg_a_1[4:0], 2'd0} : 7'd0;
    wire [41:0] packedpreprocess = {preprocess_5, preprocess_4, preprocess_3, preprocess_2, preprocess_1, preprocess_0};

    reg  [2:0] postprocessArr_0;
    reg  [2:0] postprocessArr_1;
    reg  [2:0] postprocessArr_2;
    reg  signed [4:0] aArr_in_Reg_0;
    reg  signed [4:0] aArr_in_Reg_1;
    reg  signed [44:0] packedProduct;

    wire signed [17:0] packedportS_dsp = packedportS[17:0];
    wire signed [26:0] packedportL_dsp = packedportL[26:0];
    wire signed [44:0] packedpreprocess_dsp = $signed({{3{1'b0}}, packedpreprocess});

    always @(posedge clock) begin
        valid_out <= valid_in;
        if (wArr_in_0 < 0)
            postprocessArr_0 <= neg_w_0[2:0] - POST_BIAS;
        else
            postprocessArr_0 <= neg_w_0[2:0];
        if (wArr_in_1 < 0)
            postprocessArr_1 <= neg_w_1[2:0] - POST_BIAS;
        else
            postprocessArr_1 <= neg_w_1[2:0];
        if (wArr_in_2 < 0)
            postprocessArr_2 <= neg_w_2[2:0] - POST_BIAS;
        else
            postprocessArr_2 <= neg_w_2[2:0];
        aArr_in_Reg_0 <= aArr_in_0;
        aArr_in_Reg_1 <= aArr_in_1;
        packedProduct <= (packedportS_dsp * packedportL_dsp) + packedpreprocess_dsp;
    end

    wire signed [6:0] unpackedProduct_0 = packedProduct[6:0];
    wire [2:0] unpackedProductZext_0 = {1'b0, unpackedProduct_0[6:5]};
    wire [3:0] sum_0 = unpackedProductZext_0 + postprocessArr_0;
    assign pArr_out_0 = (aArr_in_Reg_0 < 0) ?
                          $signed({sum_0[2:0], unpackedProduct_0[4:0]}) :
                          $signed({unpackedProduct_0[6], unpackedProduct_0});

    wire signed [6:0] unpackedProduct_1 = packedProduct[13:7];
    wire [2:0] unpackedProductZext_1 = {1'b0, unpackedProduct_1[6:5]};
    wire [3:0] sum_1 = unpackedProductZext_1 + postprocessArr_1;
    assign pArr_out_1 = (aArr_in_Reg_0 < 0) ?
                          $signed({sum_1[2:0], unpackedProduct_1[4:0]}) :
                          $signed({unpackedProduct_1[6], unpackedProduct_1});

    wire signed [6:0] unpackedProduct_2 = packedProduct[20:14];
    wire [2:0] unpackedProductZext_2 = {1'b0, unpackedProduct_2[6:5]};
    wire [3:0] sum_2 = unpackedProductZext_2 + postprocessArr_2;
    assign pArr_out_2 = (aArr_in_Reg_0 < 0) ?
                          $signed({sum_2[2:0], unpackedProduct_2[4:0]}) :
                          $signed({unpackedProduct_2[6], unpackedProduct_2});

    wire signed [6:0] unpackedProduct_3 = packedProduct[27:21];
    wire [2:0] unpackedProductZext_3 = {1'b0, unpackedProduct_3[6:5]};
    wire [3:0] sum_3 = unpackedProductZext_3 + postprocessArr_0;
    assign pArr_out_3 = (aArr_in_Reg_1 < 0) ?
                          $signed({sum_3[2:0], unpackedProduct_3[4:0]}) :
                          $signed({unpackedProduct_3[6], unpackedProduct_3});

    wire signed [6:0] unpackedProduct_4 = packedProduct[34:28];
    wire [2:0] unpackedProductZext_4 = {1'b0, unpackedProduct_4[6:5]};
    wire [3:0] sum_4 = unpackedProductZext_4 + postprocessArr_1;
    assign pArr_out_4 = (aArr_in_Reg_1 < 0) ?
                          $signed({sum_4[2:0], unpackedProduct_4[4:0]}) :
                          $signed({unpackedProduct_4[6], unpackedProduct_4});

    wire signed [6:0] unpackedProduct_5 = packedProduct[41:35];
    wire [2:0] unpackedProductZext_5 = {1'b0, unpackedProduct_5[6:5]};
    wire [3:0] sum_5 = unpackedProductZext_5 + postprocessArr_2;
    assign pArr_out_5 = (aArr_in_Reg_1 < 0) ?
                          $signed({sum_5[2:0], unpackedProduct_5[4:0]}) :
                          $signed({unpackedProduct_5[6], unpackedProduct_5});

endmodule