// LUT:42  FF:13
module W5A5(
    input  wire                     clock,
    input  wire                     reset,
    input  wire signed [4:0]   wArr_in_0,
    input  wire signed [4:0]   wArr_in_1,
    input  wire signed [4:0]   aArr_in_0,
    input  wire signed [4:0]   aArr_in_1,
    input  wire                     valid_in,
    output reg                      valid_out,
    output wire signed [9:0]   pArr_out_0,
    output wire signed [9:0]   pArr_out_1,
    output wire signed [9:0]   pArr_out_2,
    output wire signed [9:0]   pArr_out_3
);

    // reset is unused; kept for compatibility with Chisel-generated top ports.

    localparam [4:0] POST_BIAS = 5'd16;

    wire signed [4:0] neg_w_0 = -wArr_in_0;
    wire signed [4:0] neg_w_1 = -wArr_in_1;
    wire signed [4:0] neg_a_0 = -aArr_in_0;
    wire signed [4:0] neg_a_1 = -aArr_in_1;

    wire [8:0] packedElem_0 = {5'd0, wArr_in_0[3:0]};
    wire [8:0] packedElem_1 = {5'd0, wArr_in_1[3:0]};
    wire signed [17:0] packedportS = {packedElem_1, packedElem_0};

    wire [17:0] packedL_0 = {13'd0, aArr_in_0[4:0]};
    wire [17:0] packedL_1 = {13'd0, aArr_in_1[4:0]};
    wire signed [35:0] packedportL = {packedL_1, packedL_0};

    wire [8:0] preprocess_0 = (wArr_in_0 < 0) ? {neg_a_0[4:0], 4'd0} : 9'd0;
    wire [8:0] preprocess_1 = (wArr_in_1 < 0) ? {neg_a_0[4:0], 4'd0} : 9'd0;
    wire [8:0] preprocess_2 = (wArr_in_0 < 0) ? {neg_a_1[4:0], 4'd0} : 9'd0;
    wire [8:0] preprocess_3 = (wArr_in_1 < 0) ? {neg_a_1[4:0], 4'd0} : 9'd0;
    wire [35:0] packedpreprocess = {preprocess_3, preprocess_2, preprocess_1, preprocess_0};

    reg  [4:0] postprocessArr_0;
    reg  [4:0] postprocessArr_1;
    reg  signed [4:0] aArr_in_Reg_0;
    reg  signed [4:0] aArr_in_Reg_1;
    reg  signed [44:0] packedProduct;

    wire signed [17:0] packedportS_dsp = packedportS[17:0];
    wire signed [26:0] packedportL_dsp = packedportL[26:0];
    wire signed [44:0] packedpreprocess_dsp = $signed({{9{1'b0}}, packedpreprocess});

    always @(posedge clock) begin
        valid_out <= valid_in;
        if (wArr_in_0 < 0)
            postprocessArr_0 <= neg_w_0[4:0] - POST_BIAS;
        else
            postprocessArr_0 <= neg_w_0[4:0];
        if (wArr_in_1 < 0)
            postprocessArr_1 <= neg_w_1[4:0] - POST_BIAS;
        else
            postprocessArr_1 <= neg_w_1[4:0];
        aArr_in_Reg_0 <= aArr_in_0;
        aArr_in_Reg_1 <= aArr_in_1;
        packedProduct <= (packedportS_dsp * packedportL_dsp) + packedpreprocess_dsp;
    end

    wire signed [8:0] unpackedProduct_0 = packedProduct[8:0];
    wire [4:0] unpackedProductZext_0 = {1'b0, unpackedProduct_0[8:5]};
    wire [5:0] sum_0 = unpackedProductZext_0 + postprocessArr_0;
    assign pArr_out_0 = (aArr_in_Reg_0 < 0) ?
                          $signed({sum_0[4:0], unpackedProduct_0[4:0]}) :
                          $signed({unpackedProduct_0[8], unpackedProduct_0});

    wire signed [8:0] unpackedProduct_1 = packedProduct[17:9];
    wire [4:0] unpackedProductZext_1 = {1'b0, unpackedProduct_1[8:5]};
    wire [5:0] sum_1 = unpackedProductZext_1 + postprocessArr_1;
    assign pArr_out_1 = (aArr_in_Reg_0 < 0) ?
                          $signed({sum_1[4:0], unpackedProduct_1[4:0]}) :
                          $signed({unpackedProduct_1[8], unpackedProduct_1});

    wire signed [8:0] unpackedProduct_2 = packedProduct[26:18];
    wire [4:0] unpackedProductZext_2 = {1'b0, unpackedProduct_2[8:5]};
    wire [5:0] sum_2 = unpackedProductZext_2 + postprocessArr_0;
    assign pArr_out_2 = (aArr_in_Reg_1 < 0) ?
                          $signed({sum_2[4:0], unpackedProduct_2[4:0]}) :
                          $signed({unpackedProduct_2[8], unpackedProduct_2});

    wire signed [8:0] unpackedProduct_3 = packedProduct[35:27];
    wire [4:0] unpackedProductZext_3 = {1'b0, unpackedProduct_3[8:5]};
    wire [5:0] sum_3 = unpackedProductZext_3 + postprocessArr_1;
    assign pArr_out_3 = (aArr_in_Reg_1 < 0) ?
                          $signed({sum_3[4:0], unpackedProduct_3[4:0]}) :
                          $signed({unpackedProduct_3[8], unpackedProduct_3});

endmodule