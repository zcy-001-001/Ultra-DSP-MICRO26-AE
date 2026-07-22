// V1 resource target: sign-magnitude decoupling, no output-domain overlap.
//
// This replaces the removed behavioral synthesis target.  Vivado inferred two
// DSP48E2 cells from that target's muxed multiplication expression.  This
// version explicitly instantiates one DSP48E2 through w4a4_dsp48e2_m_sub_c.
//
// P mode layout:
//   A positions: w1@0, w2@12, w3@24
//   B positions: a1@0, a2@6
//   products: 3 x 2 = 6, at offsets 0, 6, 12, 18, 24, 30.
// D mode layout:
//   A positions: w1@0, w2@6, w3@12, w4@18, w5@24
//   B positions: a1@0
//   products: 5 x 1 = 5, at offsets 0, 6, 12, 18, 24.
module w4a4_sf_v1_signmag_nonoverlap_p6d5_single_dsp(
    input  wire        clk,
    input  wire        mode,
    input  wire [3:0]  w1, w2, w3, w4, w5, w6, w7,
    input  wire [3:0]  a1, a2, a3,
    output wire [6:0]  result1, result2, result3, result4, result5,
    output wire [6:0]  result6, result7, result8, result9, result10,
    output wire        sign1, sign2, sign3, sign4, sign5,
    output wire        sign6, sign7, sign8, sign9, sign10,
    output wire [3:0]  valid_count
);

function [2:0] to_mag3;
    input [3:0] x;
    reg [3:0] temp;
    begin
        temp = x[3] ? (~x + 1'b1) : x;
        to_mag3 = (temp == 4'b1000) ? 3'b111 : temp[2:0];
    end
endfunction

wire [2:0] w1_mag = to_mag3(w1);
wire [2:0] w2_mag = to_mag3(w2);
wire [2:0] w3_mag = to_mag3(w3);
wire [2:0] w4_mag = to_mag3(w4);
wire [2:0] w5_mag = to_mag3(w5);
wire [2:0] a1_mag = to_mag3(a1);
wire [2:0] a2_mag = to_mag3(a2);

wire [26:0] dsp_a_prefill = {w3_mag, 9'b0, w2_mag, 9'b0, w1_mag};
wire [17:0] dsp_b_prefill = {9'b0, a2_mag, 3'b0, a1_mag};
wire [26:0] dsp_a_decode  = {w5_mag, 3'b0, w4_mag, 3'b0, w3_mag, 3'b0, w2_mag, 3'b0, w1_mag};
wire [17:0] dsp_b_decode  = {15'b0, a1_mag};

wire [26:0] dsp_a = mode ? dsp_a_decode : dsp_a_prefill;
wire [17:0] dsp_b = mode ? dsp_b_decode : dsp_b_prefill;
wire [47:0] dsp_p;

w4a4_dsp48e2_m_sub_c dsp_core (
    .clk(clk),
    .dsp_a(dsp_a),
    .dsp_b(dsp_b),
    .dsp_c(48'b0),
    .dsp_p(dsp_p)
);

wire [9:0] signs_prefill;
assign signs_prefill[0] = w1[3] ^ a1[3];
assign signs_prefill[1] = w1[3] ^ a2[3];
assign signs_prefill[2] = w2[3] ^ a1[3];
assign signs_prefill[3] = w2[3] ^ a2[3];
assign signs_prefill[4] = w3[3] ^ a1[3];
assign signs_prefill[5] = w3[3] ^ a2[3];
assign signs_prefill[6] = 1'b0;
assign signs_prefill[7] = 1'b0;
assign signs_prefill[8] = 1'b0;
assign signs_prefill[9] = 1'b0;

wire [9:0] signs_decode;
assign signs_decode[0] = w1[3] ^ a1[3];
assign signs_decode[1] = w2[3] ^ a1[3];
assign signs_decode[2] = w3[3] ^ a1[3];
assign signs_decode[3] = w4[3] ^ a1[3];
assign signs_decode[4] = w5[3] ^ a1[3];
assign signs_decode[5] = 1'b0;
assign signs_decode[6] = 1'b0;
assign signs_decode[7] = 1'b0;
assign signs_decode[8] = 1'b0;
assign signs_decode[9] = 1'b0;

wire [9:0] signs_comb = mode ? signs_decode : signs_prefill;
wire [3:0] valid_count_comb = mode ? 4'd5 : 4'd6;

(* shreg_extract = "no" *) reg mode_r1, mode_r2, mode_r3;
(* shreg_extract = "no" *) reg [9:0] signs_r1, signs_r2, signs_r3;
(* shreg_extract = "no" *) reg [3:0] valid_count_r1, valid_count_r2, valid_count_r3;

always @(posedge clk) begin
    mode_r1 <= mode;
    signs_r1 <= signs_comb;
    valid_count_r1 <= valid_count_comb;
end

always @(posedge clk) begin
    mode_r2 <= mode_r1;
    signs_r2 <= signs_r1;
    valid_count_r2 <= valid_count_r1;
end

always @(posedge clk) begin
    mode_r3 <= mode_r2;
    signs_r3 <= signs_r2;
    valid_count_r3 <= valid_count_r2;
end

wire [5:0] mag1  = dsp_p[5:0];
wire [5:0] mag2  = dsp_p[11:6];
wire [5:0] mag3  = dsp_p[17:12];
wire [5:0] mag4  = dsp_p[23:18];
wire [5:0] mag5  = dsp_p[29:24];
wire [5:0] mag6  = mode_r3 ? 6'b0 : dsp_p[35:30];
wire [5:0] mag7  = 6'b0;
wire [5:0] mag8  = 6'b0;
wire [5:0] mag9  = 6'b0;
wire [5:0] mag10 = 6'b0;

assign result1  = {1'b0, mag1}  ^ {7{signs_r3[0]}};
assign result2  = {1'b0, mag2}  ^ {7{signs_r3[1]}};
assign result3  = {1'b0, mag3}  ^ {7{signs_r3[2]}};
assign result4  = {1'b0, mag4}  ^ {7{signs_r3[3]}};
assign result5  = {1'b0, mag5}  ^ {7{signs_r3[4]}};
assign result6  = {1'b0, mag6}  ^ {7{signs_r3[5]}};
assign result7  = {1'b0, mag7}  ^ {7{signs_r3[6]}};
assign result8  = {1'b0, mag8}  ^ {7{signs_r3[7]}};
assign result9  = {1'b0, mag9}  ^ {7{signs_r3[8]}};
assign result10 = {1'b0, mag10} ^ {7{signs_r3[9]}};

assign {sign10, sign9, sign8, sign7, sign6, sign5, sign4, sign3, sign2, sign1} = signs_r3;
assign valid_count = valid_count_r3;

endmodule
