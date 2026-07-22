/*
 * Auto-generated for the Ultra-DSP rebuttal LSB-depth sweep.
 * Top module: W5A4_lsb_depth5
 * Precision: W5A4
 * Magnitude widths: W=4, A=3
 * Depth: 5
 *
 * This isolates one low-product-bit correction generator. It builds only the
 * low DEPTH bits with a truncated partial-product network, so Vivado can expose
 * the incremental LUT cost of increasing LSB depth.
 */

(* keep_hierarchy = "yes", use_dsp = "no" *)
module W5A4_lsb_depth5(
    input wire [3:0] w_mag,
    input wire [2:0] a_mag,
    output wire [4:0] lsb
);

wire [4:0] partial_0 = w_mag[0] ? {1'b0, 1'b0, a_mag[2], a_mag[1], a_mag[0]} : 5'b0;
wire [4:0] partial_1 = w_mag[1] ? {1'b0, a_mag[2], a_mag[1], a_mag[0], 1'b0} : 5'b0;
wire [4:0] partial_2 = w_mag[2] ? {a_mag[2], a_mag[1], a_mag[0], 1'b0, 1'b0} : 5'b0;
wire [4:0] partial_3 = w_mag[3] ? {a_mag[1], a_mag[0], 1'b0, 1'b0, 1'b0} : 5'b0;

wire [4:0] sum_0 = partial_0;
wire [4:0] sum_1 = sum_0 + partial_1;
wire [4:0] sum_2 = sum_1 + partial_2;
wire [4:0] sum_3 = sum_2 + partial_3;
assign lsb = sum_3;

endmodule
