/*
 * Auto-generated for the Ultra-DSP rebuttal LSB-depth sweep.
 * Top module: W4A5_lsb_depth6
 * Precision: W4A5
 * Magnitude widths: W=3, A=4
 * Depth: 6
 *
 * This isolates one low-product-bit correction generator. It builds only the
 * low DEPTH bits with a truncated partial-product network, so Vivado can expose
 * the incremental LUT cost of increasing LSB depth.
 */

(* keep_hierarchy = "yes", use_dsp = "no" *)
module W4A5_lsb_depth6(
    input wire [2:0] w_mag,
    input wire [3:0] a_mag,
    output wire [5:0] lsb
);

wire [5:0] partial_0 = w_mag[0] ? {1'b0, 1'b0, a_mag[3], a_mag[2], a_mag[1], a_mag[0]} : 6'b0;
wire [5:0] partial_1 = w_mag[1] ? {1'b0, a_mag[3], a_mag[2], a_mag[1], a_mag[0], 1'b0} : 6'b0;
wire [5:0] partial_2 = w_mag[2] ? {a_mag[3], a_mag[2], a_mag[1], a_mag[0], 1'b0, 1'b0} : 6'b0;

wire [5:0] sum_0 = partial_0;
wire [5:0] sum_1 = sum_0 + partial_1;
wire [5:0] sum_2 = sum_1 + partial_2;
assign lsb = sum_2;

endmodule
