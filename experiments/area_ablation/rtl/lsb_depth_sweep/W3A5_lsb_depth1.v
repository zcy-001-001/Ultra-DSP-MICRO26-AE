/*
 * Auto-generated for the Ultra-DSP rebuttal LSB-depth sweep.
 * Top module: W3A5_lsb_depth1
 * Precision: W3A5
 * Magnitude widths: W=2, A=4
 * Depth: 1
 *
 * This isolates one low-product-bit correction generator. It builds only the
 * low DEPTH bits with a truncated partial-product network, so Vivado can expose
 * the incremental LUT cost of increasing LSB depth.
 */

(* keep_hierarchy = "yes", use_dsp = "no" *)
module W3A5_lsb_depth1(
    input wire [1:0] w_mag,
    input wire [3:0] a_mag,
    output wire [0:0] lsb
);

wire [0:0] partial_0 = w_mag[0] ? {a_mag[0]} : 1'b0;

assign lsb = partial_0;

endmodule
