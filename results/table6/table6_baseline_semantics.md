# Table 6 baseline semantics

Table 6 uses two different reference concepts that must not be conflated.

1. `Baseline (BF16)` is the unquantized BF16 model accuracy. It is an
   upper-bound reference row, not the original OSTQuant quantized baseline.
   The reproduction runner makes this explicit by evaluating with activation,
   weight, key, value, and down-projection bit widths all set to 16 and with
   rotation disabled.
2. The paper text's `original OSTQuant quantized baseline` means OSTQuant at
   the same W4A4 setting used by the packing comparison. Ultra-DSP is bit-exact,
   so its W4A4 row is expected to preserve this quantized OSTQuant accuracy.
   The runner therefore loads the learned W4A4 checkpoint and evaluates the
   Ultra-DSP row with the exact integer packing simulator.

This interpretation is visible in the paper on page 11. Table 6 separately
lists the BF16 and Ultra-DSP rows, while the accompanying paragraph says that
the Ultra-DSP averages match the original OSTQuant accuracy under the same
quantization setting. The same page states that the mixed-precision Ultra-DSP
results match the original OSTQuant quantized baseline because packing is
bit-exact.

Consequently:

- Table 6 BF16 values remain the full-precision reference.
- Table 6 Ultra-DSP values remain the W4A4 OSTQuant/bit-exact-packing result.
- Figure 12's 100% relative-accuracy anchor is the original W4A4 OSTQuant
  quantized accuracy, not the BF16 row.

No numerical row is relabeled or overwritten by this clarification.
