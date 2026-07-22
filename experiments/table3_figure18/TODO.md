# OOC evidence TODO

- Confirm in the camera-ready text that Table 3's power field is described as
  a routed Vivado estimate rather than a physical board measurement.
- The 90x90 routed report contains 8,160 DSPs while the logical PE array uses
  8,100 DSPs. The CSV preserves both values; document the 60-DSP interface
  overhead if the paper requires a physical-resource rather than PE-count view.
- The UDP W4A4 routed report contains 4,123 DSPs while the controlled Table 3
  array budget is 4,096. The CSV preserves both values for the same reason.
