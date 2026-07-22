# FPGA power provenance

This directory archives two existing Vivado 2023.2 power reports. They define
two different scopes and must not be interchanged.

| Evidence kind | Scope | Total power | Confidence | Use |
|---|---|---:|---|---|
| Full-board routed report | Original 4096-PE PD/Hybrid full design | 44.780 W | Low | Provenance for rounding the analytical FPGA input to 45 W. |
| OOC routed report | 4096-PE kernel/OOC implementation at the selected 210 MHz point | 6.485 W | Medium | OOC implementation evidence only; not a board/system power value. |

Both values are Vivado vector-less estimates, not measurements from a physical
board power sensor. The canonical batch model intentionally uses a fixed 45 W
analytical input for both array sizes. This is the rounded 44.780 W full-design
estimate, not the 6.485 W OOC estimate.

`raw_public_manifest.csv` records raw metadata/SHA-256 and public SHA-256 using
only source-relative paths. `PUBLIC_MANIFEST.sha256` covers the sanitized
copies under `reports/`. Public files contain no private path, user, or node
name.

Verify from the artifact root:

```bash
python scripts/sanitize_paths.py \
  results/figure13/evidence --check
python results/figure13/evidence/verify_power_evidence.py
```
