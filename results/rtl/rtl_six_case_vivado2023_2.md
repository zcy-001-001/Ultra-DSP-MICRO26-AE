# Six-case RTL verification with Vivado 2023.2

- Status: `REPRODUCED`
- Tool: Vivado Simulator 2023.2
- Scope: W3A4 and W4A4, each in P, D, and Hybrid mode
- Cases passed: 6/6
- Functional tests passed: 74/74
- Exit status: 0
- Runner: `../../scripts/run_rtl_sim.sh`

| Case | Passed | Failed |
|---|---:|---:|
| W4A4 P | 13 | 0 |
| W4A4 D | 16 | 0 |
| W4A4 Hybrid | 7 | 0 |
| W3A4 P | 16 | 0 |
| W3A4 D | 15 | 0 |
| W3A4 Hybrid | 7 | 0 |

Expected terminal marker:

```text
RTL_SIM_PASS cases=6
```

The run compiles `glbl.v`, links `unisims_ver`, and uses a `1ns/1ps`
timescale. Each case emitted the known DSP48E2 port-A width warning (27-bit
actual versus 30-bit formal); functional verification completed successfully.
Only this sanitized summary is archived. Temporary server paths and user
identifiers are intentionally omitted.
