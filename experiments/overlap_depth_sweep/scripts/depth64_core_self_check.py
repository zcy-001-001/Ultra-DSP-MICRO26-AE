#!/usr/bin/env python3
"""Run xsim checks for the transformed 64x64 black-box core RTL.

`verilog_self_check.py` validates the standalone sweep RTL, whose output keeps
the source-style result/sign split.  The 64x64 Vitis projects transform that
core to match the W4A4_PD_64_64 black-box ABI: each result is a signed
two's-complement ap_int<7>.  This script checks that transformed core directly
without elaborating the full 4096-PE wrapper.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REBUTTAL_ROOT = ROOT.parent
DEFAULT_PROJECT_ROOT = REBUTTAL_ROOT / "_remote_depth64_projects"
DEFAULT_VIVADO_BIN = Path(r"D:\Xilinx\Vivado\2024.1\bin")
LATENCY_STAGES = 3
INITIAL_SKIP_CYCLES = 16


@dataclass(frozen=True)
class Product:
    result_index: int
    start: int
    weight_index: int
    activation_index: int


@dataclass(frozen=True)
class Layout:
    depth: int
    prefill_x: tuple[int, ...]
    prefill_y: tuple[int, ...]
    decode_x: tuple[int, ...]
    decode_y: tuple[int, ...]

    @property
    def project(self) -> str:
        return f"W4A4_Depth{self.depth}_64_64"

    @property
    def top(self) -> str:
        return f"Hybrid_INT4_INT4_D{self.depth}"

    @property
    def core(self) -> str:
        return f"{self.top}_core"

    @property
    def prefill_products(self) -> list[Product]:
        return products_for_positions(self.prefill_x, self.prefill_y)

    @property
    def decode_products(self) -> list[Product]:
        return products_for_positions(self.decode_x, self.decode_y)

    @property
    def result_count(self) -> int:
        return max(len(self.prefill_products), len(self.decode_products))

    @property
    def activation_count(self) -> int:
        return max(len(self.prefill_x), len(self.decode_x))


def parse_int_tuple(text: str) -> tuple[int, ...]:
    return tuple(int(item) for item in text.split() if item)


def products_for_positions(x_pos: tuple[int, ...], y_pos: tuple[int, ...]) -> list[Product]:
    raw: list[tuple[int, int, int]] = []
    for activation_index, x in enumerate(x_pos, start=1):
        for weight_index, y in enumerate(y_pos, start=1):
            raw.append((x + y, weight_index, activation_index))
    raw.sort()
    return [
        Product(index, start, weight_index, activation_index)
        for index, (start, weight_index, activation_index) in enumerate(raw, start=1)
    ]


def read_layouts(path: Path) -> list[Layout]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    layouts: list[Layout] = []
    for row in rows:
        depth = int(row["depth"])
        if depth < 4:
            continue
        layouts.append(
            Layout(
                depth=depth,
                prefill_x=parse_int_tuple(row["prefill_x_pos"]),
                prefill_y=parse_int_tuple(row["prefill_y_pos"]),
                decode_x=parse_int_tuple(row["decode_x_pos"]),
                decode_y=parse_int_tuple(row["decode_y_pos"]),
            )
        )
    return layouts


def declarations(layout: Layout) -> str:
    lines = ["reg clk = 1'b0;", "reg mode = 1'b0;"]
    lines += [f"reg [3:0] w{idx};" for idx in range(1, 8)]
    lines += [f"reg [3:0] a{idx};" for idx in range(1, layout.activation_count + 1)]
    lines += [f"wire [6:0] result{idx};" for idx in range(1, layout.result_count + 1)]
    return "\n".join(lines)


def instantiation(layout: Layout) -> str:
    ports = ["    .clk(clk)", "    .mode(mode)"]
    ports += [f"    .w{idx}(w{idx})" for idx in range(1, 8)]
    ports += [f"    .a{idx}(a{idx})" for idx in range(1, layout.activation_count + 1)]
    ports += [f"    .result{idx}(result{idx})" for idx in range(1, layout.result_count + 1)]
    return f"{layout.core} dut(\n" + ",\n".join(ports) + "\n);"


def bundle_assignments(layout: Layout) -> str:
    lines = [f"wire [{layout.result_count * 7 - 1}:0] result_bundle;"]
    for idx in range(1, layout.result_count + 1):
        bit = (idx - 1) * 7
        lines.append(f"assign result_bundle[{bit} +: 7] = result{idx};")
    return "\n".join(lines)


def expected_product_lines(products: list[Product], mode_value: int, result_count: int) -> list[str]:
    lines = [f"        if (mode == 1'b{mode_value}) begin"]
    for product in products:
        bit = (product.result_index - 1) * 7
        lines.append(
            f"            next_result[{bit} +: 7] = encode_product(w{product.weight_index}, a{product.activation_index});"
        )
    for idx in range(len(products) + 1, result_count + 1):
        bit = (idx - 1) * 7
        lines.append(f"            next_result[{bit} +: 7] = 7'b0;")
    lines.append("        end")
    return lines


def drive_random_lines(layout: Layout) -> str:
    lines: list[str] = []
    for idx in range(1, 8):
        lines.append("        rand_word = $urandom(seed);")
        lines.append(f"        w{idx} = {{rand_word[0], rand_word[3:1]}};")
    for idx in range(1, layout.activation_count + 1):
        lines.append("        rand_word = $urandom(seed);")
        lines.append(f"        a{idx} = rand_word[3:0];")
    lines.append("        rand_word = $urandom(seed);")
    lines.append("        mode = rand_word[0];")
    lines.append("        compute_expected();")
    return "\n".join(lines)


def drive_zero_lines(layout: Layout) -> str:
    lines = [f"        w{idx} = 4'b0;" for idx in range(1, 8)]
    lines += [f"        a{idx} = 4'b0;" for idx in range(1, layout.activation_count + 1)]
    lines.append("        mode = 1'b0;")
    lines.append("        compute_expected();")
    return "\n".join(lines)


def testbench_text(layout: Layout, trials: int, seed: int) -> str:
    result_bits = layout.result_count * 7
    prefill_expected = "\n".join(
        expected_product_lines(layout.prefill_products, 0, layout.result_count)
    )
    decode_expected = "\n".join(
        expected_product_lines(layout.decode_products, 1, layout.result_count)
    )
    return f"""`timescale 1ns/1ps

module tb_{layout.core};
{declarations(layout)}
reg [31:0] seed = 32'h{seed:08x};
reg [31:0] rand_word;
integer cycle = 0;
integer i;
integer failures = 0;

{instantiation(layout)}

{bundle_assignments(layout)}

reg [{result_bits - 1}:0] next_result;
reg [{result_bits - 1}:0] exp_result_pipe [0:{LATENCY_STAGES - 1}];

always #5 clk = ~clk;

function [2:0] to_mag3;
    input [3:0] x;
    reg [3:0] temp;
    begin
        temp = x[3] ? (~x + 1'b1) : x;
        to_mag3 = (temp == 4'b1000) ? 3'b111 : temp[2:0];
    end
endfunction

function [6:0] encode_product;
    input [3:0] w;
    input [3:0] a;
    reg [5:0] mag;
    reg sign;
    begin
        mag = w[2:0] * to_mag3(a);
        sign = w[3] ^ a[3];
        encode_product = sign ? -$signed({{1'b0, mag}}) : {{1'b0, mag}};
    end
endfunction

task compute_expected;
    begin
        next_result = {{{result_bits}{{1'b0}}}};
{prefill_expected}
        else begin
{chr(10).join("    " + line for line in decode_expected.splitlines())}
        end
    end
endtask

task drive_random;
    begin
{drive_random_lines(layout)}
    end
endtask

task drive_zero;
    begin
{drive_zero_lines(layout)}
    end
endtask

task check_outputs;
    begin
        if (cycle >= {INITIAL_SKIP_CYCLES}) begin
            if (result_bundle !== exp_result_pipe[{LATENCY_STAGES - 1}]) begin
                $display("FAIL depth={layout.depth} cycle=%0d result got=%h expected=%h", cycle, result_bundle, exp_result_pipe[{LATENCY_STAGES - 1}]);
                $display("  dut.dsp_p=%h", dut.dsp_p);
                failures = failures + 1;
            end
            if (failures > 0) $finish;
        end
    end
endtask

always @(posedge clk) begin
    exp_result_pipe[0] <= next_result;
    for (i = 1; i < {LATENCY_STAGES}; i = i + 1) begin
        exp_result_pipe[i] <= exp_result_pipe[i - 1];
    end
end

initial begin
    exp_result_pipe[0] = {{{result_bits}{{1'b0}}}};
    exp_result_pipe[1] = {{{result_bits}{{1'b0}}}};
    exp_result_pipe[2] = {{{result_bits}{{1'b0}}}};
    drive_random();
    for (cycle = 0; cycle < {trials + LATENCY_STAGES + 4}; cycle = cycle + 1) begin
        @(negedge clk);
        check_outputs();
        if (cycle < {trials}) drive_random();
        else drive_zero();
    end
    if (failures == 0) begin
        $display("PASS depth={layout.depth} core={layout.core} trials={trials}");
    end
    $finish;
end

endmodule
"""


def run_command(command: list[str], cwd: Path, log_path: Path) -> None:
    result = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path.write_text(result.stdout, encoding="utf-8", newline="\n")
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(command)}\nSee {log_path}")


def run_xsim(layout: Layout, project_root: Path, vivado_bin: Path, out_dir: Path, trials: int, seed: int) -> None:
    work_dir = out_dir / f"depth{layout.depth}"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)

    wrapper = project_root / layout.project / "src" / f"{layout.top}_wrapper.v"
    if not wrapper.exists():
        raise FileNotFoundError(f"Missing transformed wrapper: {wrapper}")

    tb_path = work_dir / f"tb_{layout.core}.sv"
    tb_path.write_text(testbench_text(layout, trials, seed + layout.depth), encoding="utf-8", newline="\n")

    xvlog = vivado_bin / "xvlog.bat"
    xelab = vivado_bin / "xelab.bat"
    xsim = vivado_bin / "xsim.bat"
    glbl = vivado_bin.parent / "data" / "verilog" / "src" / "glbl.v"
    for tool in (xvlog, xelab, xsim):
        if not tool.exists():
            raise FileNotFoundError(f"Missing Vivado simulator tool: {tool}")
    if not glbl.exists():
        raise FileNotFoundError(f"Missing Vivado glbl.v: {glbl}")

    run_command([str(xvlog), "--sv", str(wrapper), str(tb_path), str(glbl), "-nolog"], work_dir, work_dir / "xvlog.log")
    run_command(
        [
            str(xelab),
            "-debug",
            "typical",
            "-L",
            "unisims_ver",
            f"tb_{layout.core}",
            "glbl",
            "-s",
            f"sim_{layout.core}",
            "-nolog",
        ],
        work_dir,
        work_dir / "xelab.log",
    )
    run_command([str(xsim), f"sim_{layout.core}", "-runall", "-nolog"], work_dir, work_dir / "xsim.log")
    sim_log = (work_dir / "xsim.log").read_text(encoding="utf-8", errors="ignore")
    if "PASS" not in sim_log:
        raise RuntimeError(f"Simulation did not report PASS for {layout.core}. See {work_dir / 'xsim.log'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    results = ROOT.parents[1] / "results" / "overlap_depth_sweep"
    parser.add_argument("--selected", default=str(results / "layouts_w4a4_selected.csv"))
    parser.add_argument("--project-root", default=str(DEFAULT_PROJECT_ROOT))
    parser.add_argument("--vivado-bin", default=str(DEFAULT_VIVADO_BIN))
    parser.add_argument("--out-dir", default=str(results / "depth64_core_self_check"))
    parser.add_argument("--trials", type=int, default=256)
    parser.add_argument("--seed", type=lambda value: int(value, 0), default=0xD64)
    args = parser.parse_args()

    layouts = read_layouts(Path(args.selected))
    for layout in layouts:
        run_xsim(layout, Path(args.project_root), Path(args.vivado_bin), Path(args.out_dir), args.trials, args.seed)
        print(f"PASS depth={layout.depth} core={layout.core} trials={args.trials}")


if __name__ == "__main__":
    main()
