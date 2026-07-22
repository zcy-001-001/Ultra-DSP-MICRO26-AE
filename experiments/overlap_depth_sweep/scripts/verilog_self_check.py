#!/usr/bin/env python3
"""Run xsim self-checks for generated W4A4 overlap-depth RTL.

The Python arithmetic check in generate_overlap_depth_sweep.py validates the
packing model.  This script validates the generated Verilog itself, including
top-level port mapping, DSP48E2 latency, mode alignment, and the source-style
7-bit-result-plus-sign output encoding.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
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
    top_module: str
    prefill_x: tuple[int, ...]
    prefill_y: tuple[int, ...]
    decode_x: tuple[int, ...]
    decode_y: tuple[int, ...]

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
    def weight_count(self) -> int:
        return max(len(self.prefill_y), len(self.decode_y))

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
        Product(result_index=index, start=start, weight_index=weight_index, activation_index=activation_index)
        for index, (start, weight_index, activation_index) in enumerate(raw, start=1)
    ]


def read_selected(path: Path) -> list[Layout]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    return [
        Layout(
            depth=int(row["depth"]),
            top_module=row["top_module"],
            prefill_x=parse_int_tuple(row["prefill_x_pos"]),
            prefill_y=parse_int_tuple(row["prefill_y_pos"]),
            decode_x=parse_int_tuple(row["decode_x_pos"]),
            decode_y=parse_int_tuple(row["decode_y_pos"]),
        )
        for row in rows
    ]


def signal_list(prefix: str, count: int) -> str:
    return ", ".join(f"{prefix}{idx}" for idx in range(1, count + 1))


def declarations(layout: Layout) -> str:
    result_wires = "\n".join(f"wire [6:0] result{idx};" for idx in range(1, layout.result_count + 1))
    sign_wires = "\n".join(f"wire sign{idx};" for idx in range(1, layout.result_count + 1))
    weight_regs = "\n".join(f"reg [3:0] w{idx};" for idx in range(1, layout.weight_count + 1))
    activation_regs = "\n".join(f"reg [3:0] a{idx};" for idx in range(1, layout.activation_count + 1))
    return "\n".join([weight_regs, activation_regs, result_wires, sign_wires])


def instantiation(layout: Layout) -> str:
    ports = ["    .clk(clk)", "    .mode(mode)"]
    ports += [f"    .w{idx}(w{idx})" for idx in range(1, layout.weight_count + 1)]
    ports += [f"    .a{idx}(a{idx})" for idx in range(1, layout.activation_count + 1)]
    ports += [f"    .result{idx}(result{idx})" for idx in range(1, layout.result_count + 1)]
    ports += [f"    .sign{idx}(sign{idx})" for idx in range(1, layout.result_count + 1)]
    ports.append("    .valid_count(valid_count)")
    return f"{layout.top_module} dut(\n" + ",\n".join(ports) + "\n);"


def bundle_assignments(layout: Layout) -> str:
    lines: list[str] = []
    lines.append(f"wire [{layout.result_count * 7 - 1}:0] result_bundle;")
    lines.append(f"wire [{layout.result_count - 1}:0] sign_bundle;")
    for idx in range(1, layout.result_count + 1):
        bit = idx - 1
        lines.append(f"assign result_bundle[{bit * 7} +: 7] = result{idx};")
        lines.append(f"assign sign_bundle[{bit}] = sign{idx};")
    return "\n".join(lines)


def expected_product_lines(products: list[Product], result_count: int, valid_count: int, mode_value: int) -> list[str]:
    lines = [f"        if (mode == 1'b{mode_value}) begin", f"            next_valid = 5'd{valid_count};"]
    for product in products:
        bit = product.result_index - 1
        lines.append(
            f"            next_result[{bit * 7} +: 7] = encode_product(w{product.weight_index}, a{product.activation_index});"
        )
        lines.append(f"            next_sign[{bit}] = w{product.weight_index}[3] ^ a{product.activation_index}[3];")
    lines.append("        end")
    return lines


def drive_random_lines(layout: Layout) -> str:
    lines = []
    for idx in range(1, layout.weight_count + 1):
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
    lines = []
    for idx in range(1, layout.weight_count + 1):
        lines.append(f"        w{idx} = 4'b0;")
    for idx in range(1, layout.activation_count + 1):
        lines.append(f"        a{idx} = 4'b0;")
    lines.append("        mode = 1'b0;")
    lines.append("        compute_expected();")
    return "\n".join(lines)


def testbench_text(layout: Layout, trials: int, seed: int) -> str:
    result_bits = layout.result_count * 7
    prefill_expected = "\n".join(expected_product_lines(layout.prefill_products, layout.result_count, len(layout.prefill_products), 0))
    decode_expected = "\n".join(expected_product_lines(layout.decode_products, layout.result_count, len(layout.decode_products), 1))
    drive_random = drive_random_lines(layout)
    drive_zero = drive_zero_lines(layout)
    return f"""`timescale 1ns/1ps

module tb_{layout.top_module};
reg clk = 1'b0;
reg mode = 1'b0;
reg [31:0] seed = 32'h{seed:08x};
reg [31:0] rand_word;
integer cycle = 0;
integer i;
integer failures = 0;

{declarations(layout)}
wire [4:0] valid_count;

{instantiation(layout)}

{bundle_assignments(layout)}

reg [{result_bits - 1}:0] next_result;
reg [{layout.result_count - 1}:0] next_sign;
reg [4:0] next_valid;
reg [{result_bits - 1}:0] exp_result_pipe [0:{LATENCY_STAGES - 1}];
reg [{layout.result_count - 1}:0] exp_sign_pipe [0:{LATENCY_STAGES - 1}];
reg [4:0] exp_valid_pipe [0:{LATENCY_STAGES - 1}];

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
        encode_product = {{1'b0, mag}} ^ {{7{{sign}}}};
    end
endfunction

task compute_expected;
    begin
        next_result = {{{result_bits}{{1'b0}}}};
        next_sign = {{{layout.result_count}{{1'b0}}}};
        next_valid = 5'd0;
{prefill_expected}
        else begin
{chr(10).join("    " + line for line in decode_expected.splitlines())}
        end
    end
endtask

task drive_random;
    begin
{drive_random}
    end
endtask

task drive_zero;
    begin
{drive_zero}
    end
endtask

task check_outputs;
    begin
        if (cycle >= {INITIAL_SKIP_CYCLES}) begin
            if (valid_count !== exp_valid_pipe[{LATENCY_STAGES - 1}]) begin
                $display("FAIL depth={layout.depth} cycle=%0d valid got=%0d expected=%0d", cycle, valid_count, exp_valid_pipe[{LATENCY_STAGES - 1}]);
                failures = failures + 1;
            end
            if (sign_bundle !== exp_sign_pipe[{LATENCY_STAGES - 1}]) begin
                $display("FAIL depth={layout.depth} cycle=%0d sign got=%b expected=%b", cycle, sign_bundle, exp_sign_pipe[{LATENCY_STAGES - 1}]);
                failures = failures + 1;
            end
            if (result_bundle !== exp_result_pipe[{LATENCY_STAGES - 1}]) begin
                $display("FAIL depth={layout.depth} cycle=%0d result got=%h expected=%h", cycle, result_bundle, exp_result_pipe[{LATENCY_STAGES - 1}]);
                $display("  exp_result_pipe[0]=%h", exp_result_pipe[0]);
                $display("  exp_result_pipe[1]=%h", exp_result_pipe[1]);
                $display("  exp_result_pipe[2]=%h", exp_result_pipe[2]);
                $display("  dut.dsp_p=%h", dut.dsp_p);
                failures = failures + 1;
            end
            if (failures > 0) $finish;
        end
    end
endtask

always @(posedge clk) begin
    exp_result_pipe[0] <= next_result;
    exp_sign_pipe[0] <= next_sign;
    exp_valid_pipe[0] <= next_valid;
    for (i = 1; i < {LATENCY_STAGES}; i = i + 1) begin
        exp_result_pipe[i] <= exp_result_pipe[i - 1];
        exp_sign_pipe[i] <= exp_sign_pipe[i - 1];
        exp_valid_pipe[i] <= exp_valid_pipe[i - 1];
    end
end

initial begin
    exp_result_pipe[0] = {{{result_bits}{{1'b0}}}};
    exp_result_pipe[1] = {{{result_bits}{{1'b0}}}};
    exp_result_pipe[2] = {{{result_bits}{{1'b0}}}};
    exp_sign_pipe[0] = {{{layout.result_count}{{1'b0}}}};
    exp_sign_pipe[1] = {{{layout.result_count}{{1'b0}}}};
    exp_sign_pipe[2] = {{{layout.result_count}{{1'b0}}}};
    exp_valid_pipe[0] = 5'd0;
    exp_valid_pipe[1] = 5'd0;
    exp_valid_pipe[2] = 5'd0;
    drive_random();
    for (cycle = 0; cycle < {trials + LATENCY_STAGES + 4}; cycle = cycle + 1) begin
        @(negedge clk);
        check_outputs();
        if (cycle < {trials}) drive_random();
        else drive_zero();
    end
    if (failures == 0) begin
        $display("PASS depth={layout.depth} top={layout.top_module} trials={trials}");
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


def run_xsim(layout: Layout, trials: int, seed: int, vivado_bin: Path, out_dir: Path) -> None:
    work_dir = out_dir / f"depth{layout.depth}"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)

    tb_path = work_dir / f"tb_{layout.top_module}.sv"
    tb_path.write_text(testbench_text(layout, trials, seed + layout.depth), encoding="utf-8", newline="\n")
    rtl_path = ROOT / "rtl" / f"{layout.top_module}.v"

    xvlog = vivado_bin / "xvlog.bat"
    xelab = vivado_bin / "xelab.bat"
    xsim = vivado_bin / "xsim.bat"
    glbl = vivado_bin.parent / "data" / "verilog" / "src" / "glbl.v"
    for tool in (xvlog, xelab, xsim):
        if not tool.exists():
            raise FileNotFoundError(f"Missing Vivado simulator tool: {tool}")
    if not glbl.exists():
        raise FileNotFoundError(f"Missing Vivado glbl.v: {glbl}")

    run_command([str(xvlog), "--sv", str(rtl_path), str(tb_path), str(glbl), "-nolog"], work_dir, work_dir / "xvlog.log")
    run_command(
        [
            str(xelab),
            "-debug",
            "typical",
            "-L",
            "unisims_ver",
            f"tb_{layout.top_module}",
            "glbl",
            "-s",
            f"sim_{layout.top_module}",
            "-nolog",
        ],
        work_dir,
        work_dir / "xelab.log",
    )
    run_command([str(xsim), f"sim_{layout.top_module}", "-runall", "-nolog"], work_dir, work_dir / "xsim.log")
    sim_log = (work_dir / "xsim.log").read_text(encoding="utf-8", errors="ignore")
    if "PASS" not in sim_log:
        raise RuntimeError(f"Simulation did not report PASS for {layout.top_module}. See {work_dir / 'xsim.log'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    results = ROOT.parents[1] / "results" / "overlap_depth_sweep"
    parser.add_argument("--selected", default=str(results / "layouts_w4a4_selected.csv"))
    parser.add_argument("--out-dir", default=str(results / "verilog_self_check"))
    parser.add_argument("--vivado-bin", default=str(DEFAULT_VIVADO_BIN))
    parser.add_argument("--trials", type=int, default=256)
    parser.add_argument("--seed", type=lambda value: int(value, 0), default=0x1305)
    args = parser.parse_args()

    layouts = read_selected(Path(args.selected))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for layout in layouts:
        run_xsim(layout, args.trials, args.seed, Path(args.vivado_bin), out_dir)
        print(f"PASS depth={layout.depth} top={layout.top_module} trials={args.trials}")


if __name__ == "__main__":
    main()
