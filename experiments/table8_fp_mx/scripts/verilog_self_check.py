#!/usr/bin/env python3
"""Python and xsim self-checks for generated decoded FP product PEs."""

from __future__ import annotations

import argparse
import csv
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_XILINX_BIN = Path(r"D:\Xilinx\Vivado\2024.1\bin")
SEED = 1305
RANDOM_TRIALS = 4096


@dataclass(frozen=True)
class ProductTerm:
    out_idx: int
    w_idx: int
    a_idx: int
    start: int
    overlap: int


def parse_positions(text: str) -> list[int]:
    return [int(item) for item in str(text).split() if item]


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def product_terms(x_pos: list[int], y_pos: list[int], product_width: int) -> list[ProductTerm]:
    raw = []
    for wi, xp in enumerate(x_pos):
        for ai, yp in enumerate(y_pos):
            raw.append((xp + yp, wi, ai))
    raw.sort()
    terms = []
    prev_start = None
    for out_idx, (start, wi, ai) in enumerate(raw):
        overlap = 0 if prev_start is None else max(0, prev_start + product_width - start)
        terms.append(ProductTerm(out_idx, wi, ai, start, overlap))
        prev_start = start
    return terms


def value_domain(row: dict[str, str]) -> list[int]:
    return list(range(1 << int(row["product_width"])))


def simulate_row(row: dict[str, str], trials: int = RANDOM_TRIALS, seed: int = SEED) -> dict[str, int | str]:
    width = int(row["product_width"])
    prod_width = 2 * width
    x_pos = parse_positions(row["x_pos"])
    y_pos = parse_positions(row["y_pos"])
    terms = product_terms(x_pos, y_pos, prod_width)
    rng = random.Random(seed)
    domain = value_domain(row)

    def check_values(w_vals: list[int], a_vals: list[int]) -> tuple[int, int]:
        packed_a = sum(value << pos for value, pos in zip(w_vals, x_pos))
        packed_b = sum(value << pos for value, pos in zip(a_vals, y_pos))
        raw = packed_a * packed_b
        correction = 0
        for term in terms:
            if term.overlap > 0:
                expected = w_vals[term.w_idx] * a_vals[term.a_idx]
                correction += (expected & ((1 << term.overlap) - 1)) << term.start
        corrected = raw - correction

        errors = 0
        max_err = 0
        for term in terms:
            expected = w_vals[term.w_idx] * a_vals[term.a_idx]
            if term.overlap > 0:
                high = (corrected >> (term.start + term.overlap)) & (
                    (1 << (prod_width - term.overlap)) - 1
                )
                low = expected & ((1 << term.overlap) - 1)
                observed = (high << term.overlap) | low
            else:
                observed = (corrected >> term.start) & ((1 << prod_width) - 1)
            if observed != expected:
                errors += 1
                max_err = max(max_err, abs(observed - expected))
        return errors, max_err

    pair_errors = 0
    pair_cases = 0
    max_abs_err = 0
    for term in terms:
        for w in domain:
            for a in domain:
                pair_cases += 1
                w_vals = [0 for _ in x_pos]
                a_vals = [0 for _ in y_pos]
                w_vals[term.w_idx] = w
                a_vals[term.a_idx] = a
                errors, err = check_values(w_vals, a_vals)
                pair_errors += errors
                max_abs_err = max(max_abs_err, err)

    batch_errors = 0
    for _ in range(trials):
        w_vals = [rng.choice(domain) for _ in x_pos]
        a_vals = [rng.choice(domain) for _ in y_pos]
        errors, err = check_values(w_vals, a_vals)
        batch_errors += errors
        max_abs_err = max(max_abs_err, err)

    return {
        "top_module": row["top_module"],
        "format": row["format"],
        "design": row["design"],
        "pair_cases": pair_cases,
        "pair_errors": pair_errors,
        "random_batches": trials,
        "random_output_checks": trials * len(terms),
        "random_errors": batch_errors,
        "max_abs_err": max_abs_err,
    }


def write_python_correctness(manifest: Path, out: Path, trials: int, seed: int) -> None:
    rows = read_manifest(manifest)
    results = [simulate_row(row, trials=trials, seed=seed) for row in rows]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "top_module",
                "format",
                "design",
                "pair_cases",
                "pair_errors",
                "random_batches",
                "random_output_checks",
                "random_errors",
                "max_abs_err",
            ],
        )
        writer.writeheader()
        writer.writerows(results)
    total_errors = sum(int(row["pair_errors"]) + int(row["random_errors"]) for row in results)
    print(f"Wrote Python correctness to {out}; total_errors={total_errors}")
    if total_errors:
        raise SystemExit(3)


def tb_literal(width: int, value: int) -> str:
    return f"{width}'d{value}"


def generate_tb(row: dict[str, str], tb_dir: Path, random_trials: int, seed: int) -> Path:
    width = int(row["product_width"])
    prod_width = 2 * width
    x_pos = parse_positions(row["x_pos"])
    y_pos = parse_positions(row["y_pos"])
    terms = product_terms(x_pos, y_pos, prod_width)
    top = row["top_module"]
    tb_top = f"tb_{top}"
    domain = value_domain(row)
    max_value = max(domain)

    text = "`timescale 1ns/1ps\n\n"
    text += f"module {tb_top};\n"
    text += "reg clk = 1'b0;\n"
    text += "always #5 clk = ~clk;\n\n"
    for idx in range(len(x_pos)):
        text += f"reg [{width - 1}:0] w{idx};\n"
    for idx in range(len(y_pos)):
        text += f"reg [{width - 1}:0] a{idx};\n"
    for term in terms:
        text += f"wire [{prod_width - 1}:0] mag{term.out_idx};\n"
    text += "wire [7:0] valid_count;\n\n"

    text += f"{top} dut (\n    .clk(clk),\n"
    for idx in range(len(x_pos)):
        text += f"    .w{idx}(w{idx}),\n"
    for idx in range(len(y_pos)):
        text += f"    .a{idx}(a{idx}),\n"
    for term in terms:
        text += f"    .mag{term.out_idx}(mag{term.out_idx}),\n"
    text += "    .valid_count(valid_count)\n);\n\n"

    text += "integer seed;\ninteger trial;\ninteger failures;\ninteger rv;\n\n"
    text += f"function [{width - 1}:0] rand_mag;\n"
    text += "    input dummy;\n"
    text += "    begin\n"
    text += "        rv = $random(seed);\n"
    text += "        if (rv < 0) rv = -rv;\n"
    text += f"        rand_mag = rv & {tb_literal(width, (1 << width) - 1)};\n"
    text += "    end\nendfunction\n\n"

    text += "task set_zero;\nbegin\n"
    for idx in range(len(x_pos)):
        text += f"    w{idx} = {tb_literal(width, 0)};\n"
    for idx in range(len(y_pos)):
        text += f"    a{idx} = {tb_literal(width, 0)};\n"
    text += "end\nendtask\n\n"

    text += "task set_max;\nbegin\n"
    for idx in range(len(x_pos)):
        text += f"    w{idx} = {tb_literal(width, max_value)};\n"
    for idx in range(len(y_pos)):
        text += f"    a{idx} = {tb_literal(width, max_value)};\n"
    text += "end\nendtask\n\n"

    text += "task set_mixed;\nbegin\n"
    for idx in range(len(x_pos)):
        value = domain[idx % len(domain)]
        text += f"    w{idx} = {tb_literal(width, value)};\n"
    for idx in range(len(y_pos)):
        value = domain[(idx * 3 + 1) % len(domain)]
        text += f"    a{idx} = {tb_literal(width, value)};\n"
    text += "end\nendtask\n\n"

    text += "task set_random;\nbegin\n"
    for idx in range(len(x_pos)):
        text += f"    w{idx} = rand_mag(1'b0);\n"
    for idx in range(len(y_pos)):
        text += f"    a{idx} = rand_mag(1'b0);\n"
    text += "end\nendtask\n\n"

    text += "task check_current;\ninput [255:0] case_name;\nbegin\n"
    text += "    repeat (5) @(posedge clk);\n    #1;\n"
    text += f"    if (valid_count !== 8'd{len(terms)}) begin\n"
    text += '        $display("FAIL %0s valid_count got=%0d", case_name, valid_count);\n'
    text += "        failures = failures + 1;\n    end\n"
    for term in terms:
        expected_expr = f"(w{term.w_idx} * a{term.a_idx})"
        text += f"    if (mag{term.out_idx} !== {expected_expr}) begin\n"
        text += (
            f'        $display("FAIL %0s mag{term.out_idx} got=%0d exp=%0d", '
            f"case_name, mag{term.out_idx}, {expected_expr});\n"
        )
        text += "        failures = failures + 1;\n    end\n"
    text += "end\nendtask\n\n"

    text += "initial begin\n"
    text += f"    seed = {seed};\n    failures = 0;\n"
    text += "    set_zero(); check_current(\"zero\");\n"
    text += "    set_max(); check_current(\"max\");\n"
    text += "    set_mixed(); check_current(\"mixed\");\n"
    text += f"    for (trial = 0; trial < {random_trials}; trial = trial + 1) begin\n"
    text += "        set_random();\n        check_current(\"random\");\n    end\n"
    text += "    if (failures == 0) begin\n"
    text += f'        $display("PASS {top} trials={random_trials} products={len(terms)}");\n'
    text += "    end else begin\n"
    text += f'        $display("FAIL {top} failures=%0d", failures);\n'
    text += "    end\n"
    text += "    $finish;\nend\n\nendmodule\n"

    tb_dir.mkdir(parents=True, exist_ok=True)
    tb_path = tb_dir / f"{tb_top}.v"
    tb_path.write_text(text, encoding="utf-8", newline="\n")
    return tb_path


def run_cmd(cmd: list[str], cwd: Path, log_path: Path) -> None:
    proc = subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(proc.stdout, encoding="utf-8", errors="ignore")
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}\nSee {log_path}")


def run_xsim(manifest: Path, tb_dir: Path, sim_dir: Path, log_dir: Path, random_trials: int, seed: int, xilinx_bin: Path) -> None:
    xvlog = xilinx_bin / "xvlog.bat"
    xelab = xilinx_bin / "xelab.bat"
    xsim = xilinx_bin / "xsim.bat"
    for tool in (xvlog, xelab, xsim):
        if not tool.exists():
            raise FileNotFoundError(f"Vivado simulator tool not found: {tool}")

    rows = read_manifest(manifest)
    for row in rows:
        top = row["top_module"]
        tb_top = f"tb_{top}"
        tb_path = generate_tb(row, tb_dir, random_trials, seed)
        rtl_path = ROOT / row["rtl_file"]
        work = sim_dir / top
        work.mkdir(parents=True, exist_ok=True)
        run_cmd([str(xvlog), str(rtl_path), str(tb_path)], work, log_dir / f"{top}_xvlog.log")
        run_cmd([str(xelab), tb_top, "-snapshot", tb_top], work, log_dir / f"{top}_xelab.log")
        run_cmd([str(xsim), tb_top, "-runall"], work, log_dir / f"{top}_xsim.log")
        log_text = (log_dir / f"{top}_xsim.log").read_text(encoding="utf-8", errors="ignore")
        if f"PASS {top}" not in log_text:
            raise RuntimeError(f"xsim did not report PASS for {top}; see {log_dir / f'{top}_xsim.log'}")
        print(f"PASS xsim {top}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="manifest.csv")
    results = ROOT.parents[1] / "results" / "table8"
    parser.add_argument("--python-out", default=str(results / "fp_python_correctness.csv"))
    parser.add_argument("--tb-dir", default="tb")
    parser.add_argument("--sim-dir", default="sim")
    parser.add_argument("--log-dir", default=str(results / "xsim"))
    parser.add_argument("--random-trials", type=int, default=RANDOM_TRIALS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--xilinx-bin", default=str(DEFAULT_XILINX_BIN))
    parser.add_argument("--skip-xsim", action="store_true")
    args = parser.parse_args()

    manifest = Path(args.manifest)
    if not manifest.is_absolute():
        manifest = ROOT / manifest
    write_python_correctness(
        manifest,
        ROOT / args.python_out,
        trials=args.random_trials,
        seed=args.seed,
    )
    if not args.skip_xsim:
        run_xsim(
            manifest,
            ROOT / args.tb_dir,
            ROOT / args.sim_dir,
            ROOT / args.log_dir,
            random_trials=args.random_trials,
            seed=args.seed,
            xilinx_bin=Path(args.xilinx_bin),
        )


if __name__ == "__main__":
    main()
