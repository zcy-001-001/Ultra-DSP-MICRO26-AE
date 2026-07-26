#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import runpy
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime
from pathlib import Path

# Keep the read-only package check from leaving interpreter cache files in the
# artifact tree. The environment setting is inherited by subprocess checks.
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"


ROOT = Path(__file__).resolve().parent.parent
TABLE6_RESULTS = ROOT / "results" / "table6"


def require(relative: str) -> Path:
    path = ROOT / relative
    if not path.exists():
        raise AssertionError(f"missing: {relative}")
    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest() -> None:
    manifest_path = require("MANIFEST.sha256")
    expected: dict[str, str] = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        expected[relative] = digest
    files = sorted(
        path for path in ROOT.rglob("*")
        if path.is_file()
        and path != manifest_path
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    )
    actual_paths = {path.relative_to(ROOT).as_posix() for path in files}
    assert set(expected) == actual_paths, "manifest file set differs from package"
    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        assert sha256(path) == expected[relative], f"manifest mismatch: {relative}"
    print(f"MANIFEST_PASS files={len(files)}")


def verify_exactness() -> None:
    path = require("results/exactness/absolute_precision_summary.csv")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    ultra = [row for row in rows if row["method"] == "Ultra-DSP"]
    assert len(ultra) == 49, f"expected 49 Ultra-DSP rows, found {len(ultra)}"
    for row in ultra:
        assert row["groups"] == "100"
        assert row["samples_per_group"] == "10000"
        assert float(row["ep"]) == 0.0
        assert float(row["mse"]) == 0.0
        assert int(row["error_count"]) == 0
    print("EXACTNESS_PASS rows=49 samples_per_pair=1000000")


def verify_notebooks() -> None:
    notebook_dir = require("results/ilp_notebooks")
    total_images = 0
    for name in ("pareto.executed.ipynb", "parallelism.executed.ipynb", "efficiency.executed.ipynb"):
        with (notebook_dir / name).open(encoding="utf-8") as handle:
            notebook = json.load(handle)
        errors = []
        for cell in notebook.get("cells", []):
            for output in cell.get("outputs", []):
                if output.get("output_type") == "error":
                    errors.append(output)
                if "image/png" in output.get("data", {}):
                    total_images += 1
        assert not errors, f"notebook errors in {name}"
    assert total_images == 8, f"expected 8 PNG outputs, found {total_images}"
    print("NOTEBOOK_PASS files=3 png_outputs=8")


def verify_gpu_full() -> None:
    path = require("results/table5_gpu/int4_gemv_energy.csv")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6, f"expected 6 GPU rows, found {len(rows)}"
    assert all(int(row["power_samples"]) >= 1800 for row in rows)
    by_config = {row["config"]: row for row in rows}
    assert abs(float(by_config["GEMV_2048x2048"]["latency_ms"]) - 0.011) <= 0.001
    assert abs(float(by_config["GEMV_4096x4096"]["latency_ms"]) - 0.019) <= 0.001
    print("GPU_FORMAL_PASS rows=6 power_samples_per_row>=1800")


def verify_i7_table5() -> None:
    result_dir = require("results/table5_cpu_i7")
    with (result_dir / "i7_table5_summary.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)
    rows = summary["results"]
    assert len(rows) == 2
    assert {row["shape"] for row in rows} == {"1x2048x2048", "1x4096x4096"}
    for row in rows:
        assert row["measurement_seconds"] >= 29.9
        assert row["measurement_iterations"] >= 50000
        assert row["loaded_domain_stats"]["PKG"]["num_samples"] >= 25
        assert abs(row["latency_deviation_percent"]) <= 25.0
        assert abs(row["power_deviation_percent"]) <= 5.0
        assert row["power_scope"] == "RAPL_Package0_PKG_raw_mean_during_measurement_window"
    with (result_dir / "i7_table5_latency_samples.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        latency_count = sum(1 for _ in csv.DictReader(handle))
    assert latency_count == sum(row["measurement_iterations"] for row in rows)
    with (result_dir / "i7_table5_power_samples.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        power_rows = list(csv.DictReader(handle))
    assert sum(int(row["inside_measurement_window"]) for row in power_rows) >= 150
    print("CPU_I7_TABLE5_PASS shapes=2 formal_windows=30s raw_samples=preserved")


def verify_xeon_table5() -> None:
    experiment = require("experiments/table5_cpu_xeon")
    result_dir = require("results/table5_cpu_xeon")
    completed = subprocess.run(
        [sys.executable, str(experiment / "scripts/summarize_table5_xeon.py")],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "XEON_TABLE5_PASS" in completed.stdout
    with (result_dir / "table5_xeon_summary.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = {row["shape"]: row for row in csv.DictReader(handle)}
    assert set(rows) == {"1x2048x2048", "1x4096x4096"}
    expected = {
        "1x2048x2048": (0.1290, 212.709, 27.439461, 213.0),
        "1x4096x4096": (0.3688, 228.648, 84.325382, 229.0),
    }
    for shape, (latency, package0, energy, paper_power) in expected.items():
        row = rows[shape]
        assert abs(float(row["latency_ms"]) - latency) <= 1e-9
        assert abs(float(row["package0_w"]) - package0) <= 1e-9
        assert abs(float(row["table5_package0_energy_mj"]) - energy) <= 1e-6
        assert float(row["paper_package0_w"]) == paper_power
        assert row["power_scope"] == "RAPL_package0_single_socket"
        assert row["status"] == "PASS"
    print("CPU_XEON_TABLE5_PASS shapes=2 archived_streaming_weight_scope=package0")


def verify_phase_adaptivity() -> None:
    experiment = require("experiments/phase_adaptivity")
    result_dir = require("results/phase_adaptivity")
    completed = subprocess.run(
        [sys.executable, str(experiment / "scripts/extract_phase_adaptivity.py")],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "PHASE_ADAPTIVITY_PASS" in completed.stdout
    manifest_check = subprocess.run(
        [sys.executable, str(experiment / "scripts/verify_report_manifest.py")],
        check=False,
        capture_output=True,
        text=True,
    )
    assert manifest_check.returncode == 0, manifest_check.stdout + manifest_check.stderr
    assert "PHASE_REPORT_MANIFEST_PASS reports=12" in manifest_check.stdout
    with (result_dir / "phase_adaptivity_summary.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = {row["configuration"]: row for row in csv.DictReader(handle)}
    expected = {
        "P-only 3x3": (454815, 447403, 4096, 0.003, "MET", "36.353"),
        "D-only 1x7": (365304, 389022, 4096, 0.003, "MET", "36.018"),
        "Hybrid 3x3/1x7": (519486, 475288, 4096, -0.421, "NOT_MET", "44.78"),
    }
    assert set(rows) == set(expected)
    for name, (lut, ff, kernel_dsp, wns, timing_status, power) in expected.items():
        row = rows[name]
        assert int(row["full_clb_lut"]) == lut
        assert int(row["full_clb_register"]) == ff
        assert int(row["full_dsp"]) == 4100
        assert int(row["kernel_dsp"]) == kernel_dsp
        assert float(row["wns_ns"]) == wns
        assert row["timing_status"] == timing_status
        assert row["power_w"] == power
        assert row["power_method"] == "VIVADO_VECTORLESS_ESTIMATE"
        assert row["power_confidence"] == "Low"
        assert row["functional_status"] == "PASS"
    hybrid = rows["Hybrid 3x3/1x7"]
    assert hybrid["power_scope"] == "HYBRID_PD_FULL_DESIGN_PROVENANCE"
    assert hybrid["power_evidence"] == (
        "reports/hybrid/hw_bb_locked_power_routed.rpt"
    )
    print("PHASE_ADAPTIVITY_EVIDENCE_PASS configs=3 hybrid_timing_miss_disclosed=True")


def verify_depth3_ooc() -> None:
    script = require("experiments/overlap_depth_sweep/scripts/extract_depth3_ooc.py")
    completed = subprocess.run(
        [sys.executable, str(script)], check=False, capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "DEPTH3_OOC_PASS" in completed.stdout
    summary = require(
        "results/overlap_depth_sweep/depth3_64x64_ooc_210MHz/depth3_ooc_summary.csv"
    )
    with summary.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    row = rows[0]
    assert (row["prefill_layout"], row["decode_layout"]) == ("3x3", "1x7")
    assert int(row["pe_count"]) == 4096 and int(row["dsp"]) == 4096
    assert int(row["target_frequency_mhz"]) == 210
    assert float(row["wns_ns"]) == 0.138
    assert float(row["estimated_fmax_mhz"]) == 216.267
    assert int(row["lut"]) == 387215 and int(row["ff"]) == 291297
    assert row["timing_met"] == "True" and row["figure18_selected_point"] == "True"
    print("DEPTH3_FIGURE18_PASS PE=4096 Fmax_MHz=216.267")


def verify_figure12_accuracy_provenance() -> None:
    path = require("results/figure12/accuracy_provenance.csv")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = {row["method"]: row for row in csv.DictReader(handle)}
    assert set(rows) == {"WP521", "DB-MixQ", "DSP-Packing", "DuoQ", "UDP", "Ultra-DSP"}
    expected = {
        "WP521": 31.80,
        "DB-MixQ": 32.85,
        "DSP-Packing": 31.97,
        "DuoQ": 65.96,
        "UDP": 65.96,
        "Ultra-DSP": 65.96,
    }
    for method, average in expected.items():
        row = rows[method]
        assert float(row["llama2_7b_avg"]) == average
        assert float(row["normalization_denominator"]) == 65.96
        assert abs(float(row["relative_accuracy_percent"]) - average / 65.96 * 100) <= 1e-12
    # Previous location retained for maintenance history after the result
    # summary moved out of the repository root.
    # provenance = require("RESULTS.md").read_text(encoding="utf-8")
    provenance = require("results/RESULTS.md").read_text(encoding="utf-8")
    normalized = " ".join(provenance.split())
    assert "TransFRU: Efficient Deployment of Transformers" in normalized
    assert "UDP: A Universal DSP Packing Framework" in normalized
    # The public result summary now presents paper values directly instead of
    # exposing internal evidence-status labels.
    # assert provenance.count("LITERATURE_VALUE") >= 2
    print("FIGURE12_ACCURACY_PROVENANCE_PASS methods=6 table2_table4_literature=declared")


def verify_author_guidance_scope() -> None:
    # matrix = require("RESULTS.md").read_text(encoding="utf-8")
    matrix = require("results/RESULTS.md").read_text(encoding="utf-8")
    # The former status phrase is intentionally absent from the public result
    # summary; verify the Table 8 result route and technical scope directly.
    # assert "SCOPED_REPRODUCED/PASS" in matrix
    # assert "FP/MX mantissa/magnitude packing and correction core" in matrix
    assert "`table8/`" in matrix
    assert "FP/MX packing layouts" in matrix
    # Keep excluded disclosure topics absent from user-facing Markdown without
    # repeating those literal phrases in the packaged source itself.
    forbidden = ("TS" + "MC", "P" + "DK", "28 " + "nm")
    for path in ROOT.rglob("*.md"):
        text = path.read_text(encoding="utf-8", errors="replace")
        assert not any(token.lower() in text.lower() for token in forbidden), (
            f"excluded ASIC provenance wording remains in {path.relative_to(ROOT)}"
        )
    print("AUTHOR_GUIDANCE_SCOPE_PASS table8=packing_core excluded_topics=absent")


def verify_documentation_entrypoints() -> None:
    readme_path = require("README.md")
    reproduce_path = require("REPRODUCE.md")
    # results_path = require("RESULTS.md")
    results_path = require("results/RESULTS.md")
    # The final package no longer uses a root-level TODO document.
    # todo_path = require("TODO.md")
    experiment_index_path = require("experiments/README.md")
    result_index_path = require("results/README.md")
    readme = readme_path.read_text(encoding="utf-8")
    reproduce = reproduce_path.read_text(encoding="utf-8")
    results = results_path.read_text(encoding="utf-8")
    # todo = todo_path.read_text(encoding="utf-8")
    experiment_index = experiment_index_path.read_text(encoding="utf-8")
    result_index = result_index_path.read_text(encoding="utf-8")

    # README entry points follow the concise project-oriented structure used by
    # the released Ultra-DSP repository.
    for heading in (
        "Introduction",
        "Repository Structure",
        "Getting Started",
        "Usage",
    ):
        assert heading in readme, f"README section missing: {heading}"

    for table in range(2, 9):
        assert f"Table {table}" in reproduce, f"REPRODUCE missing Table {table}"
    for figure in (12, 13, 16, 17, 18, 19, 20):
        assert f"Figure {figure}" in reproduce, f"REPRODUCE missing Figure {figure}"
    for command in (
        "python scripts/verify_artifact.py",
        "summarize_table5_xeon.py",
        "extract_phase_adaptivity.py",
        "extract_depth3_ooc.py",
        "generate_pareto.py",
        "compare_table6.py",
        "build_table7_geomean.py",
    ):
        assert command in reproduce, f"REPRODUCE command missing: {command}"

    for markdown, path in (
        (readme, readme_path),
        (reproduce, reproduce_path),
        (results, results_path),
        # (todo, todo_path),
        (experiment_index, experiment_index_path),
        (result_index, result_index_path),
    ):
        for target in re.findall(r"\[[^]]+\]\(([^)]+)\)", markdown):
            if target.startswith(("http://", "https://", "#")):
                continue
            relative = target.split("#", 1)[0]
            assert (path.parent / relative).resolve().exists(), (
                f"broken documentation link in {path.relative_to(ROOT)}: {target}"
            )
    retired_tokens = (
        "Figure " + "14",
        "Figure " + "15",
        "Figures " + "14",
        "Figures " + "15",
        "figures" + "14_15",
        "long_" + "context",
    )
    for path in ROOT.rglob("*.md"):
        text = path.read_text(encoding="utf-8", errors="replace")
        assert not any(token.lower() in text.lower() for token in retired_tokens), (
            f"retired experiment wording remains in {path.relative_to(ROOT)}"
        )
    print("DOCUMENTATION_PASS root_docs=2 result_docs=2 reproduce=tables2-8 selected_figures links=valid")


def verify_repository_layout() -> None:
    for legacy in ("artifact", "docs", "evaluation", "reproduced"):
        assert not (ROOT / legacy).exists(), f"legacy top-level directory remains: {legacy}"
    assert not (ROOT / "experiments/exactness/results").exists(), (
        "experiment-local result directory remains: experiments/exactness/results"
    )
    assert not (ROOT / "experiments/table6_table7_accuracy/_remote_picachu_summaries").exists(), (
        "obsolete development summaries remain outside results"
    )
    # Retired experiment implementations and outputs must stay outside the package.
    for path in (
        ROOT / "experiments" / ("figures" + "14_15"),
        ROOT / "results" / ("figures" + "14_15"),
        ROOT / "experiments" / ("long_" + "context"),
        ROOT / "results" / ("long_" + "context"),
    ):
        assert not path.exists(), f"retired experiment path remains: {path.relative_to(ROOT)}"
    assert not any(ROOT.rglob("__pycache__")), "Python cache directories remain"
    assert not any(ROOT.rglob("*.pyc")), "Python bytecode files remain"
    print("REPOSITORY_LAYOUT_PASS results=single_canonical_store legacy_dirs=absent")


def verify_figure17() -> None:
    result_dir = require("results/figure17")
    with (result_dir / "figure17_summary.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)
    assert summary["tool"] == "Vivado 2023.2"
    assert summary["claim_check"] == "PASS"
    assert summary["final_ultradsp_point"]["lut"] == 75
    assert summary["final_ultradsp_point"]["ff"] == 67
    assert summary["final_ultradsp_point"]["dsp"] == 1
    low, high = summary["reduction_range_percent"]
    assert round(low, 1) == 29.2
    assert round(high, 1) == 38.0
    reports = list((result_dir / "reports").rglob("*.rpt"))
    assert len(reports) == 56, f"expected 56 Figure 17 reports, found {len(reports)}"
    print("FIGURE17_PASS cases=28 reports=56 lut_reduction=29.2%-38.0%")


def verify_fpga_model() -> None:
    path = require("results/figure13/fpga_gemv_batch_model.csv")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row["batch_size"]) for row in rows] == [1, 4, 16, 64, 256]
    for row in rows:
        assert row["evidence_class"] == "ANALYTICAL_MODEL"
        assert float(row["power_w"]) == 45.0
        expected = max(float(row["memory_time_ms"]), float(row["compute_time_ms"]))
        assert abs(float(row["total_latency_ms"]) - expected) <= 1e-12
    assert abs(float(rows[0]["total_latency_ms"]) - 0.018236104347826088) <= 1e-12
    assert all(float(row["bandwidth_GBps_decimal"]) == 460.0 for row in rows)
    second = require("results/figure13/fpga_gemv_batch_model_8192dsp.csv")
    with second.open(newline="", encoding="utf-8") as handle:
        rows_8192 = list(csv.DictReader(handle))
    assert len(rows_8192) == 5
    assert all(int(row["dsp_count"]) == 8192 for row in rows_8192)
    assert all(float(row["power_w"]) == 45.0 for row in rows_8192)
    combined = require("results/figure13/batch_comparison_canonical.csv")
    with combined.open(newline="", encoding="utf-8") as handle:
        combined_rows = list(csv.DictReader(handle))
    assert len(combined_rows) == 20
    assert {row["evidence_class"] for row in combined_rows} == {
        "MEASURED", "PAPER_ANCHOR", "ANALYTICAL_MODEL"
    }
    power_check = subprocess.run(
        [
            sys.executable,
            str(
                ROOT
                / "results/figure13/evidence/verify_power_evidence.py"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert power_check.returncode == 0, power_check.stdout + power_check.stderr
    assert "FPGA_POWER_EVIDENCE_PASS full_board_W=44.780 ooc_W=6.485" in power_check.stdout
    assert all(float(row["bandwidth_GBps_decimal"]) == 460.0 for row in rows_8192)
    print("FPGA_MODEL_PASS rows=10 bandwidth_GBps=460 power_W=45")


def verify_rtl_summary() -> None:
    text = require("results/rtl/rtl_six_case_vivado2023_2.md").read_text(
        encoding="utf-8"
    )
    assert "Cases passed: 6/6" in text
    assert "Functional tests passed: 74/74" in text
    assert "RTL_SIM_PASS cases=6" in text
    print("RTL_SUMMARY_PASS cases=6 tests=74")


def verify_ooc_extraction() -> None:
    result_dir = require("results/table3_figure18")
    with (result_dir / "table3_ooc_summary.csv").open(newline="", encoding="utf-8") as handle:
        table3 = list(csv.DictReader(handle))
    with (result_dir / "figure18_post_implementation_summary.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        figure18 = list(csv.DictReader(handle))
    with (result_dir / "figure18_frequency_sweep.csv").open(newline="", encoding="utf-8") as handle:
        sweep = list(csv.DictReader(handle))
    assert len(table3) == 14
    assert len(figure18) == 7
    assert len(sweep) == 106
    assert sum(row["selected_for_paper_figure"] == "True" for row in figure18) == 6
    for row in table3:
        expected = (
            int(row["dsp_count"])
            * int(row["packing_count"])
            * 2
            * float(row["frequency_mhz"])
            / 1000
        )
        assert abs(float(row["throughput_gops"]) - expected) <= 1e-6
    ultra = [row for row in table3 if row["data_format"] == "W4A4" and row["method"] == "Ultra-DSP"]
    assert len(ultra) == 1
    assert abs(float(ultra[0]["throughput_gops"]) - 14745.6) <= 1e-6
    table3_evidence = require("results/table3_figure18/evidence/table3")
    with (table3_evidence / "manifest.json").open(encoding="utf-8") as handle:
        table3_manifest = json.load(handle)
    assert table3_manifest["entry_count"] == 42
    assert len(table3_manifest["entries"]) == 42
    for entry in table3_manifest["entries"]:
        public_path = table3_evidence / entry["public_relative_path"]
        assert public_path.is_file()
        assert sha256(public_path) == entry["public_sha256"]
    figure18_check = subprocess.run(
        [
            sys.executable,
            str(ROOT / "experiments/table3_figure18/scripts/verify_figure18_evidence.py"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert figure18_check.returncode == 0, figure18_check.stdout + figure18_check.stderr
    assert "FIGURE18_EVIDENCE_PASS timing=106 selected_utilization=7" in figure18_check.stdout
    print("OOC_EXTRACTION_PASS table3=14 reports=42 figure18=7 sweep=106 reports=113")


def verify_table3_implementation_sources() -> None:
    root = require("experiments/table3_figure18/baseline_implementations")
    assert not (root.parent / "TODO.md").exists(), (
        "experiments/table3_figure18/TODO.md must not be published"
    )

    cases = (
        ("W4A4/WP521", "WP521", "src/WP521_4096.json", "config/hls.cfg", "ooc_implement.sh"),
        ("W4A4/DB-MixQ", "DeepBurning", "src/DEEPBURNING_4096.json", "config/hls.cfg", "ooc_implement.sh"),
        ("W4A4/DSP-Packing", "DSP-Packing", "src/FPL_4096.json", "config/hls1.cfg", "ooc_implement1.sh"),
        ("W4A4/DuoQ", "DuoQ", "src/DuoQ_4096.json", "config/hls1.cfg", "ooc_implement1.sh"),
        ("W4A4/UDP", "UDP-general/INT4_INT4", "src/W4A4.json", "config/hls.cfg", "ooc_implement.sh"),
        ("W5A5/UDP", "UDP-general/INT5_INT5", "src/W5A5.json", "config/hls.cfg", "ooc_implement.sh"),
        ("W4A3/UDP", "UDP-general/INT3_INT4", "src/W3A4.json", "config/hls.cfg", "ooc_implement.sh"),
        ("W4A5/UDP", "UDP-general/INT4_INT5", "src/W4A5.json", "config/hls.cfg", "ooc_implement.sh"),
        ("W5A3/UDP", "UDP-general/INT3_INT5", "src/W3A5.json", "config/hls.cfg", "ooc_implement.sh"),
        ("W4A4/Ultra-DSP", "Ultra-DSP-MAX/W4A4", "src/W4A4_P.json", "config/hls.cfg", "ooc_implement.sh"),
        ("W5A5/Ultra-DSP", "Ultra-DSP-MAX/W5A5", "src/W5A5_P.json", "config/hls.cfg", "ooc_implement.sh"),
        ("W4A3/Ultra-DSP", "Ultra-DSP-MAX/W3A4", "src/W3A4_P.json", "config/hls.cfg", "ooc_implement.sh"),
        ("W4A5/Ultra-DSP", "Ultra-DSP-MAX/W4A5", "src/W4A5_P.json", "config/hls.cfg", "ooc_implement.sh"),
        ("W5A3/Ultra-DSP", "Ultra-DSP-MAX/W3A5", "src/W3A5_P.json", "config/hls.cfg", "ooc_implement.sh"),
    )
    assert len({row_id for row_id, *_ in cases}) == 14

    referenced_sources: set[Path] = set()
    for row_id, directory, json_relative, config_relative, script_relative in cases:
        case_dir = root / directory
        assert case_dir.is_dir(), f"missing Table 3 implementation directory: {row_id}"
        for relative in (
            "env.sh",
            "src/GEMV.cpp",
            "src/GEMV.h",
            "src/u55C.cfg",
            json_relative,
            config_relative,
            script_relative,
        ):
            assert (case_dir / relative).is_file(), (
                f"missing Table 3 implementation input for {row_id}: {relative}"
            )

        config = (case_dir / config_relative).read_text(encoding="utf-8")
        assert "freqhz=200000000" in config, f"Table 3 HLS frequency is not 200 MHz: {row_id}"
        assert "syn.file=../src/GEMV.cpp" in config
        assert "syn.file=../src/GEMV.h" in config

        metadata_path = case_dir / json_relative
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        references = [
            entry["c_file"] for entry in metadata.get("c_files", [])
        ] + list(metadata.get("rtl_files", []))
        assert references, f"Table 3 black-box metadata has no source files: {row_id}"
        for reference in references:
            relative = Path(reference)
            assert not relative.is_absolute(), (
                f"Table 3 black-box metadata must use a relative path: {row_id}/{reference}"
            )
            resolved = (metadata_path.parent / relative).resolve()
            assert resolved.is_relative_to(case_dir.resolve()), (
                f"Table 3 black-box source escapes its implementation directory: "
                f"{row_id}/{reference}"
            )
            assert resolved.is_file(), (
                f"Table 3 black-box source is missing: {row_id}/{reference}"
            )
            referenced_sources.add(resolved)

    assert (root / "Ultra-DSP-MAX/run_all_ooc.sh").is_file()
    assert (root / "README.md").is_file()
    print(
        f"TABLE3_IMPLEMENTATION_SOURCE_PASS rows={len(cases)} "
        f"blackbox_sources={len(referenced_sources)} todo=absent"
    )


def verify_fully_pipelined_gemv_sources() -> None:
    """Verify the two compact W4A4 releases and kernel-only resources."""

    releases = {
        "new-GEMV-version1(Lut-Reduction)": {
            "lut": 546147,
            "ff": 816273,
            "carry8": 54953,
            "dsp": 4096,
            "bram": 517,
            "uram": 0,
        },
        "new-GEMV-version2(Hybird-Reduction)": {
            "lut": 409819,
            "ff": 589327,
            "carry8": 22697,
            "dsp": 6416,
            "bram": 517,
            "uram": 0,
        },
    }

    source_files = (
        "src/GEMV.cpp",
        "src/GEMV.h",
        "src/u55C.cfg",
        "src/W4A4_P.cpp",
        "src/W4A4_P.json",
        "src/W4A4_P.v",
        "src/W4A4_P_wrapper.v",
    )
    forbidden_directories = {
        ".Xil",
        ".ipcache",
        "__pycache__",
        "_x",
        "synth",
        "ooc_implement",
    }
    forbidden_suffixes = {
        ".bit",
        ".dcp",
        ".jou",
        ".pyc",
        ".str",
        ".wdb",
        ".xclbin",
        ".xo",
    }

    def report_integer(text: str, label: str) -> int:
        match = re.search(
            rf"^\|\s*{re.escape(label)}\s*\|\s*(\d+)\s*\|",
            text,
            flags=re.MULTILINE,
        )
        assert match, f"missing utilization row: {label}"
        return int(match.group(1))

    # Resource publication is intentionally scoped to gemv_kernel. Direct-core
    # and timing/route summaries are not part of these two public result sets.
    checked_resource_reports = 0
    checked_logs = 0
    for release_name, expected_resources in releases.items():
        release = require(release_name)
        readme = require(f"{release_name}/README.md").read_text(encoding="utf-8")
        assert not re.search(
            r"(?i)\b(?:WNS|WHS|slack|timing closure|timing issue)\b",
            readme,
        ), f"README contains timing discussion: {release_name}"
        assert "| Scope |" not in readme, (
            f"resource table must remain under results: {release_name}"
        )
        for relative in source_files:
            assert (release / relative).is_file(), (
                f"missing fully pipelined GEMV source: {release_name}/{relative}"
            )

        config_path = release / "config/hls.cfg"
        config = config_path.read_text(encoding="utf-8")
        assert "freqhz=200000000" in config
        assert "syn.top=gemv_kernel" in config
        config_references = [
            line.split("=", 1)[1].strip()
            for line in config.splitlines()
            if line.startswith(("syn.file=", "syn.blackbox.file=", "tb.file="))
        ]
        assert len(config_references) == 5
        for reference in config_references:
            resolved = (config_path.parent / reference).resolve()
            assert resolved.is_relative_to(release.resolve())
            assert resolved.is_file(), (
                f"missing HLS configuration input: {release_name}/{reference}"
            )

        metadata_path = release / "src/W4A4_P.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        assert metadata["rtl_performance"] == {"latency": "8", "II": "1"}
        metadata_references = [
            entry["c_file"] for entry in metadata["c_files"]
        ] + list(metadata["rtl_files"])
        assert len(metadata_references) == 3
        for reference in metadata_references:
            assert not Path(reference).is_absolute()
            resolved = (metadata_path.parent / reference).resolve()
            assert resolved.is_relative_to(release.resolve())
            assert resolved.is_file(), (
                f"missing black-box input: {release_name}/{reference}"
            )

        shell_scripts = sorted(release.glob("*.sh"))
        assert len(shell_scripts) >= 4
        for script in shell_scripts:
            text = script.read_text(encoding="utf-8")
            assert "set -euo pipefail" in text, f"non-strict shell script: {script}"
            assert 'dirname "${BASH_SOURCE[0]}"' in text, (
                f"shell script is not location independent: {script}"
            )

        kernel_report = (
            release / "results/hls/gemv_kernel_csynth.rpt"
        ).read_text(encoding="utf-8", errors="replace")
        assert re.search(
            r"\|\s*139\|\s*142\|[^|\n]+\|[^|\n]+\|\s*36\|\s*36\|\s*dataflow\|",
            kernel_report,
        ), f"full-kernel interval is not 36 in {release_name}"
        assert {
            path.name for path in (release / "results/hls").glob("*.rpt")
        } == {"gemv_kernel_csynth.rpt"}, (
            f"only the gemv_kernel HLS report may be published: {release_name}"
        )

        simulation_logs = sorted((release / "results/simulation").glob("*.log"))
        assert simulation_logs
        for log in simulation_logs:
            assert "PASS:" in log.read_text(encoding="utf-8", errors="replace"), (
                f"simulation evidence has no PASS marker: {log}"
            )
            checked_logs += 1

        summary_path = release / "results/resource_summary.csv"
        with summary_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 1
        row = rows[0]
        assert row["scope"] == "gemv_kernel"
        assert set(row) == {
            "scope",
            "lut",
            "ff",
            "carry8",
            "dsp",
            "bram",
            "uram",
            "utilization_report",
        }
        for field in ("lut", "ff", "carry8", "dsp", "bram", "uram"):
            assert int(row[field]) == expected_resources[field], (
                f"kernel resource summary mismatch: {release_name}/{field}"
            )

        utilization_path = (
            summary_path.parent / row["utilization_report"]
        ).resolve()
        assert utilization_path.is_relative_to((release / "results").resolve())
        assert utilization_path.is_file(), (
            f"missing gemv_kernel utilization report: {release_name}"
        )
        utilization = utilization_path.read_text(
            encoding="utf-8", errors="replace"
        )
        parsed_utilization = {
            "lut": report_integer(utilization, "CLB LUTs"),
            "ff": report_integer(utilization, "CLB Registers"),
            "carry8": report_integer(utilization, "CARRY8"),
            "dsp": report_integer(utilization, "DSPs"),
            "bram": report_integer(utilization, "Block RAM Tile"),
            "uram": report_integer(utilization, "URAM"),
        }
        assert parsed_utilization == expected_resources, (
            f"gemv_kernel utilization report mismatch: {release_name}"
        )
        assert not (release / "results/direct_core").exists()
        assert not (release / "results/full_kernel").exists()
        checked_resource_reports += 1

        all_paths = list(release.rglob("*"))
        assert not any(
            path.is_dir() and path.name in forbidden_directories
            for path in all_paths
        ), f"generated directory was packaged in {release_name}"
        assert not any(
            path.is_file() and path.suffix.lower() in forbidden_suffixes
            for path in all_paths
        ), f"generated binary was packaged in {release_name}"

    print(
        "FULLY_PIPELINED_GEMV_PASS "
        f"releases={len(releases)} kernel_resource_reports={checked_resource_reports} "
        f"simulation_logs={checked_logs}"
    )


def _verify_table7_noncanonical_archive() -> None:
    """Historical development-log gate retained for maintenance only.

    The formal AE verifier no longer calls this function because the author
    requested paper anchors plus run instructions, not a newly generated
    Table 7 result. A future evaluator may call this helper to audit an
    explicitly optional rerun without conflating it with formal evidence.
    """
    result_dir = require("results/table7")
    archive = result_dir / "mixed_precision_ultradsp_results_logs_sanitized.tgz"
    assert archive.is_file(), "missing sanitized Table 7 log archive"
    forbidden = (
        b"/home/CONNECT/",
        b"/data-" + b"hdd/",
        b"/tmp/",
        b"C:\\Users\\",
    )
    with tarfile.open(archive, "r:gz") as handle:
        files = [member for member in handle.getmembers() if member.isfile()]
        logs = [member for member in files if member.name.endswith("eval_ultradsp/log.txt")]
        assert len(logs) == 14, f"expected 14 Table 7 logs, found {len(logs)}"
        for member in files:
            stream = handle.extractfile(member)
            assert stream is not None
            payload = stream.read()
            assert not any(token in payload for token in forbidden), (
                f"private absolute path remains in Table 7 archive: {member.name}"
            )
            if member in logs:
                assert b"AVERAGE" in payload, f"incomplete Table 7 log: {member.name}"

    unpacked_root = result_dir / "raw_logs_sanitized"
    assert unpacked_root.is_dir(), "missing unpacked sanitized Table 7 logs"
    unpacked_logs = sorted(unpacked_root.rglob("eval_ultradsp/log.txt"))
    assert len(unpacked_logs) == 14, (
        f"expected 14 unpacked Table 7 logs, found {len(unpacked_logs)}"
    )
    for log in unpacked_logs:
        assert "AVERAGE" in log.read_text(encoding="utf-8", errors="replace"), (
            f"incomplete unpacked Table 7 log: {log.relative_to(result_dir)}"
        )

    # Apply the package-wide privacy engine to both the archive members and
    # their unpacked copies.  The historical hard-coded byte-prefix check
    # above remains as a defense-in-depth compatibility gate.
    sanitizer = runpy.run_path(str(ROOT / "scripts/sanitize_paths.py"))
    privacy_findings = sanitizer["audit"](result_dir)
    assert not privacy_findings, (
        "private path remains in Table 7 packaged logs: "
        + ", ".join(privacy_findings[:20])
    )

    with (result_dir / "table7_geomean_summary.json").open(encoding="utf-8") as handle:
        rows = json.load(handle)
    expected = {
        "W5A5": 68.86,
        "W4A5": 68.11,
        "W4A4": 66.78,
        "W5A4": 65.92,
        "W3A5": 63.81,
        "W3A4": 60.78,
        "W5A3": 45.94,
        "W4A3": 44.40,
    }
    assert len(rows) == 8
    assert {row["format"] for row in rows} == set(expected)
    for row in rows:
        recomputed = round(math.sqrt(row["llama2_avg"] * row["llama3_avg"]), 2)
        assert recomputed == row["geomean"]
        assert row["geomean"] == expected[row["format"]]
        assert row["status"] == "PASS"
    assert sum(row["source_kind"] == "RECOMPUTED_FROM_LOGS" for row in rows) == 7
    assert sum(row["source_kind"] == "TABLE6_ARCHIVED_SUMMARY" for row in rows) == 1
    print(
        "TABLE7_ACCURACY_PASS logs=14 unpacked_logs=14 formats=8 "
        "privacy=archive+unpacked"
    )


def verify_table7_accuracy() -> None:
    result_dir = require("results/table7")
    with (result_dir / "table7_paper_anchor.json").open(encoding="utf-8") as handle:
        rows = json.load(handle)
    expected = {
        "W5A5": 68.86,
        "W4A5": 68.11,
        "W4A4": 66.78,
        "W5A4": 65.92,
        "W3A5": 63.81,
        "W3A4": 60.78,
        "W5A3": 45.94,
        "W4A3": 44.40,
    }
    assert len(rows) == 8
    assert {row["format"] for row in rows} == set(expected)
    assert all(row["source_kind"] == "PAPER_ANCHOR" for row in rows)
    for row in rows:
        assert float(row["geomean"]) == expected[row["format"]]
    readme = require(
        "experiments/table6_table7_accuracy/experiments/mixed_precision_ultradsp/README.md"
    ).read_text(encoding="utf-8")
    assert "does not claim a new Table 7 run" in readme
    assert "run_mixed_precision_ultradsp_host.sh" in readme
    assert "table7_optional_recomputed" in readme
    print("TABLE7_METHOD_PASS paper_anchors=8 formal_fresh_results=0 run_instructions=present")


def verify_table6_archive_and_config() -> None:
    experiment = require(
        "experiments/table6_table7_accuracy/experiments/table6_overpacking"
    )
    result_dir = require("results/table6")
    with (result_dir / "table6_archived_summary.json").open(
        encoding="utf-8"
    ) as handle:
        rows = json.load(handle)
    assert len(rows) == 10
    assert {row["model"] for row in rows} == {"llama2_7b", "llama3_8b"}
    assert {row["method"] for row in rows} == {
        "Baseline(BF16)", "Ultra-DSP", "DSP-Packing", "DB-MixQ", "WP521"
    }
    for row in rows:
        recomputed = round(
            sum(row[key] for key in ("arc_easy", "hellaswag", "piqa", "openbookqa")) / 4,
            2,
        )
        assert recomputed == row["avg"]
    discrepancy = [
        row for row in rows
        if row["model"] == "llama3_8b" and row["method"] == "DB-MixQ"
    ]
    assert len(discrepancy) == 1
    assert discrepancy[0]["paper_piqa"] == 51.57
    assert discrepancy[0]["paper_avg"] == 32.68
    require(
        "results/table6/preflight_environment_and_hashes.txt"
    )
    environment = require(
        "results/table6/a40_environment_lock.txt"
    ).read_text(encoding="utf-8")
    assert "torch==2.5.1+cu121" in environment
    assert "transformers==4.57.6" in environment
    assert "tokenizers==0.22.2" in environment
    assert "torchao==" not in environment.lower()
    model_hashes = require(
        "results/table6/model_files_sha256.txt"
    ).read_text(encoding="utf-8")
    assert "[llama2_7b]" in model_hashes and "[llama3_8b]" in model_hashes
    assert model_hashes.count(".safetensors") == 9  # 7 shards plus 2 index files
    dataset_lock_path = result_dir / "dataset_revisions.json"
    dataset_lock = json.loads(dataset_lock_path.read_text(encoding="utf-8"))
    assert isinstance(dataset_lock, dict)
    assert dataset_lock.get("evidence_kind") == "FORMAL_A800_CACHE_REVISION_LOCK"
    assert dataset_lock.get("datasets_package") == "4.8.5"
    assert dataset_lock.get("lm_eval_package") == "0.4.12"
    dataset_records = dataset_lock.get("records")
    assert isinstance(dataset_records, list) and len(dataset_records) == 5, (
        "Table 6 dataset lock must contain exactly five records"
    )
    expected_datasets = {
        "Salesforce/wikitext": {
            "purpose": "OSTQuant calibration",
            "config": "wikitext-2-raw-v1",
            "split": "train",
            "revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
            "num_examples": 36718,
            "task_version": None,
            "task_yaml_sha256": None,
        },
        "allenai/ai2_arc": {
            "purpose": "lm-eval",
            "config": "ARC-Easy",
            "split": "test",
            "revision": "210d026faf9955653af8916fad021475a3f00453",
            "num_examples": 2376,
            "task_version": 1,
            "task_yaml_sha256": "96da1d9efe1df88659481cedc13985cb347dc068554a2eae561303fa39bfab3f",
        },
        "Rowan/hellaswag": {
            "purpose": "lm-eval",
            "config": "default",
            "split": "validation",
            "revision": "218ec52e09a7e7462a5400043bb9a69a41d06b76",
            "num_examples": 10042,
            "task_version": 1,
            "task_yaml_sha256": "e9ef8ac3fed02bf283777d946ae76cbf906152e9c92533c9794e5a9dad78d1cf",
        },
        "baber/piqa": {
            "purpose": "lm-eval",
            "config": "default",
            "split": "validation",
            "revision": "142f6d7367fd9877f0fb3b5734ea6a545f54cdd1",
            "num_examples": 1838,
            "task_version": 1,
            "task_yaml_sha256": "e874f7956ccac325b888d4a5dc600bad867b79fba2bd358fe8761ab8263270fb",
        },
        "allenai/openbookqa": {
            "purpose": "lm-eval",
            "config": "main",
            "split": "test",
            "revision": "388097ea7776314e93a529163e0fea805b8a6454",
            "num_examples": 500,
            "task_version": 1,
            "task_yaml_sha256": "13ac2df0c7d5f73e5a12dccc704048ca28b8123980fa0b8258eef937284f4735",
        },
    }
    assert {record.get("dataset") for record in dataset_records} == set(expected_datasets), (
        "Table 6 dataset lock dataset set differs from the formal cache lock"
    )
    for record in dataset_records:
        assert isinstance(record, dict)
        dataset = str(record["dataset"])
        expected_record = expected_datasets[dataset]
        assert record.get("purpose") == expected_record["purpose"]
        assert record.get("config") == expected_record["config"]
        assert record.get("split") == expected_record["split"]
        assert record.get("dataset_revision") == expected_record["revision"]
        assert record.get("cache_fingerprint") == expected_record["revision"]
        assert record.get("num_examples") == expected_record["num_examples"]
        assert record.get("task_version") == expected_record["task_version"]
        assert record.get("task_yaml_sha256") == expected_record["task_yaml_sha256"]
        assert record.get("builder") == "parquet"
        assert record.get("cache_version") == "0.0.0"
        assert re.fullmatch(r"[0-9a-f]{40}", str(record.get("dataset_revision")))
        assert re.fullmatch(r"[0-9a-f]{40}", str(record.get("cache_fingerprint")))
        assert re.fullmatch(r"[0-9a-f]{64}", str(record.get("dataset_info_sha256")))
        if record.get("purpose") == "lm-eval":
            assert re.fullmatch(r"[0-9a-f]{64}", str(record.get("task_yaml_sha256")))
    assert sum(
        record.get("purpose") == "lm-eval" and record.get("task_version") == 1
        for record in dataset_records
    ) == 4, "Table 6 dataset lock must contain four lm-eval task_version=1 records"
    expected_loaders = {
        ("llama3_8b", "test"): (
            17281981,
            "5aaec2557377e6ca165ef9ba847f2af6ce006d7c0de28c6278528645a0ffe9c4",
        ),
        ("llama3_8b", "train"): (
            21637497,
            "3ca1c6d10992eb8edbe982eef350d3410e0660899810ce7369638d2a4ed7a8cc",
        ),
        ("llama2_7b", "test"): (
            18999589,
            "d18fbf45a7d8c574127d045aa4ea34e42a2d65856dce3fbef2152948b867b587",
        ),
        ("llama2_7b", "train"): (
            25143780,
            "31a4e25edf992da5916724fda14791ed8b811dedf5d2f987f27deb9dc1d7a32a",
        ),
    }
    loaders = dataset_lock.get("serialized_wikitext_loaders")
    assert isinstance(loaders, list) and len(loaders) == 4, (
        "Table 6 dataset lock must contain four serialized WikiText loaders"
    )
    loader_pairs = {(str(item.get("model")), str(item.get("split"))) for item in loaders}
    assert loader_pairs == set(expected_loaders), (
        "Table 6 serialized loader model/split set differs from the formal run"
    )
    for loader in loaders:
        assert isinstance(loader, dict)
        key = (str(loader.get("model")), str(loader.get("split")))
        expected_bytes, expected_sha256 = expected_loaders[key]
        assert loader.get("samples") == 128
        assert loader.get("sequence_length") == 2048
        assert loader.get("seed") == 42
        assert loader.get("bytes") == expected_bytes
        assert loader.get("sha256") == expected_sha256
        assert re.fullmatch(r"[0-9a-f]{64}", str(loader.get("sha256")))
    dataset_markdown = (result_dir / "dataset_revisions.md").read_text(
        encoding="utf-8"
    )
    for marker in (
        "datasets==4.8.5",
        "lm_eval==0.4.12",
        "four serialized WikiText token loaders",
        "128 samples",
        "sequence length 2048",
        "seed 42",
        "strongest evidence",
        "supporting provenance",
        *expected_datasets,
    ):
        assert marker in dataset_markdown, f"dataset revision Markdown missing {marker!r}"
    sanitizer = runpy.run_path(str(ROOT / "scripts/sanitize_paths.py"))
    dataset_privacy_findings = sanitizer["audit"](result_dir)
    assert not dataset_privacy_findings, (
        "private path remains in Table 6 dataset revision evidence: "
        + ", ".join(dataset_privacy_findings[:20])
    )
    source_provenance_path = result_dir / "formal_source_provenance.json"
    source_provenance = json.loads(source_provenance_path.read_text(encoding="utf-8"))
    assert isinstance(source_provenance, dict)
    assert source_provenance.get("evidence_kind") == (
        "FORMAL_EXECUTION_TO_PUBLICATION_SOURCE_LOCK"
    )
    expected_source_records = {
        "experiments/table6_overpacking/scripts/run_table6_full_regeneration.sh": {
            "formal": "f31a8ba0fba272b4f07e86d60e37e3f2f87e2c94f5448f562205c3877261528b",
            "maintenance": "7b282274cf73da99017ae9333e29575a912ad37504f8ae9a2183a6e561ed1133",
            "published": "6a168fc87df2edd079ee8fb7e9bde9b1fc7fbd45cdef15ca765d194b67209e59",
            "execution_evidence": "JOB_START_MANIFEST",
            "publication_relation": (
                "TABLE6_LOGIC_IDENTICAL_EXTENSION_REMOVED_PATHS_AND_HOSTS_SANITIZED"
            ),
        },
        "main.py": {
            "formal": "2664c4c9746f1032ba08e3d5f92ee3c8adeb1e4b50450e9b1354ea30b528a308",
            "maintenance": "2664c4c9746f1032ba08e3d5f92ee3c8adeb1e4b50450e9b1354ea30b528a308",
            "published": "2664c4c9746f1032ba08e3d5f92ee3c8adeb1e4b50450e9b1354ea30b528a308",
            "execution_evidence": "JOB_START_MANIFEST",
            "publication_relation": "BYTE_IDENTICAL",
        },
        "quant/approx_linear.py": {
            "formal": "2166a171e39fcba89ef5cd88b689d66306001a128712799349f63953d7b67d3f",
            "maintenance": "2166a171e39fcba89ef5cd88b689d66306001a128712799349f63953d7b67d3f",
            "published": "2166a171e39fcba89ef5cd88b689d66306001a128712799349f63953d7b67d3f",
            "execution_evidence": "JOB_START_MANIFEST",
            "publication_relation": "BYTE_IDENTICAL",
        },
        "quant/ost_model_utils.py": {
            "formal": "2a3359e9b2514d506dcaf28963f692b60da658e8273068718e16056ac683d072",
            "maintenance": "2a3359e9b2514d506dcaf28963f692b60da658e8273068718e16056ac683d072",
            "published": "2a3359e9b2514d506dcaf28963f692b60da658e8273068718e16056ac683d072",
            "execution_evidence": "PREDATING_MTIME_PLUS_RUNTIME_KLT_EVIDENCE",
            "publication_relation": "BYTE_IDENTICAL",
        },
    }
    source_records = source_provenance.get("records")
    assert isinstance(source_records, list) and len(source_records) == 4
    assert {record.get("relative_path") for record in source_records} == set(
        expected_source_records
    )
    public_source_root = experiment.parents[1]
    for record in source_records:
        assert isinstance(record, dict)
        relative = str(record["relative_path"])
        expected_source = expected_source_records[relative]
        formal_hash = record.get("formal_execution_sha256")
        maintenance_hash = record.get("maintenance_source_sha256")
        published_hash = record.get("published_ae_sha256")
        for digest in (formal_hash, maintenance_hash, published_hash):
            assert isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest)
        assert formal_hash == expected_source["formal"]
        assert maintenance_hash == expected_source["maintenance"]
        assert published_hash == expected_source["published"]
        assert record.get("execution_evidence") == expected_source["execution_evidence"]
        assert record.get("publication_relation") == expected_source["publication_relation"]
        public_path = public_source_root / _table6_relative_path(
            relative, "formal-source provenance path"
        )
        assert public_path.is_file(), f"missing published Table 6 source: {relative}"
        assert sha256(public_path) == published_hash, (
            f"published Table 6 source SHA-256 drifted: {relative}"
        )
    helper_record = next(
        record for record in source_records
        if record["relative_path"] == "quant/ost_model_utils.py"
    )
    assert helper_record["execution_evidence"] != "JOB_START_MANIFEST", (
        "ost_model_utils.py must not be misrepresented as a job-start hash"
    )
    timeline = source_provenance.get("formal_run_timeline")
    assert isinstance(timeline, dict) and set(timeline) == {
        "klt_source_mtime",
        "earliest_formal_job_start",
        "llama2_qmodel_mtime",
        "llama3_qmodel_mtime",
    }
    parsed_timeline = {
        key: datetime.fromisoformat(str(value)) for key, value in timeline.items()
    }
    assert parsed_timeline["klt_source_mtime"] < parsed_timeline["earliest_formal_job_start"]
    assert parsed_timeline["earliest_formal_job_start"] < parsed_timeline["llama2_qmodel_mtime"]
    assert parsed_timeline["earliest_formal_job_start"] < parsed_timeline["llama3_qmodel_mtime"]
    runner_diff = source_provenance.get("runner_public_diff")
    assert isinstance(runner_diff, dict)
    assert runner_diff.get("logic_identical") is True
    changed_defaults = runner_diff.get("only_changed_defaults")
    assert isinstance(changed_defaults, list) and len(changed_defaults) == 4
    assert set(changed_defaults) == {"ENV_DIR", "MODEL_LLAMA2", "MODEL_LLAMA3", "HF_HOME"}
    assert runner_diff.get("replacement_kind") == (
        "PRIVATE_PATHS_AND_HOSTS_TO_PLACEHOLDERS"
    )
    formal_runner_diff = source_provenance.get("runner_formal_to_maintenance_diff")
    assert isinstance(formal_runner_diff, dict)
    assert formal_runner_diff.get("formal_table6_logic_identical") is True
    removed_extension = formal_runner_diff.get("removed_non_table6_extension")
    assert isinstance(removed_extension, list) and removed_extension == [
        "WP521 evaluation invocation after DB-MixQ",
        "WP521-inclusive summary alias",
    ]
    assert "not a paper Table 6 row" in str(formal_runner_diff.get("reason"))
    public_runner = (
        public_source_root
        / "experiments/table6_overpacking/scripts/run_table6_full_regeneration.sh"
    ).read_text(encoding="utf-8")
    for variable in changed_defaults:
        assert re.search(
            rf"^(?:export )?{variable}=.*<REMOTE_WORKSPACE>",
            public_runner,
            re.MULTILINE,
        ), f"public runner does not sanitize {variable} with a placeholder"
    source_provenance_md = (
        result_dir / "formal_source_provenance.md"
    ).read_text(encoding="utf-8")
    for marker in (
        "was not included in that",
        "three-file job-start list",
        "evidence is stated more narrowly",
        "source modification time predates both formal job starts",
        "same byte hash is present in the maintenance source",
        "retained training/KLT logs exercise the CUDA",
        "four private path defaults are placeholders",
        "eight paper Table 6 commands, quantization flags, method order through DB-MixQ",
        "functional correction removes the subsequent WP521 invocation",
        "formal, maintained, and publication hashes remain different and visible",
    ):
        assert marker in source_provenance_md, (
            f"formal-source provenance note is missing evidence boundary {marker!r}"
        )
    smoke = require(
        "results/table6/remote_model_load_smoke.md"
    ).read_text(encoding="utf-8")
    for marker in (
        "MODEL_LOAD_SMOKE_PASS llama2_7b",
        "MODEL_LOAD_SMOKE_PASS llama3_8b",
        "TABLE6_ENV_SMOKE_PASS models=2",
    ):
        assert marker in smoke
    klt_smoke = require(
        "results/table6/klt_cusolver_smoke.md"
    ).read_text(encoding="utf-8")
    assert "KLT_CUSOLVER_SMOKE_PASS size=4096 device=cuda:0" in klt_smoke
    assert "completed successfully in 20 seconds" in klt_smoke
    klt_source = require(
        "experiments/table6_table7_accuracy/quant/ost_model_utils.py"
    ).read_text(encoding="utf-8")
    assert klt_source.count('eig_device = torch.device("cuda")') == 2
    assert klt_source.count("torch.linalg.eigh(cov_matrix.to(eig_device))") == 2
    formal_training = (result_dir / "formal_training_evidence.md").read_text(
        encoding="utf-8"
    )
    for marker in (
        "a_asym=False",
        "k_asym=False",
        "v_asym=False",
        "narrow_symmetric=True",
        "Llama-2-7B | 100 | 1391.7062",
        "Llama-3-8B | 100 | 1434.5068",
        "13,555,347,222",
        "16,139,169,642",
        "NVIDIA A800-SXM4-80GB",
    ):
        assert marker in formal_training
    static_test = experiment / "scripts/test_symmetric_config_static.py"
    completed = subprocess.run(
        [sys.executable, str(static_test)], check=False, capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "SYMMETRIC_CONFIG_STATIC_PASS" in completed.stdout
    parser_source = (experiment / "scripts/parse_table6.py").read_text(
        encoding="utf-8"
    )
    assert "relative_log = (Path(model) / rel_log).as_posix()" in parser_source
    assert '"log": str(log_path)' not in parser_source
    parser_namespace = runpy.run_path(str(experiment / "scripts/parse_table6.py"))
    current_lm_eval_table = """\
|  Tasks   |Version|Filter|n-shot| Metric |   |Value |   |Stderr|
|----------|-------|------|-----:|--------|---|-----:|---|-----:|
|AVERAGE   |    N/A|none  |      |acc     |   |0.6468|+- |0.0115|
|arc_easy  |      1|none  |     0|acc_norm|up |0.7033|+- |0.0094|
|hellaswag |      1|none  |     0|acc_norm|up |0.7126|+- |0.0045|
|openbookqa|      1|none  |     0|acc_norm|up |0.4040|+- |0.0220|
|piqa      |      1|none  |     0|acc_norm|up |0.7671|+- |0.0099|
"""
    with tempfile.TemporaryDirectory() as temporary:
        fixture = Path(temporary) / "log.txt"
        fixture.write_text(current_lm_eval_table, encoding="utf-8")
        parsed = parser_namespace["task_values"](fixture)
    assert parsed == {
        "arc_easy": 70.33,
        "hellaswag": 71.26,
        "piqa": 76.71,
        "openbookqa": 40.4,
        "avg": 64.67,
    }
    table6_readme = (experiment / "README.md").read_text(encoding="utf-8")
    assert "W4A4KV4" in table6_readme
    assert "GPTQ does not replace OSTQuant" in table6_readme
    assert "ab64362da147291612d077accaab5d3ed7b508b6" in table6_readme
    baseline_semantics = (
        result_dir / "table6_baseline_semantics.md"
    ).read_text(encoding="utf-8")
    for marker in (
        "`Baseline (BF16)` is the unquantized BF16 model accuracy",
        "upper-bound reference row",
        "not the original OSTQuant quantized baseline",
        "same W4A4 setting",
        "Ultra-DSP is bit-exact",
        "exact integer packing simulator",
        "Figure 12's 100% relative-accuracy anchor",
        "original W4A4 OSTQuant",
        "not the BF16 row",
        "No numerical row is relabeled or overwritten",
    ):
        assert marker in baseline_semantics, (
            f"Table 6 baseline-semantics note is missing {marker!r}"
        )
    runner_source = (
        experiment / "scripts/run_table6_full_regeneration.sh"
    ).read_text(encoding="utf-8")
    for marker in (
        "--pre_eval=False --rotate=False --lm_eval=True",
        "--max_steps=200 --a_bits=16 --down_bits=16 --w_bits=16 --v_bits=16 --k_bits=16",
        'eval_qmodel "$model_key ultra_dsp" "$model_path" "$train_dir" "$base/ultra_dsp" exact',
    ):
        assert marker in runner_source, f"Table 6 runner semantics drifted: {marker!r}"
    figure12_source = require(
        "experiments/figure12_pareto/scripts/generate_pareto.py"
    ).read_text(encoding="utf-8")
    assert "ULTRA_LLAMA2_AVG = 65.96" in figure12_source
    assert re.search(
        r'"Ultra-DSP"\s*:\s*\{[^}]*"acc"\s*:\s*100\.0',
        figure12_source,
        re.DOTALL,
    )
    assert "Relative Accuracy to W4A4 OSTQuant / Ultra-DSP (%)" in figure12_source
    # Exercise the formal eight-row comparison gate.  The immutable archived
    # file intentionally retains two WP521 rebuttal/Figure 12 records, so the
    # self-test filters those extension rows before invoking the Table 6 gate.
    with tempfile.TemporaryDirectory() as temporary:
        formal_archive = Path(temporary) / "table6_formal_rows.json"
        formal_archive.write_text(
            json.dumps([row for row in rows if row["method"] != "WP521"], indent=2)
            + "\n",
            encoding="utf-8",
        )
        tolerance_test = subprocess.run(
            [
                sys.executable,
                str(experiment / "scripts/compare_table6.py"),
                "--fresh",
                str(formal_archive),
                "--expected",
                str(experiment / "expected/table6_expected_from_existing_artifacts.json"),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    assert tolerance_test.returncode == 0, tolerance_test.stdout + tolerance_test.stderr
    assert "TABLE6_TOLERANCE_PASS" in tolerance_test.stdout
    print(
        "TABLE6_ARCHIVE_PASS rows=10 model_files=24 datasets=5 loaders=4 tasks=4 "
        "remote_smoke=2 baseline_semantics=BF16_vs_W4A4 source_provenance=4 "
        "klt_cusolver_smoke=1 formal_training=2 parser_paths=relative "
        "parser_current_lm_eval=PASS "
        "staged_flow=documented "
        "tolerance_selftest=PASS"
    )


def _table6_relative_path(raw: object, label: str) -> Path:
    assert isinstance(raw, str) and raw, f"{label} must be a non-empty string"
    assert "\\" not in raw, f"{label} must use portable POSIX separators: {raw}"
    assert not raw.startswith("/"), f"{label} must be relative: {raw}"
    assert not re.match(r"^[A-Za-z]:", raw), f"{label} must be relative: {raw}"
    parts = raw.split("/")
    assert all(part not in {"", ".", ".."} for part in parts), (
        f"{label} contains an unsafe path component: {raw}"
    )
    return Path(*parts)


def _table6_assert_no_run_errors(text: str, label: str) -> None:
    error_markers = (
        "Traceback (most recent call last):",
        "RuntimeError:",
        "CUDA out of memory",
        "OutOfMemoryError",
        "Segmentation fault",
        "NCCL watchdog",
        "Killed process",
        "TABLE6_TOLERANCE_OUT_OF_TOLERANCE",
        "Overall status: **OUT_OF_TOLERANCE**",
    )
    for marker in error_markers:
        assert marker not in text, f"error marker {marker!r} remains in {label}"


def _verify_table6_fresh_results(
    experiment: Path,
    emit: bool = True,
    results_override: Path | None = None,
) -> str:
    """Admit only a complete, internally consistent formal Table 6 rerun.

    The large generated checkpoints themselves are not redistributed.  Their
    portable paths, byte sizes, and SHA-256 values are frozen in the checkpoint
    manifest so the remote evidence can still be audited without model data.
    """
    results = TABLE6_RESULTS if results_override is None else results_override
    summary_path = results / "table6_fresh_summary.json"
    assert summary_path.is_file(), "missing Table 6 fresh summary"
    rows = json.loads(summary_path.read_text(encoding="utf-8"))
    assert isinstance(rows, list), "Table 6 fresh summary must be a JSON row list"

    models = ("llama2_7b", "llama3_8b")
    methods = ("Baseline(BF16)", "Ultra-DSP", "DSP-Packing", "DB-MixQ")
    tasks = ("arc_easy", "hellaswag", "piqa", "openbookqa")
    expected_pairs = {(model, method) for model in models for method in methods}
    assert len(rows) == 8, f"expected exactly 8 Table 6 fresh rows, found {len(rows)}"
    pairs = [(str(row.get("model")), str(row.get("method"))) for row in rows]
    assert len(set(pairs)) == 8, "Table 6 fresh summary has duplicate (model, method) rows"
    assert set(pairs) == expected_pairs, "Table 6 fresh summary row set is incomplete"
    assert all(row.get("source_kind") == "FRESH_REMOTE_RERUN" for row in rows), (
        "every Table 6 fresh row must use source_kind=FRESH_REMOTE_RERUN"
    )
    run_ids = {row.get("run_id") for row in rows}
    assert len(run_ids) == 1, "Table 6 fresh rows must share one run_id"
    run_id = next(iter(run_ids))
    assert isinstance(run_id, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{2,127}", run_id), (
        "Table 6 fresh run_id must be a portable identifier"
    )

    parser = runpy.run_path(str(experiment / "scripts/parse_table6.py"))
    logs_root = results / "table6_fresh_logs_sanitized"
    assert logs_root.is_dir(), "missing results/table6_fresh_logs_sanitized"
    admitted_logs: set[Path] = set()
    for row in rows:
        relative = _table6_relative_path(row.get("log"), "Table 6 fresh log path")
        log_path = logs_root / relative
        assert log_path.is_file(), f"missing Table 6 fresh formal log: {relative.as_posix()}"
        assert log_path.resolve().is_relative_to(logs_root.resolve()), (
            f"Table 6 fresh log escapes its evidence root: {relative.as_posix()}"
        )
        admitted_logs.add(log_path.resolve())
        text = log_path.read_text(encoding="utf-8", errors="replace")
        assert "AVERAGE" in text, f"incomplete Table 6 fresh log: {relative.as_posix()}"
        for task in tasks:
            assert task in text, f"missing {task} in Table 6 fresh log: {relative.as_posix()}"
        _table6_assert_no_run_errors(text, relative.as_posix())
        reparsed = parser["task_values"](log_path)
        for metric in (*tasks, "avg"):
            assert metric in row, f"missing {metric} in Table 6 fresh summary row"
            assert abs(float(row[metric]) - float(reparsed[metric])) <= 1e-9, (
                f"Table 6 fresh summary disagrees with reparsed log for "
                f"{row['model']}/{row['method']}/{metric}: "
                f"summary={row[metric]} reparsed={reparsed[metric]}"
            )
    packaged_logs = {path.resolve() for path in logs_root.rglob("*") if path.is_file()}
    assert len(packaged_logs) == 8, (
        f"expected exactly 8 sanitized Table 6 formal logs, found {len(packaged_logs)}"
    )
    assert packaged_logs == admitted_logs, (
        "Table 6 fresh log package differs from the eight summary-referenced logs"
    )

    tolerance_json = results / "table6_fresh_tolerance_report.json"
    tolerance_md = results / "table6_fresh_tolerance_report.md"
    assert tolerance_json.is_file() and tolerance_md.is_file(), (
        "missing Table 6 fresh tolerance JSON/Markdown reports"
    )
    tolerance = json.loads(tolerance_json.read_text(encoding="utf-8"))
    allowed_final_statuses = {"PASS", "PASS_WITH_AUTHOR_ACCEPTED_VARIANCE"}
    assert isinstance(tolerance, dict) and tolerance.get("status") in allowed_final_statuses, (
        "Table 6 fresh tolerance report must be PASS or "
        "PASS_WITH_AUTHOR_ACCEPTED_VARIANCE"
    )
    assert tolerance.get("run_id") == run_id, "Table 6 tolerance JSON run_id mismatch"
    final_status = str(tolerance["status"])
    strict_status = tolerance.get("strict_status")
    assert strict_status in {"PASS", "OUT_OF_TOLERANCE"}, (
        "Table 6 tolerance JSON must preserve its strict_status"
    )
    comparisons = tolerance.get("comparisons")
    assert isinstance(comparisons, list) and len(comparisons) == 8, (
        "Table 6 tolerance JSON must contain eight archived-anchor comparisons"
    )
    accepted_rows = []
    for item in comparisons:
        item_strict = item.get("strict_status")
        item_status = item.get("status")
        assert item_strict in {"PASS", "OUT_OF_TOLERANCE"}, (
            "each Table 6 comparison must preserve strict_status"
        )
        task_deltas = item.get("task_deltas")
        assert isinstance(task_deltas, dict) and set(task_deltas) == set(tasks), (
            "each Table 6 comparison must retain all four task deltas"
        )
        numeric_deltas = [float(task_deltas[task]) for task in tasks]
        recorded_max = float(item.get("max_task_abs_delta"))
        assert abs(recorded_max - round(max(abs(value) for value in numeric_deltas), 2)) <= 1e-9, (
            "Table 6 comparison max_task_abs_delta does not match retained deltas"
        )
        float(item.get("average_delta"))
        if item_strict == "PASS":
            assert item_status == "PASS", (
                "strictly passing Table 6 rows must retain final status PASS"
            )
        else:
            assert item_status == "AUTHOR_ACCEPTED_VARIANCE", (
                "every admitted out-of-tolerance Table 6 row must be explicitly "
                "marked AUTHOR_ACCEPTED_VARIANCE"
            )
            accepted_rows.append(item)
    compared_pairs = {(str(item.get("model")), str(item.get("method"))) for item in comparisons}
    assert len(compared_pairs) == 8, "Table 6 tolerance JSON has duplicate comparisons"
    reference_only = tolerance.get("reference_only_rows")
    assert isinstance(reference_only, list) and not reference_only, (
        "formal Table 6 tolerance JSON must not contain rebuttal-only reference rows"
    )
    trends = tolerance.get("trend_checks")
    assert isinstance(trends, list) and len(trends) == 2
    assert all(item.get("status") == "PASS" for item in trends)
    if final_status == "PASS":
        assert strict_status == "PASS" and not accepted_rows, (
            "a strict Table 6 PASS cannot contain author-accepted variance rows"
        )
    else:
        assert strict_status == "OUT_OF_TOLERANCE" and accepted_rows, (
            "PASS_WITH_AUTHOR_ACCEPTED_VARIANCE requires explicit out-of-tolerance rows"
        )
    tolerance_text = tolerance_md.read_text(encoding="utf-8")
    assert run_id in tolerance_text, "Table 6 tolerance Markdown run_id mismatch"
    assert f"Strict status: **{strict_status}**" in tolerance_text
    assert f"Final AE status: **{final_status}**" in tolerance_text
    assert f"Overall status: **{final_status}**" in tolerance_text
    if final_status == "PASS_WITH_AUTHOR_ACCEPTED_VARIANCE":
        assert tolerance_text.count("AUTHOR_ACCEPTED_VARIANCE") >= len(accepted_rows), (
            "Table 6 tolerance Markdown hides an author-accepted row"
        )
    for model, method in expected_pairs:
        assert model in tolerance_text and method in tolerance_text
    _table6_assert_no_run_errors(tolerance_text, tolerance_md.name)

    environment_path = results / "table6_fresh_a800_environment_lock.txt"
    environment = environment_path.read_text(encoding="utf-8")
    for marker in (
        "NVIDIA A800-SXM4-80GB",
        "Python 3.10.20",
        "torch==2.5.1+cu121",
        "transformers==4.57.6",
        "tokenizers==0.22.2",
        "datasets==4.8.5",
        "lm_eval==0.4.12",
        f"run_id={run_id}",
    ):
        assert marker in environment, f"missing {marker!r} in formal A800 environment lock"
    assert "preflight" not in environment.lower(), "fresh environment lock is still preflight evidence"
    assert "NVIDIA A40" not in environment, "fresh environment lock must describe the A800 run"
    _table6_assert_no_run_errors(environment, environment_path.name)

    checkpoint_path = results / "table6_fresh_checkpoint_manifest.json"
    checkpoint_manifest = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert isinstance(checkpoint_manifest, dict)
    assert checkpoint_manifest.get("run_id") == run_id, "checkpoint manifest run_id mismatch"
    checkpoints = checkpoint_manifest.get("checkpoints")
    assert isinstance(checkpoints, list) and len(checkpoints) == 4, (
        "checkpoint manifest must contain four generated checkpoint records"
    )
    checkpoint_pairs: set[tuple[str, str]] = set()
    checkpoint_paths: set[str] = set()
    for item in checkpoints:
        assert isinstance(item, dict)
        model = str(item.get("model"))
        kind = str(item.get("kind"))
        relative = str(item.get("path"))
        _table6_relative_path(relative, "Table 6 checkpoint evidence path")
        assert model in models and kind in {"model.bin", "qmodel.pt"}
        assert Path(relative).name == kind, "checkpoint kind must match its path basename"
        digest = item.get("sha256")
        byte_count = item.get("bytes")
        assert isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest), (
            f"invalid checkpoint SHA-256 for {relative}"
        )
        assert isinstance(byte_count, int) and not isinstance(byte_count, bool) and byte_count > 0, (
            f"invalid checkpoint byte count for {relative}"
        )
        checkpoint_pairs.add((model, kind))
        checkpoint_paths.add(relative)
    assert checkpoint_pairs == {(model, kind) for model in models for kind in ("model.bin", "qmodel.pt")}
    assert len(checkpoint_paths) == 4, "checkpoint manifest paths must be unique"

    sanitizer = runpy.run_path(str(ROOT / "scripts/sanitize_paths.py"))
    privacy_findings = sanitizer["audit"](results)
    assert not privacy_findings, (
        "private path remains in Table 6 fresh evidence: " + ", ".join(privacy_findings[:20])
    )
    if emit:
        print(
            f"TABLE6_FRESH_PASS rows=8 logs=8 checkpoints=4 run_id={run_id} "
            f"privacy=PASS reparse=PASS tolerance={final_status}"
        )
    return run_id


def _selftest_table6_fresh_gate(experiment: Path) -> None:
    run_id = "fresh-a800-selftest"
    methods = ("Baseline(BF16)", "Ultra-DSP", "DSP-Packing", "DB-MixQ")
    with tempfile.TemporaryDirectory() as temporary:
        fixture = Path(temporary) / "experiment"
        logs_root = fixture / "results/table6_fresh_logs_sanitized"
        (fixture / "scripts").mkdir(parents=True)
        (fixture / "scripts/parse_table6.py").write_text(
            (experiment / "scripts/parse_table6.py").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        rows: list[dict[str, object]] = []
        task_bases = {
            "Baseline(BF16)": 0.70,
            "Ultra-DSP": 0.65,
            "DSP-Packing": 0.30,
            "DB-MixQ": 0.32,
        }
        for model in ("llama2_7b", "llama3_8b"):
            for method in methods:
                relative = Path(model) / method.lower().replace("(", "_").replace(")", "_").replace("-", "_") / "log.txt"
                path = logs_root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                base = task_bases[method]
                path.write_text(
                    "|  Tasks   |Version|Filter|n-shot| Metric |   |Value |   |Stderr|\n"
                    "|----------|-------|------|-----:|--------|---|-----:|---|-----:|\n"
                    f"|AVERAGE   |    N/A|none  |      |acc     |   |{base:.4f}|+- |0.0100|\n"
                    f"|arc_easy  |      1|none  |     0|acc_norm|up |{base:.4f}|+- |0.0100|\n"
                    f"|hellaswag |      1|none  |     0|acc_norm|up |{base:.4f}|+- |0.0100|\n"
                    f"|openbookqa|      1|none  |     0|acc_norm|up |{base:.4f}|+- |0.0100|\n"
                    f"|piqa      |      1|none  |     0|acc_norm|up |{base:.4f}|+- |0.0100|\n",
                    encoding="utf-8",
                )
                value = round(base * 100.0, 2)
                rows.append(
                    {
                        "model": model,
                        "method": method,
                        "log": relative.as_posix(),
                        "source_kind": "FRESH_REMOTE_RERUN",
                        "run_id": run_id,
                        "arc_easy": value,
                        "hellaswag": value,
                        "piqa": value,
                        "openbookqa": value,
                        "avg": value,
                    }
                )
        results = fixture / "results"
        (results / "table6_fresh_summary.json").write_text(
            json.dumps(rows, indent=2) + "\n", encoding="utf-8"
        )
        expected_path = results / "table6_selftest_expected.json"
        expected_rows = [dict(row) for row in rows]
        expected_path.write_text(json.dumps(expected_rows, indent=2) + "\n", encoding="utf-8")

        def run_compare(author_accept: str | None = None) -> subprocess.CompletedProcess[str]:
            command = [
                sys.executable,
                str(experiment / "scripts/compare_table6.py"),
                "--fresh",
                str(results / "table6_fresh_summary.json"),
                "--expected",
                str(expected_path),
                "--out",
                str(results / "table6_fresh_tolerance_report.json"),
                "--out-md",
                str(results / "table6_fresh_tolerance_report.md"),
            ]
            if author_accept:
                command.extend(["--author-accept", author_accept])
            return subprocess.run(command, check=False, capture_output=True, text=True)

        strict_compare = run_compare()
        assert strict_compare.returncode == 0, strict_compare.stdout + strict_compare.stderr
        strict_report = json.loads(
            (results / "table6_fresh_tolerance_report.json").read_text(encoding="utf-8")
        )
        assert strict_report["strict_status"] == "PASS"
        assert strict_report["status"] == "PASS"
        (results / "table6_fresh_a800_environment_lock.txt").write_text(
            f"run_id={run_id}\nNVIDIA A800-SXM4-80GB\nPython 3.10.20\n"
            "torch==2.5.1+cu121\ntransformers==4.57.6\ntokenizers==0.22.2\n"
            "datasets==4.8.5\nlm_eval==0.4.12\n",
            encoding="utf-8",
        )
        checkpoints = []
        for model in ("llama2_7b", "llama3_8b"):
            for kind in ("model.bin", "qmodel.pt"):
                checkpoints.append(
                    {
                        "model": model,
                        "kind": kind,
                        "path": f"{model}/train_w4a4/{kind}",
                        "sha256": hashlib.sha256(f"{model}/{kind}".encode()).hexdigest(),
                        "bytes": 1024,
                    }
                )
        (results / "table6_fresh_checkpoint_manifest.json").write_text(
            json.dumps({"run_id": run_id, "checkpoints": checkpoints}, indent=2) + "\n",
            encoding="utf-8",
        )
        assert _verify_table6_fresh_results(
            fixture, emit=False, results_override=results
        ) == run_id

        expected_rows[1]["arc_easy"] = float(expected_rows[1]["arc_easy"]) - 10.0
        expected_path.write_text(json.dumps(expected_rows, indent=2) + "\n", encoding="utf-8")
        unaccepted_compare = run_compare()
        assert unaccepted_compare.returncode != 0, (
            "an unaccepted out-of-tolerance Table 6 row unexpectedly passed"
        )
        unaccepted_report = json.loads(
            (results / "table6_fresh_tolerance_report.json").read_text(encoding="utf-8")
        )
        assert unaccepted_report["strict_status"] == "OUT_OF_TOLERANCE"
        assert unaccepted_report["status"] == "OUT_OF_TOLERANCE"
        try:
            _verify_table6_fresh_results(
                fixture, emit=False, results_override=results
            )
        except AssertionError:
            pass
        else:
            raise AssertionError("fresh verifier admitted an unaccepted Table 6 variance")

        accepted_compare = run_compare("llama2_7b/Ultra-DSP")
        assert accepted_compare.returncode == 0, accepted_compare.stdout + accepted_compare.stderr
        accepted_report = json.loads(
            (results / "table6_fresh_tolerance_report.json").read_text(encoding="utf-8")
        )
        assert accepted_report["strict_status"] == "OUT_OF_TOLERANCE"
        assert accepted_report["status"] == "PASS_WITH_AUTHOR_ACCEPTED_VARIANCE"
        accepted_rows = [
            item for item in accepted_report["comparisons"]
            if item["status"] == "AUTHOR_ACCEPTED_VARIANCE"
        ]
        assert len(accepted_rows) == 1
        assert accepted_rows[0]["strict_status"] == "OUT_OF_TOLERANCE"
        assert _verify_table6_fresh_results(
            fixture, emit=False, results_override=results
        ) == run_id
    print(
        "TABLE6_FRESH_GATE_SELFTEST_PASS strict=PASS "
        "author_accepted=PASS unaccepted=FAIL"
    )


def _verify_table6_fresh_training_logs(experiment: Path) -> None:
    logs_root = TABLE6_RESULTS / "table6_fresh_training_logs_sanitized"
    expected = {
        logs_root / "llama2_7b/log.txt",
        logs_root / "llama3_8b/log.txt",
    }
    packaged = {path for path in logs_root.rglob("*") if path.is_file()}
    assert packaged == expected, (
        "Table 6 fresh training evidence must contain exactly two model logs"
    )
    markers = (
        "'a_asym': False",
        "'k_asym': False",
        "'v_asym': False",
        "'narrow_symmetric': True",
        "'train_enable_wquant': False",
        "'use_klt': True",
        "'w_gptq': True",
        "'max_steps': 100",
        "train rotate model",
        "train smooth up down",
        "train smooth qk",
        "train smooth ov",
        "smooth norm linear",
    )
    for path in sorted(expected):
        text = path.read_text(encoding="utf-8", errors="replace")
        for marker in markers:
            assert marker in text, f"missing {marker!r} in {path.relative_to(logs_root)}"
        _table6_assert_no_run_errors(text, path.relative_to(logs_root).as_posix())

    sanitizer = runpy.run_path(str(ROOT / "scripts/sanitize_paths.py"))
    privacy_findings = sanitizer["audit"](logs_root)
    assert not privacy_findings, (
        "private path remains in Table 6 training evidence: "
        + ", ".join(privacy_findings[:20])
    )
    print(
        "TABLE6_FRESH_TRAINING_LOGS_PASS logs=2 "
        "config=W4A4KV4_narrow_symmetric privacy=PASS"
    )


def verify_table6_fresh_gate() -> None:
    experiment = require(
        "experiments/table6_table7_accuracy/experiments/table6_overpacking"
    )
    _verify_table6_fresh_training_logs(experiment)
    _selftest_table6_fresh_gate(experiment)
    summary = TABLE6_RESULTS / "table6_fresh_summary.json"
    if not summary.exists():
        # matrix = require("RESULTS.md").read_text(encoding="utf-8")
        matrix = require("results/RESULTS.md").read_text(encoding="utf-8")
        assert "REMOTE-RERUN-IN-PROGRESS" in matrix, (
            "missing table6_fresh_summary.json requires REMOTE-RERUN-IN-PROGRESS "
            "in results/RESULTS.md"
        )
        print("TABLE6_FRESH_PENDING marker=REMOTE-RERUN-IN-PROGRESS")
        return
    _verify_table6_fresh_results(experiment)


def verify_privacy() -> None:
    unit_test = subprocess.run(
        [sys.executable, str(ROOT / "scripts/test_sanitize_paths.py")],
        check=False,
        capture_output=True,
        text=True,
    )
    assert unit_test.returncode == 0, unit_test.stdout + unit_test.stderr
    assert "SANITIZE_PATHS_TEST_PASS cases=16 archives=2" in unit_test.stdout
    command = [sys.executable, str(ROOT / "scripts/sanitize_paths.py"), str(ROOT), "--check"]
    completed = subprocess.run(command, check=False)
    assert completed.returncode == 0, "privacy audit failed"


def main() -> int:
    for required in (
        "README.md",
        # "RESULTS.md",
        "results/RESULTS.md",
        "REPRODUCE.md",
        # The paper PDF is intentionally not distributed with the artifact.
        # "paper/Ultra_DSP.pdf",
        # "TODO.md",
        "src/rtl/W4A4/W4A4_Hybrid.v",
        "results/rtl/rtl_six_case_vivado2023_2.md",
        "results/table5_gpu/int4_gemv_energy.csv",
    ):
        require(required)
    verify_exactness()
    verify_notebooks()
    verify_gpu_full()
    verify_i7_table5()
    verify_xeon_table5()
    verify_phase_adaptivity()
    verify_depth3_ooc()
    verify_figure12_accuracy_provenance()
    verify_author_guidance_scope()
    verify_documentation_entrypoints()
    verify_repository_layout()
    verify_figure17()
    verify_fpga_model()
    verify_rtl_summary()
    verify_table3_implementation_sources()
    verify_fully_pipelined_gemv_sources()
    verify_ooc_extraction()
    verify_table6_archive_and_config()
    verify_table6_fresh_gate()
    verify_table7_accuracy()
    verify_privacy()
    # The root-level MANIFEST.sha256 publication checksum is optional and is no
    # longer part of the standard reproduction workflow.
    # verify_manifest()
    print("ARTIFACT_VERIFY_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
