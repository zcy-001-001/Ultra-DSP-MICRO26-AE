from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
import pandas as pd

plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 12


ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = ROOT.parents[1]
RESULTS = PACKAGE_ROOT / "results" / "figure13"
BATCHES = [1, 4, 16, 64, 256]
POWER_4096_W = 45.0
POWER_8192_W = 45.0


def find_csv(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    names = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"cannot find any candidate CSV: {names}")


def fpga_rows(array_name: str, dsp_count: int, power_w: float) -> list[dict[str, float | int | str]]:
    t_mem_ms = 1000 * 4096 * 4096 * 4 / (460e9 * 8)
    rows = []
    for batch in BATCHES:
        t_comp_ms = 5e-6 * batch * 4096 * 4096 / (dsp_count * 9)
        latency_ms = max(t_mem_ms, t_comp_ms)
        tops = 2 * batch * 4096 * 4096 / (latency_ms * 1e9)
        rows.append(
            {
                "series": array_name,
                "batch_size": batch,
                "latency_ms": latency_ms,
                "tops": tops,
                "power_w": power_w,
                "energy_mj": power_w * latency_ms,
                "tops_per_w": tops / power_w,
                "dominant": "memory" if t_mem_ms >= t_comp_ms else "compute",
            }
        )
    return rows


def build_plot_data() -> pd.DataFrame:
    gpu_csv = find_csv(
        RESULTS / "gpu" / "int4_batch_streaming_gpu_paper_anchored.csv",
        ROOT / "int4_batch_streaming_gpu_paper_anchored.csv",
    )
    cpu_csv = find_csv(
        RESULTS / "cpu" / "int8_batch_streaming_cpu_paper_anchored.csv",
        ROOT / "int8_batch_streaming_cpu_paper_anchored.csv",
    )

    gpu = pd.read_csv(gpu_csv)
    gpu_rows = pd.DataFrame(
        {
            "series": "GPU (RTX 6000 Ada)",
            "batch_size": gpu["batch_size"].astype(int),
            "latency_ms": gpu["latency_ms"].astype(float),
            "tops": gpu["tops"].astype(float),
            "power_w": gpu["power_avg_w"].astype(float),
            "energy_mj": gpu["energy_mj"].astype(float),
            "tops_per_w": gpu["tops_per_w"].astype(float),
            "dominant": "measured",
        }
    )

    cpu = pd.read_csv(cpu_csv)
    cpu_rows = pd.DataFrame(
        {
            "series": "CPU (Xeon 6544Y)",
            "batch_size": cpu["batch_size"].astype(int),
            "latency_ms": cpu["latency_ms"].astype(float),
            "tops": cpu["tops"].astype(float),
            "power_w": cpu["paper_power_w"].astype(float),
            "energy_mj": cpu["paper_energy_mj"].astype(float),
            "tops_per_w": cpu["paper_tops_per_w"].astype(float),
            "dominant": "measured",
        }
    )

    fpga = pd.DataFrame(
        fpga_rows("U55C 4096-DSP Array", 4096, POWER_4096_W)
        + fpga_rows("U55C 8192-DSP Array", 8192, POWER_8192_W)
    )
    return pd.concat([gpu_rows, cpu_rows, fpga], ignore_index=True)


def style_axis(ax, ylabel: str) -> None:
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(BATCHES)
    ax.set_xticklabels([str(b) for b in BATCHES])
    ax.set_xlabel("Batch Size", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.grid(True, which="major", axis="y", color="#d9d9d9", linewidth=0.9)
    ax.grid(True, which="minor", axis="y", color="#eeeeee", linewidth=0.5)
    ax.tick_params(axis="both", labelsize=12)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def main() -> None:
    df = build_plot_data()
    RESULTS.mkdir(parents=True, exist_ok=True)
    out_csv = RESULTS / "gpu_cpu_fpga_batch_throughput_efficiency.csv"
    out_png = RESULTS / "gpu_cpu_fpga_batch_throughput_efficiency.png"
    out_pdf = RESULTS / "gpu_cpu_fpga_batch_throughput_efficiency.pdf"
    df.to_csv(out_csv, index=False, float_format="%.6f")

    styles = {
        "GPU (RTX 6000 Ada)": {"color": "#2b6f8a", "marker": "s", "linestyle": "-"},
        "CPU (Xeon 6544Y)": {"color": "#6b6b6b", "marker": "x", "linestyle": "--"},
        "U55C 4096-DSP Array": {"color": "#8a7ea8", "marker": "D", "linestyle": "-"},
        "U55C 8192-DSP Array": {"color": "#6f9b73", "marker": "o", "linestyle": "-"},
    }

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.3), constrained_layout=False)
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.20, top=0.74, wspace=0.30)
    for series, part in df.groupby("series", sort=False):
        part = part.sort_values("batch_size")
        axes[0].plot(part["batch_size"], part["tops"], label=series, linewidth=2.2, markersize=6.5, **styles[series])
        axes[1].plot(part["batch_size"], part["tops_per_w"], label=series, linewidth=2.2, markersize=6.5, **styles[series])

    style_axis(axes[0], "Throughput (TOPS)")
    style_axis(axes[1], "Energy Eff. (TOPS/W)")
    axes[0].set_ylim(5e-2, 1e3)
    axes[0].yaxis.set_major_locator(LogLocator(base=10, numticks=6))

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=12, bbox_to_anchor=(0.53, 0.985))

    written: list[Path] = [out_csv]
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    written.append(out_png)
    try:
        fig.savefig(out_pdf, bbox_inches="tight")
        written.append(out_pdf)
    except PermissionError:
        fallback_pdf = out_pdf.with_name(f"{out_pdf.stem}_updated{out_pdf.suffix}")
        fig.savefig(fallback_pdf, bbox_inches="tight")
        written.append(fallback_pdf)
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
