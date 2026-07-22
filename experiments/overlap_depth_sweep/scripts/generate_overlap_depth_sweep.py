#!/usr/bin/env python3
"""Generate W4A4 overlap-depth sweep layouts, RTL, and arithmetic checks.

The generated PE follows the Ultra-DSP W4A4 RTL convention: weights are
sign-magnitude encoded as sign plus 3-bit magnitude, while activations are
signed two's-complement INT4 values clamped from -8 to -7 before magnitude
packing.
"""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT.parents[1]
RTL_DIR = ROOT / "rtl"
OUT_DIR = PACKAGE_ROOT / "results" / "overlap_depth_sweep"

DEPTH_CAPS = (3, 4, 5, 6)
MAG_BITS = 3
PRODUCT_WIDTH = MAG_BITS + MAG_BITS
B_WIDTH = 18
A_WIDTH = 27
P_WIDTH = 48
X_LIM = B_WIDTH - 1
Y_LIM = A_WIDTH - 1
R_LIM = P_WIDTH - 1
POINTWISE_LIMIT_BY_DEPTH = {
    3: 2,
    4: 3,
    5: 4,
    6: 4,
}
RTL_WEIGHT_PORTS = 8
RTL_ACTIVATION_PORTS = 5
RTL_RESULT_PORTS = 20
RTL_STYLE = "source_compatible_single_dsp"

DECODE_X = (0,)
DECODE_Y = (0, 4, 8, 11, 15, 19, 23)
PREFILL_X_PAPER = (0, 4, 8)
PREFILL_Y_PAPER = (0, 11, 23)


@dataclass(frozen=True)
class Candidate:
    depth: int
    nx: int
    ny: int
    x_extent: int
    y_extent: int
    x_pos: tuple[int, ...]
    y_pos: tuple[int, ...]
    cost: int
    source: str = "solver"

    @property
    def t(self) -> int:
        return self.nx * self.ny

    @property
    def decode_t(self) -> int:
        return self.ny

    @property
    def product_starts(self) -> tuple[int, ...]:
        return tuple(sorted(x + y for x in self.x_pos for y in self.y_pos))

    @property
    def adjacent_overlaps(self) -> tuple[int, ...]:
        starts = self.product_starts
        return tuple(
            max(0, PRODUCT_WIDTH - (right - left))
            for left, right in zip(starts[:-1], starts[1:])
        )

    @property
    def max_adjacent_overlap(self) -> int:
        overlaps = self.adjacent_overlaps
        return max(overlaps) if overlaps else 0

    @property
    def total_overlap(self) -> int:
        return sum(self.adjacent_overlaps)

    @property
    def pointwise_overlap(self) -> int:
        return pointwise_overlap(self.product_starts)


@dataclass(frozen=True)
class Product:
    result_index: int
    start: int
    overlap: int
    weight_index: int
    activation_index: int


@dataclass(frozen=True)
class SelectedHybrid:
    depth: int
    prefill: Candidate
    decode: Candidate
    note: str


def spread_positions(count: int, extent: int) -> tuple[int, ...]:
    """Distribute integer offsets using the paper Eq. 10 remainder rule."""
    if count == 1:
        return (0,)
    base = extent // (count - 1)
    rem = extent % (count - 1)
    return tuple(
        i * base + max(0, (i + 1) - count + rem)
        for i in range(count)
    )


def product_start_multiset(x_pos: tuple[int, ...], y_pos: tuple[int, ...]) -> list[int]:
    return sorted(x + y for x in x_pos for y in y_pos)


def pointwise_overlap(product_starts: tuple[int, ...]) -> int:
    counts = [0] * P_WIDTH
    for start in product_starts:
        if start < 0 or start + PRODUCT_WIDTH > P_WIDTH:
            return -1
        for idx in range(start, start + PRODUCT_WIDTH):
            counts[idx] += 1
    return max(counts, default=0)


def pointwise_limit_for_depth(depth: int) -> int:
    return POINTWISE_LIMIT_BY_DEPTH.get(depth, max(POINTWISE_LIMIT_BY_DEPTH.values()))


def is_legal_positions(x_pos: tuple[int, ...], y_pos: tuple[int, ...], depth: int) -> bool:
    if x_pos[0] < 0 or y_pos[0] < 0:
        return False
    # Keep the migrated solver faithful to the notebook convention:
    # X_lim/Y_lim/R_lim are the packing bounds used by pareto.ipynb, while
    # B_WIDTH/A_WIDTH/P_WIDTH remain the physical DSP port widths used by RTL.
    if x_pos[-1] + MAG_BITS > X_LIM:
        return False
    if y_pos[-1] + MAG_BITS > Y_LIM:
        return False
    if any((right - left) < MAG_BITS for left, right in zip(x_pos[:-1], x_pos[1:])):
        return False
    if any((right - left) < MAG_BITS for left, right in zip(y_pos[:-1], y_pos[1:])):
        return False

    starts = tuple(product_start_multiset(x_pos, y_pos))
    if not starts or starts[-1] + PRODUCT_WIDTH > R_LIM:
        return False

    min_spacing = PRODUCT_WIDTH - depth
    if any((right - left) < min_spacing for left, right in zip(starts[:-1], starts[1:])):
        return False
    return pointwise_overlap(starts) <= pointwise_limit_for_depth(depth)


def is_legal(candidate: Candidate) -> bool:
    return is_legal_positions(candidate.x_pos, candidate.y_pos, candidate.depth)


def iter_feasible_extents(nx: int, ny: int, depth: int):
    # Legacy notebook-style bound-pruned iterator. It is kept for reference, but
    # the main generator now uses iter_uniform_extents_full below because the
    # bound derivation is too tight after switching to the paper Eq. 10
    # remainder placement rule in spread_positions().
    if nx == 1 and ny == 1:
        return

    min_spacing = PRODUCT_WIDTH - depth

    if nx == 1:
        y_low = (ny - 1) * max(MAG_BITS, min_spacing)
        y_high = min(Y_LIM - MAG_BITS, R_LIM - PRODUCT_WIDTH, PRODUCT_WIDTH * (ny - 1))
        if MAG_BITS * ny > A_WIDTH or y_low > y_high:
            return
        for y_extent in range(int(y_high), int(y_low) - 1, -1):
            yield 0, y_extent
        return

    if ny == 1:
        x_low = (nx - 1) * max(MAG_BITS, min_spacing)
        x_high = min(X_LIM - MAG_BITS, R_LIM - PRODUCT_WIDTH, PRODUCT_WIDTH * (nx - 1))
        if MAG_BITS * nx > B_WIDTH or x_low > x_high:
            return
        for x_extent in range(int(x_high), int(x_low) - 1, -1):
            yield x_extent, 0
        return

    x_low = (nx - 1) * max(min_spacing, MAG_BITS)
    x_high = min(X_LIM - MAG_BITS, (nx - 1) * PRODUCT_WIDTH)
    y_base_low = ny - 1
    y_base_high = Y_LIM - MAG_BITS
    if x_low > x_high or y_base_low > y_base_high:
        return

    for x_extent in range(int(x_high), int(x_low) - 1, -1):
        y_low = max(y_base_low, (ny - 1) * (x_extent + PRODUCT_WIDTH - depth))
        y_high = min(
            y_base_high,
            R_LIM - PRODUCT_WIDTH - x_extent,
            (ny - 1) * (x_extent + PRODUCT_WIDTH),
        )
        if y_low > y_high:
            continue
        for y_extent in range(int(y_high), int(y_low) - 1, -1):
            yield x_extent, y_extent


def iter_uniform_extents_full(nx: int, ny: int, depth: int):
    """Enumerate all uniform extents, then let is_legal apply every constraint."""
    if nx == 1 and ny == 1:
        return

    x_high = 0 if nx == 1 else X_LIM - MAG_BITS
    y_high = 0 if ny == 1 else Y_LIM - MAG_BITS
    for x_extent in range(int(x_high), -1, -1):
        for y_extent in range(int(y_high), -1, -1):
            yield x_extent, y_extent


def best_candidate_for_count(nx: int, ny: int, depth: int) -> Candidate | None:
    # Legacy helper kept for reference: the first draft emitted only the
    # minimum-cost legal layout for each (nx, ny). The rebuttal audit needs all
    # legal uniform layouts, so enumerate_candidates below no longer calls it.
    best: Candidate | None = None
    for x_extent, y_extent in iter_feasible_extents(nx, ny, depth) or []:
        x_pos = spread_positions(nx, x_extent)
        y_pos = spread_positions(ny, y_extent)
        cost = (nx * ny - 1) * PRODUCT_WIDTH - (x_extent + y_extent)
        candidate = Candidate(depth, nx, ny, x_extent, y_extent, x_pos, y_pos, cost)
        if not is_legal(candidate):
            continue
        if best is None:
            best = candidate
            continue
        best_key = (best.cost, -best.t, best.nx, best.ny, best.x_pos, best.y_pos)
        cand_key = (candidate.cost, -candidate.t, candidate.nx, candidate.ny, candidate.x_pos, candidate.y_pos)
        if cand_key < best_key:
            best = candidate
    return best


def enumerate_candidates(depth: int) -> list[Candidate]:
    candidates: list[Candidate] = []
    for nx in range(1, B_WIDTH // MAG_BITS + 1):
        for ny in range(1, A_WIDTH // MAG_BITS + 1):
            if nx == 1 and ny == 1:
                continue
            for x_extent, y_extent in iter_uniform_extents_full(nx, ny, depth) or []:
                x_pos = spread_positions(nx, x_extent)
                y_pos = spread_positions(ny, y_extent)
                cost = (nx * ny - 1) * PRODUCT_WIDTH - (x_extent + y_extent)
                candidate = Candidate(depth, nx, ny, x_extent, y_extent, x_pos, y_pos, cost)
                if is_legal(candidate):
                    candidates.append(candidate)
    return candidates


def pareto_mask(points: list[tuple[float, float]]) -> list[bool]:
    keep = [True] * len(points)
    for i, point_i in enumerate(points):
        for j, point_j in enumerate(points):
            if i == j:
                continue
            dominates = point_j[0] <= point_i[0] and point_j[1] <= point_i[1]
            strict = point_j[0] < point_i[0] or point_j[1] < point_i[1]
            if dominates and strict:
                keep[i] = False
                break
    return keep


def make_manual_candidate(depth: int, nx: int, ny: int, x_pos: tuple[int, ...], y_pos: tuple[int, ...], source: str) -> Candidate:
    cost = (nx * ny - 1) * PRODUCT_WIDTH - (x_pos[-1] + y_pos[-1])
    candidate = Candidate(depth, nx, ny, x_pos[-1], y_pos[-1], x_pos, y_pos, cost, source=source)
    if not is_legal(candidate):
        raise ValueError(f"Manual candidate {source} is not legal for depth {depth}: {candidate}")
    return candidate


def select_hybrid(
    depth: int,
    candidates: list[Candidate],
    prefill_pareto: list[bool],
    decode_pareto: list[bool],
) -> SelectedHybrid:
    pareto_candidates = [candidate for candidate, keep in zip(candidates, prefill_pareto) if keep]

    # Keep the D=3 design point paper/source-compatible.  A throughput-first
    # D=3 selection can legally choose 5x2/1x8, but that is not the final
    # Ultra-DSP W4A4 PE used by the paper and makes the resource table look
    # inconsistent with the published/code baseline.
    if depth == 3:
        prefill = make_manual_candidate(
            depth,
            3,
            3,
            PREFILL_X_PAPER,
            PREFILL_Y_PAPER,
            source="paper_prefill_3x3",
        )
    else:
        prefill = min(
            pareto_candidates,
            key=lambda c: (-c.t, c.cost, c.pointwise_overlap, c.max_adjacent_overlap, c.nx, c.ny, c.x_pos, c.y_pos),
        )

    # Decode is fixed to the paper-compatible 1x7 lane pattern for every depth.
    # This keeps the PE comparable to the current ACE/W4A4 design point while
    # the prefill side sweeps the relaxed overlap-depth benefit.
    decode = make_manual_candidate(
        depth,
        1,
        7,
        DECODE_X,
        DECODE_Y,
        source="paper_decode_1x7",
    )

    note = (
        f"{RTL_STYLE}; D3 prefill/decode are paper-compatible, "
        f"D>3 prefill is throughput-first Pareto; pointwise cap={pointwise_limit_for_depth(depth)}"
    )
    return SelectedHybrid(depth, prefill, decode, note)


def products_for_layout(candidate: Candidate) -> list[Product]:
    products: list[tuple[int, int, int]] = []
    for ax_idx, x in enumerate(candidate.x_pos, start=1):
        for wt_idx, y in enumerate(candidate.y_pos, start=1):
            products.append((x + y, wt_idx, ax_idx))
    products.sort()

    out: list[Product] = []
    prev_start: int | None = None
    for result_index, (start, weight_index, activation_index) in enumerate(products, start=1):
        overlap = 0 if prev_start is None else max(0, PRODUCT_WIDTH - (start - prev_start))
        out.append(Product(result_index, start, overlap, weight_index, activation_index))
        prev_start = start
    return out


def sign_mag_to_int(sign: int, mag: int) -> int:
    return -mag if sign else mag


def activation_mag(value: int) -> int:
    value = -7 if value == -8 else value
    return -value if value < 0 else value


def simulate_layout(candidate: Candidate, rng: random.Random, trials: int) -> None:
    products = products_for_layout(candidate)
    max_weight = max(product.weight_index for product in products)
    max_activation = max(product.activation_index for product in products)

    for _ in range(trials):
        weight_signs = [rng.randrange(2) for _ in range(max_weight)]
        weight_mags = [rng.randrange(8) for _ in range(max_weight)]
        activations = [rng.randrange(-7, 8) for _ in range(max_activation)]
        activation_mags = [activation_mag(value) for value in activations]

        exact_mags: list[int] = []
        exact_signed: list[int] = []
        raw = 0
        correction = 0
        for product in products:
            w_mag = weight_mags[product.weight_index - 1]
            a_mag = activation_mags[product.activation_index - 1]
            mag = w_mag * a_mag
            exact_mags.append(mag)
            sign = weight_signs[product.weight_index - 1] ^ (1 if activations[product.activation_index - 1] < 0 else 0)
            exact_signed.append(sign_mag_to_int(sign, mag))
            raw += mag << product.start
            if product.overlap:
                correction += (mag & ((1 << product.overlap) - 1)) << product.start

        packed = raw - correction
        got: list[int] = []
        for product, exact_mag in zip(products, exact_mags):
            if product.overlap >= PRODUCT_WIDTH:
                mag = exact_mag & ((1 << PRODUCT_WIDTH) - 1)
            elif product.overlap == 0:
                mag = (packed >> product.start) & ((1 << PRODUCT_WIDTH) - 1)
            else:
                upper_width = PRODUCT_WIDTH - product.overlap
                upper = (packed >> (product.start + product.overlap)) & ((1 << upper_width) - 1)
                low = exact_mag & ((1 << product.overlap) - 1)
                mag = (upper << product.overlap) | low
            sign = weight_signs[product.weight_index - 1] ^ (1 if activations[product.activation_index - 1] < 0 else 0)
            got.append(sign_mag_to_int(sign, mag))

        if got != exact_signed:
            raise AssertionError(
                f"Arithmetic check failed for depth={candidate.depth}, "
                f"layout={candidate.nx}x{candidate.ny}, starts={candidate.product_starts}"
            )


def self_check(selected: list[SelectedHybrid], trials: int) -> None:
    rng = random.Random(1305)
    for hybrid in selected:
        simulate_layout(hybrid.prefill, rng, trials)
        simulate_layout(hybrid.decode, rng, trials)


def fmt_ints(values: tuple[int, ...]) -> str:
    return " ".join(str(value) for value in values)


def candidate_row(candidate: Candidate, is_prefill_pareto: bool, is_decode_pareto: bool) -> dict[str, object]:
    return {
        "depth": candidate.depth,
        "nx": candidate.nx,
        "ny": candidate.ny,
        "T": candidate.t,
        "decode_T": candidate.decode_t,
        "pointwise_limit": pointwise_limit_for_depth(candidate.depth),
        "x_pos": fmt_ints(candidate.x_pos),
        "y_pos": fmt_ints(candidate.y_pos),
        "product_offsets": fmt_ints(candidate.product_starts),
        "cost": candidate.cost,
        "cost_per_mult": f"{candidate.cost / candidate.t:.6f}",
        "cost_per_decode_mult": f"{candidate.cost / candidate.decode_t:.6f}",
        "max_adjacent_overlap": candidate.max_adjacent_overlap,
        "total_overlap": candidate.total_overlap,
        "pointwise_overlap": pointwise_overlap(candidate.product_starts),
        "is_prefill_pareto": int(is_prefill_pareto),
        "is_decode_pareto": int(is_decode_pareto),
        "source": candidate.source,
    }


def write_layout_csvs(all_rows: list[dict[str, object]], selected: list[SelectedHybrid]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_path = OUT_DIR / "layouts_w4a4_all.csv"
    all_fields = [
        "depth",
        "nx",
        "ny",
        "T",
        "decode_T",
        "pointwise_limit",
        "x_pos",
        "y_pos",
        "product_offsets",
        "cost",
        "cost_per_mult",
        "cost_per_decode_mult",
        "max_adjacent_overlap",
        "total_overlap",
        "pointwise_overlap",
        "is_prefill_pareto",
        "is_decode_pareto",
        "source",
    ]
    with all_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_fields)
        writer.writeheader()
        writer.writerows(all_rows)

    selected_path = OUT_DIR / "layouts_w4a4_selected.csv"
    selected_fields = [
        "depth",
        "top_module",
        "prefill_schema",
        "decode_schema",
        "prefill_T",
        "decode_T",
        "prefill_x_pos",
        "prefill_y_pos",
        "decode_x_pos",
        "decode_y_pos",
        "prefill_product_offsets",
        "decode_product_offsets",
        "pointwise_limit",
        "prefill_actual_max_overlap",
        "decode_actual_max_overlap",
        "prefill_pointwise_overlap",
        "decode_pointwise_overlap",
        "prefill_total_overlap",
        "decode_total_overlap",
        "prefill_cost",
        "decode_cost",
        "prefill_source",
        "decode_source",
        "rtl_style",
        "note",
    ]
    with selected_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=selected_fields)
        writer.writeheader()
        for hybrid in selected:
            writer.writerow(
                {
                    "depth": hybrid.depth,
                    "top_module": top_name(hybrid.depth),
                    "prefill_schema": f"{hybrid.prefill.nx}x{hybrid.prefill.ny}",
                    "decode_schema": f"{hybrid.decode.nx}x{hybrid.decode.ny}",
                    "prefill_T": hybrid.prefill.t,
                    "decode_T": hybrid.decode.decode_t,
                    "prefill_x_pos": fmt_ints(hybrid.prefill.x_pos),
                    "prefill_y_pos": fmt_ints(hybrid.prefill.y_pos),
                    "decode_x_pos": fmt_ints(hybrid.decode.x_pos),
                    "decode_y_pos": fmt_ints(hybrid.decode.y_pos),
                    "prefill_product_offsets": fmt_ints(hybrid.prefill.product_starts),
                    "decode_product_offsets": fmt_ints(hybrid.decode.product_starts),
                    "pointwise_limit": pointwise_limit_for_depth(hybrid.depth),
                    "prefill_actual_max_overlap": hybrid.prefill.max_adjacent_overlap,
                    "decode_actual_max_overlap": hybrid.decode.max_adjacent_overlap,
                    "prefill_pointwise_overlap": hybrid.prefill.pointwise_overlap,
                    "decode_pointwise_overlap": hybrid.decode.pointwise_overlap,
                    "prefill_total_overlap": hybrid.prefill.total_overlap,
                    "decode_total_overlap": hybrid.decode.total_overlap,
                    "prefill_cost": hybrid.prefill.cost,
                    "decode_cost": hybrid.decode.cost,
                    "prefill_source": hybrid.prefill.source,
                    "decode_source": hybrid.decode.source,
                    "rtl_style": RTL_STYLE,
                    "note": hybrid.note,
                }
            )


def top_name(depth: int) -> str:
    return f"w4a4_overlap_depth{depth}_hybrid"


def lsb_module_name(depth: int) -> str:
    return f"w4a4_overlap_depth{depth}_lsb_gen"


def verilog_vector_or(width: int, terms: list[str]) -> str:
    if not terms:
        return f"{width}'b0"
    return " |\n    ".join(terms)


def verilog_vector_sum(width: int, terms: list[str]) -> str:
    if not terms:
        return f"{width}'b0"
    return " +\n    ".join(terms)


def pack_terms(port_width: int, prefix: str, positions: tuple[int, ...]) -> list[str]:
    terms = []
    for idx, pos in enumerate(positions, start=1):
        terms.append(f"({{{{{port_width - MAG_BITS}{{1'b0}}}}, {prefix}{idx}[2:0]}} << {pos})")
    return terms


def pack_activation_terms(port_width: int, positions: tuple[int, ...]) -> list[str]:
    terms = []
    for idx, pos in enumerate(positions, start=1):
        terms.append(f"({{{{{port_width - MAG_BITS}{{1'b0}}}}, a{idx}m}} << {pos})")
    return terms


def lsb_decl_lines(depth: int, phase: str, products: list[Product]) -> list[str]:
    lines: list[str] = []
    for product in products:
        if product.overlap == 0:
            continue
        wire_name = f"{phase}_lsb{product.result_index}"
        lines.append(f"wire [{depth - 1}:0] {wire_name};")
        lines.append(
            f"{lsb_module_name(depth)} #(.DEPTH({depth})) u_{wire_name}("
            f".w_mag(w{product.weight_index}[2:0]), .a_mag(a{product.activation_index}m), .lsb({wire_name}));"
        )
    return lines


def correction_terms(depth: int, phase: str, products: list[Product]) -> list[str]:
    terms: list[str] = []
    for product in products:
        if product.overlap == 0:
            continue
        wire_name = f"{phase}_lsb{product.result_index}"
        terms.append(
            f"({{{{{P_WIDTH - product.overlap}{{1'b0}}}}, "
            f"{wire_name}[{product.overlap - 1}:0]}} << {product.start})"
        )
    return terms


def recover_expr(product: Product, phase: str) -> str:
    if product.overlap == 0:
        return f"p[{product.start + PRODUCT_WIDTH - 1}:{product.start}]"
    wire_name = f"{phase}_lsb{product.result_index}"
    if product.overlap >= PRODUCT_WIDTH:
        return f"{wire_name}[{PRODUCT_WIDTH - 1}:0]"
    upper_hi = product.start + PRODUCT_WIDTH - 1
    upper_lo = product.start + product.overlap
    return f"{{p[{upper_hi}:{upper_lo}], {wire_name}[{product.overlap - 1}:0]}}"


def result_assignment_lines(products: list[Product], phase: str, valid_count: int) -> list[str]:
    lines = [f"        valid_count <= 5'd{valid_count};"]
    for product in products:
        sign_expr = f"w{product.weight_index}[3] ^ a{product.activation_index}[3]"
        lines.append(
            f"        result{product.result_index} <= recover({recover_expr(product, phase)}, {sign_expr});"
        )
    for result_index in range(len(products) + 1, RTL_RESULT_PORTS + 1):
        lines.append(f"        result{result_index} <= 16'sd0;")
    return lines


def port_decl_lines(prefix: str, count: int, signed: bool = False) -> str:
    sign_text = " signed" if signed else ""
    names = ", ".join(f"{prefix}{idx}" for idx in range(1, count + 1))
    return f"    input  wire{sign_text} [3:0] {names},"


def output_result_decl_lines() -> str:
    chunks: list[str] = []
    for start in range(1, RTL_RESULT_PORTS + 1, 5):
        end = min(start + 4, RTL_RESULT_PORTS)
        names = ", ".join(f"result{idx}" for idx in range(start, end + 1))
        chunks.append(f"    output reg  signed [15:0] {names},")
    return "\n".join(chunks)


def output_result_wire_decl_lines(result_count: int) -> str:
    chunks: list[str] = []
    for start in range(1, result_count + 1, 5):
        end = min(start + 4, result_count)
        names = ", ".join(f"result{idx}" for idx in range(start, end + 1))
        chunks.append(f"    output wire [6:0]  {names},")
    return "\n".join(chunks)


def output_sign_decl_lines(result_count: int) -> str:
    chunks: list[str] = []
    for start in range(1, result_count + 1, 5):
        end = min(start + 4, result_count)
        names = ", ".join(f"sign{idx}" for idx in range(start, end + 1))
        chunks.append(f"    output wire        {names},")
    return "\n".join(chunks)


def activation_mag_lines() -> str:
    return "\n".join(
        f"wire [2:0] a{idx}m = amag(a{idx});"
        for idx in range(1, RTL_ACTIVATION_PORTS + 1)
    )


def dynamic_activation_mag_lines(count: int) -> str:
    return "\n".join(
        f"wire [2:0] a{idx}m = to_mag3(a{idx});"
        for idx in range(1, count + 1)
    )


def dynamic_weight_mag_lines(count: int) -> str:
    return "\n".join(
        f"wire [2:0] w{idx}m = w{idx}[2:0];"
        for idx in range(1, count + 1)
    )


def dynamic_pack_terms(port_width: int, prefix: str, positions: tuple[int, ...], mag_suffix: str = "m") -> list[str]:
    terms = []
    for idx, pos in enumerate(positions, start=1):
        terms.append(f"({{{{{port_width - MAG_BITS}{{1'b0}}}}, {prefix}{idx}{mag_suffix}}} << {pos})")
    return terms


def max_product_count(hybrid: SelectedHybrid) -> int:
    return max(hybrid.prefill.t, hybrid.decode.decode_t)


def max_weight_count(hybrid: SelectedHybrid) -> int:
    return max(hybrid.prefill.ny, hybrid.decode.ny)


def max_activation_count(hybrid: SelectedHybrid) -> int:
    return max(hybrid.prefill.nx, hybrid.decode.nx)


def lsb_signal_name(phase: str, product: Product, suffix: str = "") -> str:
    return f"{phase}_lsb{product.result_index}{suffix}"


def lsb_pipeline_lines(phase: str, products: list[Product]) -> tuple[list[str], list[str], list[str], list[str]]:
    decls: list[str] = []
    r1_assigns: list[str] = []
    r2_assigns: list[str] = []
    r3_assigns: list[str] = []
    for product in products:
        if product.overlap == 0:
            continue
        prod_name = f"{phase}_prod{product.result_index}"
        lsb = lsb_signal_name(phase, product)
        decls.append(
            f"wire [5:0] {prod_name} = product3x3(w{product.weight_index}m, a{product.activation_index}m);"
        )
        decls.append(f"wire [{product.overlap - 1}:0] {lsb} = {prod_name}[{product.overlap - 1}:0];")
        decls.append(f"reg  [{product.overlap - 1}:0] {lsb}_r1, {lsb}_r2, {lsb}_r3;")
        r1_assigns.append(f"    {lsb}_r1 <= {lsb};")
        r2_assigns.append(f"    {lsb}_r2 <= {lsb}_r1;")
        r3_assigns.append(f"    {lsb}_r3 <= {lsb}_r2;")
    return decls, r1_assigns, r2_assigns, r3_assigns


def correction_terms_source(phase: str, products: list[Product]) -> list[str]:
    terms: list[str] = []
    for product in products:
        if product.overlap == 0:
            continue
        wire_name = lsb_signal_name(phase, product, "_r1")
        terms.append(
            f"({{{{{P_WIDTH - product.overlap}{{1'b0}}}}, "
            f"{wire_name}[{product.overlap - 1}:0]}} << {product.start})"
        )
    return terms


def recover_expr_source(product: Product, phase: str) -> str:
    if product.overlap == 0:
        return f"dsp_p[{product.start + PRODUCT_WIDTH - 1}:{product.start}]"
    wire_name = lsb_signal_name(phase, product, "_r3")
    if product.overlap >= PRODUCT_WIDTH:
        return f"{wire_name}[{PRODUCT_WIDTH - 1}:0]"
    upper_hi = product.start + PRODUCT_WIDTH - 1
    upper_lo = product.start + product.overlap
    return f"{{dsp_p[{upper_hi}:{upper_lo}], {wire_name}[{product.overlap - 1}:0]}}"


def sign_assignment_lines(phase: str, products: list[Product], result_count: int) -> list[str]:
    lines: list[str] = []
    for idx in range(1, result_count + 1):
        if idx <= len(products):
            product = products[idx - 1]
            expr = f"w{product.weight_index}[3] ^ a{product.activation_index}[3]"
        else:
            expr = "1'b0"
        lines.append(f"assign signs_{phase}[{idx - 1}] = {expr};")
    return lines


def magnitude_assignment_lines(
    prefill_products: list[Product],
    decode_products: list[Product],
    result_count: int,
) -> list[str]:
    lines: list[str] = []
    for idx in range(1, result_count + 1):
        prefill_expr = recover_expr_source(prefill_products[idx - 1], "prefill") if idx <= len(prefill_products) else "6'b0"
        decode_expr = recover_expr_source(decode_products[idx - 1], "decode") if idx <= len(decode_products) else "6'b0"
        lines.append(f"wire [5:0] mag{idx} = mode_r3 ? {decode_expr} : {prefill_expr};")
    return lines


def result_output_lines(result_count: int) -> list[str]:
    lines: list[str] = []
    for idx in range(1, result_count + 1):
        bit = idx - 1
        lines.append(f"assign result{idx} = {{1'b0, mag{idx}}} ^ {{7{{signs_r3[{bit}]}}}};")
    sign_names = ", ".join(f"sign{idx}" for idx in range(result_count, 0, -1))
    lines.append(f"assign {{{sign_names}}} = signs_r3;")
    lines.append("assign valid_count = valid_count_r3;")
    return lines


def dsp48e2_helper_text() -> str:
    return """
module w4a4_overlap_dsp48e2_m_sub_c(
    input  wire        clk,
    input  wire [26:0] dsp_a,
    input  wire [17:0] dsp_b,
    input  wire [47:0] dsp_c,
    output wire [47:0] dsp_p
);

DSP48E2 #(
    .AMULTSEL("A"), .A_INPUT("DIRECT"), .BMULTSEL("B"), .B_INPUT("DIRECT"), .PREADDINSEL("A"),
    .RND(48'h000000000000), .USE_MULT("MULTIPLY"), .USE_SIMD("ONE48"), .USE_WIDEXOR("FALSE"), .XORSIMD("XOR24_48_96"),
    .AUTORESET_PATDET("NO_RESET"), .AUTORESET_PRIORITY("RESET"), .MASK(48'h3fffffffffff), .PATTERN(48'h000000000000),
    .SEL_MASK("MASK"), .SEL_PATTERN("PATTERN"), .USE_PATTERN_DETECT("NO_PATDET"),
    .IS_ALUMODE_INVERTED(4'b0000), .IS_CARRYIN_INVERTED(1'b0), .IS_CLK_INVERTED(1'b0), .IS_INMODE_INVERTED(5'b00000),
    .IS_OPMODE_INVERTED(9'b000000000), .IS_RSTALLCARRYIN_INVERTED(1'b0), .IS_RSTALUMODE_INVERTED(1'b0),
    .IS_RSTA_INVERTED(1'b0), .IS_RSTB_INVERTED(1'b0), .IS_RSTCTRL_INVERTED(1'b0), .IS_RSTC_INVERTED(1'b0),
    .IS_RSTD_INVERTED(1'b0), .IS_RSTINMODE_INVERTED(1'b0), .IS_RSTM_INVERTED(1'b0), .IS_RSTP_INVERTED(1'b0),
    .ACASCREG(1), .ADREG(0), .ALUMODEREG(1), .AREG(1), .BCASCREG(1), .BREG(1), .CARRYINREG(1),
    .CARRYINSELREG(1), .CREG(1), .DREG(0), .INMODEREG(1), .MREG(1), .OPMODEREG(1), .PREG(1)
) dsp_inst (
    .ACOUT(), .BCOUT(), .CARRYCASCOUT(), .MULTSIGNOUT(), .PCOUT(),
    .OVERFLOW(), .PATTERNBDETECT(), .PATTERNDETECT(), .UNDERFLOW(),
    .CARRYOUT(), .P(dsp_p), .XOROUT(),
    .ACIN(30'b0), .BCIN(18'b0), .CARRYCASCIN(1'b0), .MULTSIGNIN(1'b0), .PCIN(48'b0),
    .ALUMODE(4'b0001), .CARRYINSEL(3'b000), .CLK(clk), .INMODE(5'b00000), .OPMODE(9'b000110101),
    .A({3'b0, dsp_a}), .B(dsp_b), .C(dsp_c), .CARRYIN(1'b1), .D(27'b0),
    .CEA1(1'b0), .CEA2(1'b1), .CEAD(1'b0), .CEALUMODE(1'b1), .CEB1(1'b0), .CEB2(1'b1),
    .CEC(1'b1), .CECARRYIN(1'b1), .CECTRL(1'b1), .CED(1'b0), .CEINMODE(1'b1), .CEM(1'b1), .CEP(1'b1),
    .RSTA(1'b0), .RSTALLCARRYIN(1'b0), .RSTALUMODE(1'b0), .RSTB(1'b0), .RSTC(1'b0), .RSTCTRL(1'b0),
    .RSTD(1'b0), .RSTINMODE(1'b0), .RSTM(1'b0), .RSTP(1'b0)
);

endmodule
"""


def emit_source_compatible_rtl(hybrid: SelectedHybrid) -> None:
    depth = hybrid.depth
    top = top_name(depth)
    prefill_products = products_for_layout(hybrid.prefill)
    decode_products = products_for_layout(hybrid.decode)
    result_count = max_product_count(hybrid)
    weight_count = max_weight_count(hybrid)
    activation_count = max_activation_count(hybrid)

    prefill_a_terms = dynamic_pack_terms(A_WIDTH, "w", hybrid.prefill.y_pos)
    prefill_b_terms = dynamic_pack_terms(B_WIDTH, "a", hybrid.prefill.x_pos)
    decode_a_terms = dynamic_pack_terms(A_WIDTH, "w", hybrid.decode.y_pos)
    decode_b_terms = dynamic_pack_terms(B_WIDTH, "a", hybrid.decode.x_pos)

    prefill_lsb_decls, prefill_r1, prefill_r2, prefill_r3 = lsb_pipeline_lines("prefill", prefill_products)
    decode_lsb_decls, decode_r1, decode_r2, decode_r3 = lsb_pipeline_lines("decode", decode_products)
    lsb_decls = prefill_lsb_decls + decode_lsb_decls
    r1_assigns = prefill_r1 + decode_r1
    r2_assigns = prefill_r2 + decode_r2
    r3_assigns = prefill_r3 + decode_r3

    # C-port correction terms can overlap once pointwise overlap is relaxed
    # beyond 2.  They must be numerically accumulated; a bitwise OR would be
    # correct only for non-overlapping correction fields and would undercount
    # both functionality and LUT overhead for D>3.
    c_prefill = verilog_vector_sum(P_WIDTH, correction_terms_source("prefill", prefill_products))
    c_decode = verilog_vector_sum(P_WIDTH, correction_terms_source("decode", decode_products))

    text = f"""`timescale 1ns/1ps

/*
 * Auto-generated by scripts/generate_overlap_depth_sweep.py.
 * W4A4 overlap-depth cap: {depth}
 * RTL style: {RTL_STYLE}
 * Pointwise-overlap cap used by solver: {pointwise_limit_for_depth(depth)}
 * Input convention: weights use sign-magnitude w[3]|w[2:0]; activations use signed INT4 with -8 clamped to -7.
 * Output convention: source-compatible 7-bit magnitude-coded result plus separate sign bit.
 * Prefill layout: {hybrid.prefill.nx}x{hybrid.prefill.ny}, pointwise={hybrid.prefill.pointwise_overlap}, B/activation x={hybrid.prefill.x_pos}, A/weight y={hybrid.prefill.y_pos}, source={hybrid.prefill.source}
 * Decode layout: {hybrid.decode.nx}x{hybrid.decode.ny}, pointwise={hybrid.decode.pointwise_overlap}, B/activation x={hybrid.decode.x_pos}, A/weight y={hybrid.decode.y_pos}, source={hybrid.decode.source}
 * Note: {hybrid.note}
 *
 * The first sweep draft emitted a behavioral "(dsp_a * dsp_b) - dsp_c" and
 * 16-bit signed result registers.  That path is kept in the generator as a
 * reference, but it is intentionally not used here because it is not comparable
 * with the paper/source resource numbers.
 */

{dsp48e2_helper_text()}

module {top}(
    input  wire        clk,
    input  wire        mode,
{port_decl_lines("w", weight_count)}
{port_decl_lines("a", activation_count)}
{output_result_wire_decl_lines(result_count)}
{output_sign_decl_lines(result_count)}
    output wire [4:0]  valid_count
);

function [2:0] to_mag3;
    input [3:0] x;
    reg [3:0] temp;
    begin
        temp = x[3] ? (~x + 1'b1) : x;
        to_mag3 = (temp == 4'b1000) ? 3'b111 : temp[2:0];
    end
endfunction

function [5:0] product3x3;
    input [2:0] w;
    input [2:0] a;
    reg [5:0] partial_0;
    reg [5:0] partial_1;
    reg [5:0] partial_2;
    begin
        partial_0 = w[0] ? {{3'b0, a}} : 6'b0;
        partial_1 = w[1] ? {{2'b0, a, 1'b0}} : 6'b0;
        partial_2 = w[2] ? {{1'b0, a, 2'b0}} : 6'b0;
        product3x3 = partial_0 + partial_1 + partial_2;
    end
endfunction

{dynamic_weight_mag_lines(weight_count)}
{dynamic_activation_mag_lines(activation_count)}

wire [26:0] dsp_a_prefill =
    {verilog_vector_or(A_WIDTH, prefill_a_terms)};
wire [17:0] dsp_b_prefill =
    {verilog_vector_or(B_WIDTH, prefill_b_terms)};
wire [26:0] dsp_a_decode =
    {verilog_vector_or(A_WIDTH, decode_a_terms)};
wire [17:0] dsp_b_decode =
    {verilog_vector_or(B_WIDTH, decode_b_terms)};

{chr(10).join(lsb_decls)}

wire [{result_count - 1}:0] signs_prefill;
wire [{result_count - 1}:0] signs_decode;
{chr(10).join(sign_assignment_lines("prefill", prefill_products, result_count))}
{chr(10).join(sign_assignment_lines("decode", decode_products, result_count))}

wire [{result_count - 1}:0] signs_comb = mode ? signs_decode : signs_prefill;
wire [4:0] valid_count_comb = mode ? 5'd{hybrid.decode.decode_t} : 5'd{hybrid.prefill.t};

(* shreg_extract = "no" *) reg mode_r1, mode_r2, mode_r3;
(* shreg_extract = "no" *) reg [{result_count - 1}:0] signs_r1, signs_r2, signs_r3;
(* shreg_extract = "no" *) reg [4:0] valid_count_r1, valid_count_r2, valid_count_r3;

always @(posedge clk) begin
    mode_r1 <= mode;
    signs_r1 <= signs_comb;
    valid_count_r1 <= valid_count_comb;
{chr(10).join(r1_assigns)}
end

always @(posedge clk) begin
    mode_r2 <= mode_r1;
    signs_r2 <= signs_r1;
    valid_count_r2 <= valid_count_r1;
{chr(10).join(r2_assigns)}
end

always @(posedge clk) begin
    mode_r3 <= mode_r2;
    signs_r3 <= signs_r2;
    valid_count_r3 <= valid_count_r2;
{chr(10).join(r3_assigns)}
end

wire [47:0] c_prefill =
    {c_prefill};
wire [47:0] c_decode =
    {c_decode};

wire [26:0] dsp_a = mode ? dsp_a_decode : dsp_a_prefill;
wire [17:0] dsp_b = mode ? dsp_b_decode : dsp_b_prefill;
wire [47:0] dsp_c = mode_r1 ? c_decode : c_prefill;
wire [47:0] dsp_p;

w4a4_overlap_dsp48e2_m_sub_c dsp_core (
    .clk(clk),
    .dsp_a(dsp_a),
    .dsp_b(dsp_b),
    .dsp_c(dsp_c),
    .dsp_p(dsp_p)
);

{chr(10).join(magnitude_assignment_lines(prefill_products, decode_products, result_count))}
{chr(10).join(result_output_lines(result_count))}

endmodule
"""
    RTL_DIR.mkdir(parents=True, exist_ok=True)
    (RTL_DIR / f"{top}.v").write_text(text, encoding="utf-8", newline="\n")


def emit_rtl(hybrid: SelectedHybrid) -> None:
    if RTL_STYLE == "source_compatible_single_dsp":
        emit_source_compatible_rtl(hybrid)
        return

    # Legacy behavioral emitter retained for auditability.  It was useful for
    # quickly validating arbitrary layouts, but its wide signed outputs and
    # inferred "(dsp_a * dsp_b) - dsp_c" mapping are not comparable with the
    # Ultra-DSP source/resource reports.
    depth = hybrid.depth
    top = top_name(depth)
    helper = lsb_module_name(depth)
    prefill_products = products_for_layout(hybrid.prefill)
    decode_products = products_for_layout(hybrid.decode)

    prefill_a_terms = pack_terms(A_WIDTH, "w", hybrid.prefill.y_pos)
    prefill_b_terms = pack_activation_terms(B_WIDTH, hybrid.prefill.x_pos)
    decode_a_terms = pack_terms(A_WIDTH, "w", hybrid.decode.y_pos)
    decode_b_terms = pack_activation_terms(B_WIDTH, hybrid.decode.x_pos)

    lsb_lines = lsb_decl_lines(depth, "prefill", prefill_products)
    lsb_lines += lsb_decl_lines(depth, "decode", decode_products)

    # Legacy behavioral path: keep the same numeric correction accumulation
    # rule used by the source-compatible emitter.
    c_prefill = verilog_vector_sum(P_WIDTH, correction_terms(depth, "prefill", prefill_products))
    c_decode = verilog_vector_sum(P_WIDTH, correction_terms(depth, "decode", decode_products))

    prefill_assigns = "\n".join(result_assignment_lines(prefill_products, "prefill", hybrid.prefill.t))
    decode_assigns = "\n".join(result_assignment_lines(decode_products, "decode", hybrid.decode.decode_t))

    text = f"""`timescale 1ns/1ps

/*
 * Auto-generated by scripts/generate_overlap_depth_sweep.py.
 * W4A4 overlap-depth cap: {depth}
 * Pointwise-overlap cap used by solver: {pointwise_limit_for_depth(depth)}
 * Input convention: weights use sign-magnitude w[3]|w[2:0]; activations use signed INT4 with -8 clamped to -7.
 * Prefill layout: {hybrid.prefill.nx}x{hybrid.prefill.ny}, pointwise={hybrid.prefill.pointwise_overlap}, B/activation x={hybrid.prefill.x_pos}, A/weight y={hybrid.prefill.y_pos}
 * Decode layout: {hybrid.decode.nx}x{hybrid.decode.ny}, pointwise={hybrid.decode.pointwise_overlap}, B/activation x={hybrid.decode.x_pos}, A/weight y={hybrid.decode.y_pos}
 * Note: {hybrid.note}
 */

(* keep_hierarchy = "yes", use_dsp = "no" *)
module {helper} #(
    parameter integer DEPTH = {depth}
)(
    input  wire [2:0] w_mag,
    input  wire [2:0] a_mag,
    output wire [DEPTH-1:0] lsb
);

(* keep = "true", use_dsp = "no" *) wire [5:0] product_exact;
assign product_exact = w_mag * a_mag;
assign lsb = product_exact[DEPTH-1:0];

endmodule

module {top}(
    input  wire        clk,
    input  wire        mode,
{port_decl_lines("w", RTL_WEIGHT_PORTS)}
{port_decl_lines("a", RTL_ACTIVATION_PORTS, signed=True)}
{output_result_decl_lines()}
    output reg  [4:0]  valid_count
);

function [2:0] amag;
    input signed [3:0] x;
    reg signed [3:0] xc;
    begin
        xc = (x == -4'sd8) ? -4'sd7 : x;
        amag = xc[3] ? -xc : xc;
    end
endfunction

function signed [15:0] recover;
    input [5:0] mag;
    input sign;
    begin
        recover = sign ? -$signed({{10'b0, mag}}) : $signed({{10'b0, mag}});
    end
endfunction

{activation_mag_lines()}

wire [26:0] dsp_a_prefill =
    {verilog_vector_or(A_WIDTH, prefill_a_terms)};
wire [17:0] dsp_b_prefill =
    {verilog_vector_or(B_WIDTH, prefill_b_terms)};
wire [26:0] dsp_a_decode =
    {verilog_vector_or(A_WIDTH, decode_a_terms)};
wire [17:0] dsp_b_decode =
    {verilog_vector_or(B_WIDTH, decode_b_terms)};

{chr(10).join(lsb_lines)}

wire [47:0] c_prefill =
    {c_prefill};
wire [47:0] c_decode =
    {c_decode};

wire [26:0] dsp_a = mode ? dsp_a_decode : dsp_a_prefill;
wire [17:0] dsp_b = mode ? dsp_b_decode : dsp_b_prefill;
wire [47:0] dsp_c = mode ? c_decode : c_prefill;
(* use_dsp = "yes" *) wire [47:0] p = (dsp_a * dsp_b) - dsp_c;

always @(posedge clk) begin
    if (mode == 1'b0) begin
{prefill_assigns}
    end else begin
{decode_assigns}
    end
end

endmodule
"""
    RTL_DIR.mkdir(parents=True, exist_ok=True)
    (RTL_DIR / f"{top}.v").write_text(text, encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check-trials", type=int, default=2000)
    parser.add_argument("--skip-self-check", action="store_true")
    args = parser.parse_args()

    all_rows: list[dict[str, object]] = []
    selected: list[SelectedHybrid] = []

    for depth in DEPTH_CAPS:
        candidates = enumerate_candidates(depth)
        prefill_points = [(1.0 / c.t, c.cost / c.t) for c in candidates]
        decode_points = [(1.0 / c.decode_t, c.cost / c.decode_t) for c in candidates]
        prefill_pareto = pareto_mask(prefill_points)
        decode_pareto = pareto_mask(decode_points)

        for candidate, is_prefill, is_decode in zip(candidates, prefill_pareto, decode_pareto):
            all_rows.append(candidate_row(candidate, is_prefill, is_decode))

        selected.append(select_hybrid(depth, candidates, prefill_pareto, decode_pareto))

    if not args.skip_self_check:
        self_check(selected, args.self_check_trials)

    for hybrid in selected:
        emit_rtl(hybrid)

    write_layout_csvs(all_rows, selected)
    print(f"Wrote {len(all_rows)} legal-layout rows to {OUT_DIR / 'layouts_w4a4_all.csv'}")
    print(f"Wrote {len(selected)} selected layouts to {OUT_DIR / 'layouts_w4a4_selected.csv'}")
    print(f"Wrote RTL files under {RTL_DIR}")
    if not args.skip_self_check:
        print(f"Arithmetic self-check passed for {len(selected)} selected hybrid PEs")


if __name__ == "__main__":
    main()
