"""Flux-weighted first-hit slope densities from fixed-endpoint GP proposals."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

if __package__:
    from .plot_first_passage import COLORS, _float, _is_true, _read_csv, _resolve_path, _svg_line_chart
else:
    from plot_first_passage import COLORS, _float, _is_true, _read_csv, _resolve_path, _svg_line_chart


def _selected_keys(rows: list[dict[str, str]], config: dict[str, Any]) -> list[tuple[str, str]]:
    states = config.get("state_ids", [])
    if config.get("comparison", "states") == "states":
        if not config.get("kernel_id") or not states:
            raise ValueError("states comparison requires kernel_id and state_ids")
        return [(config["kernel_id"], state) for state in states]
    if config.get("comparison") == "kernels" and len(states) == 1:
        kernels = list(dict.fromkeys(row["kernel_id"] for row in rows if row["state_id"] == states[0]))
        if kernels:
            return [(kernel, states[0]) for kernel in kernels]
    raise ValueError("kernels comparison requires exactly one existing state_ids entry")


def run(config_path: str | Path) -> list[Path]:
    with Path(config_path).open(encoding="utf-8") as stream:
        config = json.load(stream)
    if config.get("schema_version") != 1:
        raise ValueError("schema_version must be 1")
    root = Path(__file__).resolve().parents[1]
    source = _resolve_path(config["input_directory"], root)
    destination = _resolve_path(config.get("output_directory", config["input_directory"]), root)
    summaries = [row for row in _read_csv(source / "fixed_endpoint_summary.csv")
                 if _is_true(row["training_resolution"])]
    keys = _selected_keys(summaries, config)
    distances = config.get("distances", list(dict.fromkeys(_float(row, "target_distance") for row in summaries)))
    if not distances or any(not math.isfinite(float(t)) or float(t) <= 0 for t in distances):
        raise ValueError("distances must be nonempty positive finite physical distances")
    distances = list(dict.fromkeys(float(t) for t in distances))
    bins = int(config.get("bins", 48))
    units = config.get("slope_units", "physical")
    if bins < 2 or units not in ("physical", "dimensionless"):
        raise ValueError("bins must be >=2 and slope_units physical or dimensionless")
    groups: dict[tuple[int, tuple[str, str]], list[tuple[float, float]]] = defaultdict(list)
    metadata = {}
    for index, distance in enumerate(distances):
        for key in keys:
            matches = [row for row in summaries if (row["kernel_id"], row["state_id"]) == key
                       and math.isclose(_float(row, "target_distance"), distance, rel_tol=1e-10, abs_tol=0.0)]
            if len(matches) != 1:
                raise ValueError(f"expected one finest-resolution summary for {key} at {distance}, found {len(matches)}")
            metadata[(index, key)] = matches[0]
    with (source / "fixed_endpoint_samples.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            key = row["kernel_id"], row["state_id"]
            if key not in keys or not _is_true(row["training_resolution"]):
                continue
            weight = _float(row, "flux_weight_q")
            if weight < 0 or not math.isfinite(weight):
                raise ValueError("invalid flux weight")
            if weight == 0:
                continue
            slope = -_float(row, "endpoint_derivative" if units == "physical" else "endpoint_slope")
            if not _is_true(row["survived_to_endpoint"]) or not math.isfinite(slope) or slope <= 0:
                raise ValueError("positive flux weights require survival and a negative endpoint slope")
            for index, distance in enumerate(distances):
                if math.isclose(_float(row, "target_distance"), distance, rel_tol=1e-10, abs_tol=0.0):
                    groups[(index, key)].append((slope, weight))
    # Validate every requested series before replacing any visualization outputs.
    for index, distance in enumerate(distances):
        for key in keys:
            observations = groups[(index, key)]
            if not observations:
                raise ValueError(f"no positive-weight samples for {key} at {distance}; increase proposals or change distance")
            total = sum(weight for _, weight in observations)
            if not math.isclose(total, _float(metadata[(index, key)], "sum_weight_q"), rel_tol=1e-8, abs_tol=1e-10):
                raise ValueError("sample weights disagree with C++ summary; outputs may come from different runs")
    automatic_max = max(slope for observations in groups.values() for slope, _ in observations) * 1.000001
    slope_max = float(config.get("slope_max", automatic_max))
    if not math.isfinite(slope_max) or slope_max <= 0:
        raise ValueError("slope_max must be finite and positive")
    if any(slope > slope_max for observations in groups.values() for slope, _ in observations):
        raise ValueError("slope_max truncates positive-weight samples; increase it or omit it")
    destination.mkdir(parents=True, exist_ok=True)
    result_csv = destination / "fixed_endpoint_slope_density.csv"
    fields = ["target_distance", "target_q", "kernel_id", "state_id", "slope_units", "slope_bin",
              "slope_begin", "slope_end", "count", "weight_sum", "total_weight", "effective_samples",
              "probability_density"]
    generated = [result_csv]
    width = slope_max / bins
    with result_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, distance in enumerate(distances):
            series = []
            for color_index, key in enumerate(keys):
                observations = groups[(index, key)]
                total = sum(weight for _, weight in observations)
                ess = total * total / sum(weight * weight for _, weight in observations)
                counts, weights = [0] * bins, [0.0] * bins
                for slope, weight in observations:
                    bin_index = min(bins - 1, int(slope / width))
                    counts[bin_index] += 1
                    weights[bin_index] += weight
                points = [(0.0, 0.0)]
                for bin_index, (count, weight) in enumerate(zip(counts, weights)):
                    begin, end = width * bin_index, width * (bin_index + 1)
                    density = weight / (total * width)
                    points.extend(((begin, density), (end, density)))
                    writer.writerow({"target_distance": distance,
                        "target_q": metadata[(index, key)]["target_q"], "kernel_id": key[0], "state_id": key[1],
                        "slope_units": units, "slope_bin": bin_index, "slope_begin": begin, "slope_end": end,
                        "count": count, "weight_sum": weight, "total_weight": total, "effective_samples": ess,
                        "probability_density": density})
                points.append((slope_max, 0.0))
                legend = key[1] if config.get("comparison", "states") == "states" else key[0]
                series.append({"label": f"{legend} (n={len(observations)}, ESS={ess:.0f})",
                               "color": COLORS[color_index % len(COLORS)], "points": points})
            path = destination / f"fixed_endpoint_slope_density_d{index}.svg"
            _svg_line_chart(path, title=f"First-hit slope density at distance {distance:g}",
                x_label="Incoming slope -F'(t*)" if units == "physical" else "Incoming slope -(ell/sigma) F'(t*)",
                y_label="Probability density", series=series,
                footer="Fixed endpoint F(t*)=0; no earlier crossing; negative-slope flux weights; finest GP grid")
            generated.append(path)
    return generated


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    for path in run(parser.parse_args(argv).config):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
