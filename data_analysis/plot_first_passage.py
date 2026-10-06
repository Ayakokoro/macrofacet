"""Render first-passage CSV outputs as SVG without third-party dependencies."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


COLORS = (
    "#1565c0",
    "#c62828",
    "#2e7d32",
    "#6a1b9a",
    "#ef6c00",
    "#00838f",
    "#ad1457",
    "#558b2f",
    "#4527a0",
    "#6d4c41",
)


def _float(row: dict[str, str], name: str) -> float:
    try:
        return float(row[name])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid or missing numeric CSV field {name!r}") from error


def _is_true(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"required C++ output does not exist: {path}")
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _resolve_path(value: str, repository_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repository_root / path


def _format_number(value: float) -> str:
    if abs(value) >= 1000.0 or (0.0 < abs(value) < 0.001):
        return f"{value:.3g}"
    return f"{value:.4g}"


def _svg_line_chart(
    path: Path,
    *,
    title: str,
    x_label: str,
    y_label: str,
    series: Sequence[dict[str, Any]],
    footer: str = "",
    y_limits: tuple[float, float] | None = None,
) -> None:
    finite_points = [
        (float(x), float(y))
        for item in series
        for x, y in item["points"]
        if math.isfinite(float(x)) and math.isfinite(float(y))
    ]
    if not finite_points:
        raise ValueError(f"plot {title!r} contains no finite points")

    x_min = min(x for x, _ in finite_points)
    x_max = max(x for x, _ in finite_points)
    if x_max <= x_min:
        x_min, x_max = 0.0, max(1.0, x_max)
    if y_limits is None:
        y_min = min(0.0, min(y for _, y in finite_points))
        y_max = max(y for _, y in finite_points)
        if y_max <= y_min:
            y_max = y_min + 1.0
        margin = 0.06 * (y_max - y_min)
        y_max += margin
        if y_min < 0.0:
            y_min -= margin
    else:
        y_min, y_max = y_limits

    left, right, top, bottom = 84.0, 770.0, 48.0, 410.0
    plot_width, plot_height = right - left, bottom - top

    def sx(value: float) -> float:
        return left + plot_width * (value - x_min) / (x_max - x_min)

    def sy(value: float) -> float:
        return bottom - plot_height * (value - y_min) / (y_max - y_min)

    elements = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1100 500">',
        '<rect width="1100" height="500" fill="white"/>',
        f'<text x="450" y="25" text-anchor="middle" font-size="17">{html.escape(title)}</text>',
    ]
    for tick in range(6):
        fraction = tick / 5.0
        x_value = x_min + fraction * (x_max - x_min)
        x = left + fraction * plot_width
        y_value = y_max - fraction * (y_max - y_min)
        y = top + fraction * plot_height
        elements.extend(
            [
                f'<path d="M{x:.3f} {top} V{bottom}" stroke="#eeeeee"/>',
                f'<path d="M{left} {y:.3f} H{right}" stroke="#eeeeee"/>',
                f'<text x="{x:.3f}" y="432" text-anchor="middle" font-size="11">'
                f'{html.escape(_format_number(x_value))}</text>',
                f'<text x="76" y="{y + 4:.3f}" text-anchor="end" font-size="11">'
                f'{html.escape(_format_number(y_value))}</text>',
            ]
        )
    elements.append(
        f'<path d="M{left} {top} V{bottom} H{right}" fill="none" stroke="#222"/>'
    )

    legend_x = 790.0
    for index, item in enumerate(series):
        color = item.get("color", COLORS[index % len(COLORS)])
        dash = item.get("dash", "")
        dash_attribute = f' stroke-dasharray="{dash}"' if dash else ""
        points = [
            f"{sx(float(x)):.3f},{sy(float(y)):.3f}"
            for x, y in item["points"]
            if math.isfinite(float(x)) and math.isfinite(float(y))
        ]
        if points:
            elements.append(
                f'<polyline fill="none" stroke="{color}" stroke-width="2"'
                f'{dash_attribute} points="{" ".join(points)}"/>'
            )
        legend_y = 55.0 + 18.0 * index
        elements.extend(
            [
                f'<path d="M{legend_x} {legend_y} h18" stroke="{color}"'
                f' stroke-width="2"{dash_attribute}/>',
                f'<text x="{legend_x + 23}" y="{legend_y + 4}" font-size="11">'
                f'{html.escape(str(item["label"]))}</text>',
            ]
        )

    elements.extend(
        [
            f'<text x="{(left + right) / 2}" y="465" text-anchor="middle">'
            f'{html.escape(x_label)}</text>',
            f'<text x="18" y="{(top + bottom) / 2}" text-anchor="middle" '
            f'transform="rotate(-90 18 {(top + bottom) / 2})">{html.escape(y_label)}</text>',
            f'<text x="{left}" y="485" font-size="11">{html.escape(footer)}</text>',
            "</svg>\n",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(elements), encoding="utf-8")


def _collision_groups(
    rows: list[dict[str, str]], config: dict[str, Any]
) -> list[tuple[tuple[str, str], str, list[dict[str, str]]]]:
    training = [row for row in rows if _is_true(row.get("training_resolution", "0"))]
    if not training:
        raise ValueError("collision curve CSV has no training_resolution=1 rows")

    comparison = config.get("comparison", "kernels")
    state_ids = list(config.get("state_ids", []))
    kernel_id = config.get("kernel_id")
    if comparison == "kernels":
        if len(state_ids) != 1:
            raise ValueError("comparison='kernels' requires exactly one state_ids entry")
        selected_state = state_ids[0]
        order = []
        for row in training:
            if row["state_id"] == selected_state and row["kernel_id"] not in order:
                order.append(row["kernel_id"])
        keys = [(item, selected_state) for item in order]
        labels = {key: key[0] for key in keys}
    elif comparison == "states":
        if not kernel_id or not state_ids:
            raise ValueError("comparison='states' requires kernel_id and nonempty state_ids")
        keys = [(kernel_id, state_id) for state_id in state_ids]
        labels = {key: key[1] for key in keys}
    else:
        raise ValueError("comparison must be 'kernels' or 'states'")

    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in training:
        grouped[(row["kernel_id"], row["state_id"])].append(row)
    missing = [key for key in keys if key not in grouped]
    if missing:
        raise ValueError(f"selected kernel/state series are absent from CSV: {missing}")
    return [
        (key, labels[key], sorted(grouped[key], key=lambda row: _float(row, "q_end")))
        for key in keys
    ]


def _plot_collision_curves(
    groups: list[tuple[tuple[str, str], str, list[dict[str, str]]]],
    analysis: dict[str, Any],
    output_directory: Path,
) -> list[Path]:
    plots = analysis.get("plots", {})
    physical_distance = analysis.get("x_axis", "q") == "distance"
    if analysis.get("x_axis", "q") not in ("q", "distance"):
        raise ValueError("x_axis must be 'q' or 'distance'")
    minimum_risk = int(analysis.get("minimum_risk_set", 1))
    comparison = analysis.get("comparison", "kernels")
    if comparison == "states":
        footer = f"kernel: {analysis['kernel_id']}; training resolution"
    else:
        footer = f"state: {analysis['state_ids'][0]}; training resolution"

    definitions = (
        ("survival", "survival", "first_passage_survival.svg", "Survival", (0.0, 1.0)),
        ("density", "first_passage_density", "first_passage_density.svg", "Density", None),
        ("hazard", "hazard_mc", "first_passage_hazard.svg", "Hazard", None),
        (
            "cumulative_hazard",
            "cumulative_hazard_product_limit",
            "first_passage_cumulative_hazard.svg",
            "Cumulative hazard",
            None,
        ),
    )
    generated: list[Path] = []
    for switch, field, file_name, label, limits in definitions:
        if not plots.get(switch, switch != "density"):
            continue
        chart_series = []
        for index, (_, legend, rows) in enumerate(groups):
            points = []
            for row in rows:
                if switch == "hazard" and int(row["at_risk"]) < minimum_risk:
                    continue
                value = _float(row, field)
                ell = _float(row, "ell") if physical_distance else 1.0
                if physical_distance and switch in ("hazard", "density"):
                    value /= ell
                if math.isfinite(value):
                    points.append((_float(row, "q_end") * (ell if physical_distance else 1.0), value))
            chart_series.append(
                {"label": legend, "points": points, "color": COLORS[index % len(COLORS)]}
            )
        path = output_directory / file_name
        _svg_line_chart(
            path,
            title=f"First-passage {label.lower()}",
            x_label="Distance t (scene units)" if physical_distance else "q = t / ell",
            y_label=label + (" (1 / scene unit)" if physical_distance and switch in ("hazard", "density") else ""),
            series=chart_series,
            footer=footer,
            y_limits=limits,
        )
        generated.append(path)
    return generated


def _plot_mean_profiles(
    groups: list[tuple[tuple[str, str], str, list[dict[str, str]]]],
    analysis: dict[str, Any], input_directory: Path, output_directory: Path,
) -> list[Path]:
    if not analysis.get("plots", {}).get("mean_profile", False):
        return []
    rows = _read_csv(input_directory / "first_passage_mean_profiles.csv")
    physical = analysis.get("x_axis", "q") == "distance"
    series = []
    for index, (key, legend, _) in enumerate(groups):
        selected = sorted(
            (row for row in rows if (row["kernel_id"], row["state_id"]) == key),
            key=lambda row: _float(row, "q"),
        )
        if not selected:
            raise ValueError(f"mean profile missing for {key}")
        series.append({"label": legend, "color": COLORS[index % len(COLORS)],
            "points": [(_float(row, "distance" if physical else "q"),
                        _float(row, "mean" if physical else "beta_mean")) for row in selected]})
    path = output_directory / "first_passage_mean_profile.svg"
    _svg_line_chart(path, title="SDF mean along the ray",
        x_label="Distance t (scene units)" if physical else "q = t / ell",
        y_label="Mean SDF (scene units)" if physical else "m(t) / sigma",
        series=series, footer="Deterministic full-field mean, not a GP realization")
    return [path]


def _slope_density(
    groups: list[tuple[tuple[str, str], str, list[dict[str, str]]]],
    analysis: dict[str, Any],
    input_directory: Path,
    output_directory: Path,
) -> list[Path]:
    density_config = analysis.get("plots", {}).get("crossing_slope_density", {})
    if not density_config.get("enabled", False):
        return []
    bins = int(density_config.get("bins", 64))
    if bins < 2:
        raise ValueError("crossing_slope_density.bins must be at least 2")
    raw_intervals = density_config.get("q_intervals", [])
    intervals = [(float(item[0]), float(item[1])) for item in raw_intervals]
    if not intervals or any(not (0.0 <= begin < end) for begin, end in intervals):
        raise ValueError("crossing_slope_density.q_intervals must be increasing nonnegative pairs")

    selected_keys = [key for key, _, _ in groups]
    values: dict[tuple[tuple[str, str], int], list[float]] = defaultdict(list)
    sample_path = input_directory / "first_passage_samples.csv"
    if not sample_path.is_file():
        raise FileNotFoundError(
            "crossing-slope density needs first_passage_samples.csv; "
            "set monte_carlo.write_raw_samples=true in the C++ config"
        )
    with sample_path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            key = (row["kernel_id"], row["state_id"])
            if key not in selected_keys or not _is_true(row["training_resolution"]):
                continue
            if not _is_true(row["event"]):
                continue
            try:
                q = float(row["event_q"])
                slope = float(row["crossing_slope"])
            except ValueError:
                continue
            if not math.isfinite(slope) or slope < 0.0:
                continue
            for interval_index, (begin, end) in enumerate(intervals):
                if begin < q <= end:
                    values[(key, interval_index)].append(slope)

    all_slopes = [value for observations in values.values() for value in observations]
    configured_maximum = density_config.get("slope_max")
    if configured_maximum is None:
        slope_maximum = max(all_slopes, default=1.0) * 1.000001
    else:
        slope_maximum = float(configured_maximum)
    if not math.isfinite(slope_maximum) or slope_maximum <= 0.0:
        raise ValueError("crossing_slope_density.slope_max must be positive")
    bin_width = slope_maximum / bins

    for stale in output_directory.glob("first_passage_crossing_slope_density_q*.svg"):
        stale.unlink()
    output_directory.mkdir(parents=True, exist_ok=True)
    density_csv = output_directory / "first_passage_crossing_slope_density.csv"
    header = (
        "q_interval",
        "q_begin",
        "q_end",
        "kernel_id",
        "kernel_type",
        "state_id",
        "beta_0",
        "beta_a",
        "beta_g",
        "events",
        "slope_bin",
        "slope_begin",
        "slope_end",
        "slope_center",
        "count",
        "probability_density",
    )
    generated = [density_csv]
    with density_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=header)
        writer.writeheader()
        for interval_index, (q_begin, q_end) in enumerate(intervals):
            chart_series = []
            for series_index, (key, legend, curve_rows) in enumerate(groups):
                observations = values[(key, interval_index)]
                counts = [0] * bins
                for slope in observations:
                    if slope <= slope_maximum:
                        index = min(bins - 1, int(slope / bin_width))
                        counts[index] += 1
                event_count = sum(counts)
                densities = [
                    count / (event_count * bin_width) if event_count else 0.0
                    for count in counts
                ]
                metadata = curve_rows[0]
                for index, (count, density) in enumerate(zip(counts, densities)):
                    begin = index * bin_width
                    end = begin + bin_width
                    writer.writerow(
                        {
                            "q_interval": interval_index,
                            "q_begin": q_begin,
                            "q_end": q_end,
                            "kernel_id": key[0],
                            "kernel_type": metadata["kernel_type"],
                            "state_id": key[1],
                            "beta_0": metadata["beta_0"],
                            "beta_a": metadata["beta_a"],
                            "beta_g": metadata["beta_g"],
                            "events": event_count,
                            "slope_bin": index,
                            "slope_begin": begin,
                            "slope_end": end,
                            "slope_center": 0.5 * (begin + end),
                            "count": count,
                            "probability_density": density,
                        }
                    )
                step_points: list[tuple[float, float]] = [(0.0, 0.0)]
                for index, density in enumerate(densities):
                    begin = index * bin_width
                    end = begin + bin_width
                    step_points.extend(((begin, density), (end, density)))
                step_points.append((slope_maximum, 0.0))
                chart_series.append(
                    {
                        "label": f"{legend} (n={event_count})",
                        "points": step_points,
                        "color": COLORS[series_index % len(COLORS)],
                    }
                )
            if analysis.get("comparison", "kernels") == "states":
                selection = f"kernel: {analysis['kernel_id']}"
            else:
                selection = f"state: {analysis['state_ids'][0]}"
            path = output_directory / f"first_passage_crossing_slope_density_q{interval_index}.svg"
            _svg_line_chart(
                path,
                title="First-passage crossing-slope density",
                x_label="r = -(ell / sigma) F'(tau)",
                y_label="Conditional density",
                series=chart_series,
                footer=f"q in ({q_begin:g}, {q_end:g}]; {selection}; training resolution",
            )
            generated.append(path)
    return generated


def _plot_fixed_value(
    rows: list[dict[str, str]], analysis: dict[str, Any], output_directory: Path
) -> list[Path]:
    by_kernel: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_kernel[row["kernel_id"]].append(row)
    selected: list[tuple[str, list[dict[str, str]]]] = []
    for kernel_id, kernel_rows in by_kernel.items():
        finest = min(_float(row, "grid_step") for row in kernel_rows)
        chosen = [
            row
            for row in kernel_rows
            if math.isclose(_float(row, "grid_step"), finest, rel_tol=1e-12, abs_tol=0.0)
        ]
        selected.append((kernel_id, sorted(chosen, key=lambda row: _float(row, "time"))))

    plots = analysis.get("plots", {})
    minimum_risk = int(analysis.get("minimum_risk_set", 1))
    generated: list[Path] = []

    def points(
        rows_for_kernel: list[dict[str, str]],
        field: str,
        risk: bool = False,
        interval_end: bool = False,
    ):
        result = []
        for row in rows_for_kernel:
            if risk and int(row["at_risk"]) < minimum_risk:
                continue
            value = _float(row, field)
            if math.isfinite(value):
                x = _float(row, "time_over_length_scale")
                if interval_end:
                    time = _float(row, "time")
                    x *= _float(row, "bin_end") / time
                result.append((x, value))
        return result

    if plots.get("survival", True):
        series = []
        for index, (kernel_id, kernel_rows) in enumerate(selected):
            color = COLORS[index % len(COLORS)]
            series.extend(
                (
                    {"label": f"{kernel_id}: MC", "points": points(kernel_rows, "survival", interval_end=True), "color": color},
                    {"label": f"{kernel_id}: Sigma1", "points": points(kernel_rows, "start_conditioned_sigma1_survival", interval_end=True), "color": color, "dash": "8 5"},
                    {"label": f"{kernel_id}: endpoint", "points": points(kernel_rows, "start_conditioned_endpoint_survival", interval_end=True), "color": color, "dash": "2 4"},
                )
            )
        path = output_directory / "first_passage_survival.svg"
        _svg_line_chart(path, title="First-passage survival", x_label="t / ell", y_label="Survival", series=series, footer="finest grid per kernel", y_limits=(0.0, 1.0))
        generated.append(path)

    if plots.get("hazard", True):
        series = []
        for index, (kernel_id, kernel_rows) in enumerate(selected):
            color = COLORS[index % len(COLORS)]
            series.extend(
                (
                    {"label": f"{kernel_id}: MC", "points": points(kernel_rows, "hazard_mc", True), "color": color},
                    {"label": f"{kernel_id}: Sigma1", "points": points(kernel_rows, "endpoint_conditioned_hazard_sigma1", True), "color": color, "dash": "8 5"},
                )
            )
        path = output_directory / "first_passage_hazard.svg"
        _svg_line_chart(path, title="First-passage hazard", x_label="t / ell", y_label="Hazard", series=series, footer="finest grid per kernel")
        generated.append(path)

    if plots.get("rice_density", True):
        series = []
        for index, (kernel_id, kernel_rows) in enumerate(selected):
            color = COLORS[index % len(COLORS)]
            series.extend(
                (
                    {"label": f"{kernel_id}: MC", "points": points(kernel_rows, "first_passage_density"), "color": color},
                    {"label": f"{kernel_id}: Rice 1", "points": points(kernel_rows, "rice_w1"), "color": color, "dash": "8 5"},
                    {"label": f"{kernel_id}: Rice 2", "points": points(kernel_rows, "rice_density_order2"), "color": color, "dash": "2 4"},
                )
            )
        path = output_directory / "first_passage_rice_density.svg"
        _svg_line_chart(path, title="First-passage density and Rice truncations", x_label="t / ell", y_label="Density", series=series, footer="finest grid per kernel")
        generated.append(path)
    return generated


def run(config_path: str | Path) -> list[Path]:
    config_path = Path(config_path).resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        analysis = json.load(stream)
    if analysis.get("schema_version") != 1:
        raise ValueError("data-analysis schema_version must be 1")
    repository_root = Path(__file__).resolve().parents[1]
    input_directory = _resolve_path(analysis["input_directory"], repository_root)
    output_directory = _resolve_path(
        analysis.get("output_directory", analysis["input_directory"]), repository_root
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    rows = _read_csv(input_directory / "first_passage_curves.csv")
    if not rows:
        raise ValueError("first_passage_curves.csv is empty")
    if "state_id" in rows[0]:
        groups = _collision_groups(rows, analysis)
        generated = _plot_collision_curves(groups, analysis, output_directory)
        generated.extend(_plot_mean_profiles(groups, analysis, input_directory, output_directory))
        generated.extend(
            _slope_density(groups, analysis, input_directory, output_directory)
        )
        return generated
    return _plot_fixed_value(rows, analysis, output_directory)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Visualize CSV data generated by macrofacet first-passage experiments."
    )
    parser.add_argument("--config", required=True, type=Path, help="visualization JSON")
    arguments = parser.parse_args(argv)
    generated = run(arguments.config)
    for path in generated:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
