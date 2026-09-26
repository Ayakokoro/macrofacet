#!/usr/bin/env python3
"""Dependency-free SVG plots for Macrofacet CSV experiment outputs."""

import argparse
import csv
import math
from pathlib import Path


COLORS = {"classic": "#2563eb", "classic expected": "#059669"}


def load(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def points(values, left, top, width, height, x_range, y_range):
    x0, x1 = x_range
    y0, y1 = y_range
    result = []
    for x, y in values:
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        px = left + width * (x - x0) / max(x1 - x0, 1e-30)
        py = top + height * (1.0 - (y - y0) / max(y1 - y0, 1e-30))
        result.append(f"{px:.2f},{py:.2f}")
    return " ".join(result)


def panel(svg, title, series, left, top, width, height, y_min=0.0):
    all_values = [item for values in series.values() for item in values]
    x_max = max(x for x, _ in all_values)
    y_max = max(y for _, y in all_values if math.isfinite(y))
    y_max = max(y_max * 1.05, 1e-12)
    svg.append(f'<rect x="{left}" y="{top}" width="{width}" height="{height}" fill="white" stroke="#64748b"/>')
    svg.append(f'<text x="{left}" y="{top-10}" font-size="16" font-family="sans-serif">{title}</text>')
    for name, values in series.items():
        color = COLORS.get(name, "#7c3aed")
        path = points(values, left, top, width, height, (0.0, x_max), (y_min, y_max))
        svg.append(f'<polyline points="{path}" fill="none" stroke="{color}" stroke-width="2"/>')
    svg.append(f'<text x="{left}" y="{top+height+18}" font-size="12" font-family="sans-serif">0</text>')
    svg.append(f'<text x="{left+width-40}" y="{top+height+18}" font-size="12" font-family="sans-serif">t={x_max:.3g}</text>')
    svg.append(f'<text x="{left+4}" y="{top+14}" font-size="11" font-family="sans-serif">{y_max:.3g}</text>')


def plot_flights(directory):
    rows = load(directory / "flight_curves.csv")
    hist = load(directory / "flight_histograms.csv")
    hazards, survivals, observed, expected = {}, {}, {}, {}
    for row in rows:
        mode = row["mode"]
        hazards.setdefault(mode, []).append((float(row["t"]), float(row["h"])))
        survivals.setdefault(mode, []).append((float(row["t"]), float(row["T_model"])))
    for row in hist:
        if row["escape_bin"] == "1":
            continue
        mode = row["mode"]
        center = 0.5 * (float(row["bin_left"]) + float(row["bin_right"]))
        observed.setdefault(mode, []).append((center, float(row["observed_mass"])))
        expected.setdefault(mode + " expected", []).append((center, float(row["expected_mass"])))

    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="850" viewBox="0 0 1100 850">',
           '<rect width="100%" height="100%" fill="#f8fafc"/>']
    panel(svg, "Hazard h(t)", hazards, 70, 70, 960, 190)
    panel(svg, "Model survival exp(-integral h)", survivals, 70, 335, 960, 190)
    combined = dict(observed)
    combined.update(expected)
    panel(svg, "Flight-bin mass: observed and expected", combined, 70, 600, 960, 190)
    x, y = 75, 30
    for mode in ("classic", "classic expected"):
        svg.append(f'<line x1="{x}" y1="{y}" x2="{x+28}" y2="{y}" stroke="{COLORS[mode]}" stroke-width="3"/>')
        svg.append(f'<text x="{x+34}" y="{y+4}" font-size="13" font-family="sans-serif">{mode}</text>')
        x += 190
    svg.append('</svg>')
    (directory / "flight_curves.svg").write_text("\n".join(svg), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    plot_flights(args.directory)


if __name__ == "__main__":
    main()

