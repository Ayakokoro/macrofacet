from __future__ import annotations

import argparse
import csv
import sys

import numpy as np

from .inference import evaluate_curves, invert_cumulative_hazard, load_checkpoint


def _numbers(value: str) -> np.ndarray:
    return np.asarray([float(item) for item in value.split(",")], dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained first-passage surrogate")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--beta-0", type=float, required=True)
    parser.add_argument("--beta-a", type=float, required=True)
    parser.add_argument("--beta-g", type=float, required=True)
    parser.add_argument("--q", help="comma-separated dimensionless distances")
    parser.add_argument("--optical-depth", help="comma-separated H targets to invert")
    parser.add_argument("--ell", type=float, default=1.0, help="physical length scale")
    args = parser.parse_args()
    if not args.q and not args.optical_depth:
        parser.error("at least one of --q or --optical-depth is required")

    model, basis, _ = load_checkpoint(args.checkpoint)
    beta = np.asarray([args.beta_0, args.beta_a, args.beta_g], dtype=np.float64)
    coefficients = evaluate_curves(model, basis, beta, np.asarray([0.0]))["coefficients"][0]

    if args.q:
        q = _numbers(args.q)
        values = evaluate_curves(model, basis, beta, q)
        writer = csv.writer(sys.stdout)
        writer.writerow(
            ["q", "distance", "cumulative_hazard", "transmittance", "dimensionless_extinction", "extinction"]
        )
        for index, value in enumerate(q):
            dimensionless = values["dimensionless_extinction"][0, index]
            writer.writerow(
                [
                    value,
                    value * args.ell,
                    values["cumulative_hazard"][0, index],
                    values["transmittance"][0, index],
                    dimensionless,
                    dimensionless / args.ell,
                ]
            )
    if args.optical_depth:
        targets = _numbers(args.optical_depth)
        q_samples = invert_cumulative_hazard(coefficients, basis, targets)
        writer = csv.writer(sys.stdout)
        writer.writerow(["optical_depth", "q", "distance"])
        for target, q_sample in zip(targets, q_samples):
            writer.writerow([target, q_sample, q_sample * args.ell])


if __name__ == "__main__":
    main()
