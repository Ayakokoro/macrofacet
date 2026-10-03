from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from .data import load_curve_dataset
from .splines import ISplineBasis


def fit_state_coefficients(
    delta_i: np.ndarray,
    at_risk: np.ndarray,
    events: np.ndarray,
) -> tuple[np.ndarray, float, bool]:
    """Maximum-likelihood non-negative spline fit for one state, with a_0 fixed to zero."""

    design = np.asarray(delta_i[:, 1:], dtype=np.float64)
    risks = np.asarray(at_risk, dtype=np.float64)
    event_counts = np.asarray(events, dtype=np.float64)

    def objective(values: np.ndarray) -> tuple[float, np.ndarray]:
        delta_h = np.maximum(design @ values, 1.0e-12)
        probability = -np.expm1(-delta_h)
        loss = np.sum(-event_counts * np.log(probability) + (risks - event_counts) * delta_h)
        gradient_h = (risks - event_counts) - event_counts / np.expm1(delta_h)
        gradient = design.T @ gradient_h
        return float(loss / risks[0]), gradient / risks[0]

    initial = np.full(design.shape[1], 0.05, dtype=np.float64)
    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        bounds=[(0.0, None)] * design.shape[1],
        options={"maxiter": 1000, "ftol": 1.0e-12, "gtol": 1.0e-8},
    )
    coefficients = np.concatenate(([0.0], np.maximum(result.x, 0.0)))
    return coefficients, float(result.fun), bool(result.success)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit one non-negative I-spline independently per state")
    parser.add_argument("--data", required=True)
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--basis-count", type=int, default=32)
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--knot-power", type=float, default=1.5)
    parser.add_argument("--max-states", type=int)
    args = parser.parse_args()

    data = load_curve_dataset(args.data, args.kernel)
    basis = ISplineBasis.create(
        q_max=float(data.q_edges[-1]),
        basis_count=args.basis_count,
        degree=args.degree,
        knot_power=args.knot_power,
    )
    delta_i = basis.delta_i(data.q_edges)
    i_end = basis.i_spline(data.q_edges[1:])
    state_count = data.state_count if args.max_states is None else min(data.state_count, args.max_states)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    rows: list[list[object]] = []
    metric_rows: list[dict[str, object]] = []
    for index in range(state_count):
        coefficients, normalized_nll, converged = fit_state_coefficients(
            delta_i, data.at_risk[index], data.events[index]
        )
        predicted = np.exp(-(i_end @ coefficients))
        maximum_error = float(np.max(np.abs(predicted - data.empirical_survival[index])))
        rows.append(
            [
                data.state_ids[index],
                *data.beta[index].tolist(),
                normalized_nll,
                int(converged),
                maximum_error,
                *coefficients.tolist(),
            ]
        )
        metric_rows.append(
            {
                "state_id": data.state_ids[index],
                "normalized_nll": normalized_nll,
                "converged": converged,
                "max_transmittance_error": maximum_error,
            }
        )

    with (output / "independent_coefficients.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "state_id",
                "beta_0",
                "beta_a",
                "beta_g",
                "normalized_nll",
                "converged",
                "max_transmittance_error",
            ]
            + [f"a_{index}" for index in range(basis.basis_count)]
        )
        writer.writerows(rows)

    errors = np.asarray([row["max_transmittance_error"] for row in metric_rows], dtype=float)
    summary = {
        "kernel_id": data.kernel_id,
        "state_count": state_count,
        "basis": basis.to_dict(),
        "converged_states": sum(bool(row["converged"]) for row in metric_rows),
        "max_transmittance_error_mean": float(np.mean(errors)),
        "max_transmittance_error_p95": float(np.quantile(errors, 0.95)),
        "max_transmittance_error_max": float(np.max(errors)),
        "states": metric_rows,
    }
    with (output / "independent_fit_metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "states"}, indent=2))


if __name__ == "__main__":
    main()
