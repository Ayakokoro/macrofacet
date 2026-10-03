from __future__ import annotations

import argparse
import json

import numpy as np

from .data import load_curve_dataset
from .inference import evaluate_curves, load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on a curve CSV")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--kernel")
    args = parser.parse_args()

    model, basis, checkpoint = load_checkpoint(args.checkpoint)
    kernel = args.kernel or checkpoint["kernel_id"]
    data = load_curve_dataset(args.data, kernel)
    values = evaluate_curves(model, basis, data.beta, data.q_edges[1:])
    error = np.max(np.abs(values["transmittance"] - data.empirical_survival), axis=1)
    metrics = {
        "kernel_id": kernel,
        "state_count": data.state_count,
        "max_transmittance_error_mean": float(np.mean(error)),
        "max_transmittance_error_p95": float(np.quantile(error, 0.95)),
        "max_transmittance_error_max": float(np.max(error)),
    }
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
