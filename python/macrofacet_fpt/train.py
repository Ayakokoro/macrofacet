from __future__ import annotations

import argparse
import copy
import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .data import CurveDataset, load_curve_dataset, split_state_indices
from .export import export_model_bundle
from .loss import interval_nll, per_state_interval_nll
from .model import CoefficientNet, transformed_beta
from .splines import ISplineBasis


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _tensor(values: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(values, dtype=torch.float32, device=device)


def _split_metrics(
    model: CoefficientNet,
    data: CurveDataset,
    indices: np.ndarray,
    delta_i: torch.Tensor,
    i_end: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        beta = _tensor(data.beta[indices], device)
        risks = _tensor(data.at_risk[indices], device)
        events = _tensor(data.events[indices], device)
        coefficients = model(beta)
        state_nll = per_state_interval_nll(coefficients, delta_i, risks, events)
        predicted = torch.exp(-(coefficients @ i_end.transpose(0, 1)))
        empirical = _tensor(data.empirical_survival[indices], device)
        max_error = torch.max(torch.abs(predicted - empirical), dim=1).values
    errors = max_error.detach().cpu().numpy()
    return {
        "normalized_nll": float(state_nll.mean().cpu()),
        "max_transmittance_error_mean": float(np.mean(errors)),
        "max_transmittance_error_p95": float(np.quantile(errors, 0.95)),
        "max_transmittance_error_max": float(np.max(errors)),
    }


def _write_predictions(
    path: Path,
    model: CoefficientNet,
    basis: ISplineBasis,
    data: CurveDataset,
    splits: dict[str, np.ndarray],
    device: torch.device,
) -> None:
    i_end_np = basis.i_spline(data.q_edges[1:])
    m_end_np = basis.m_spline(data.q_edges[1:])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "split",
                "state_id",
                "beta_0",
                "beta_a",
                "beta_g",
                "bin",
                "q",
                "at_risk",
                "events",
                "empirical_survival",
                "predicted_cumulative_hazard",
                "predicted_survival",
                "predicted_dimensionless_extinction",
            ]
        )
        model.eval()
        with torch.no_grad():
            for split_name, indices in splits.items():
                coefficients = model(_tensor(data.beta[indices], device)).cpu().numpy()
                hazard = coefficients @ i_end_np.T
                extinction = coefficients @ m_end_np.T
                for local, state_index in enumerate(indices):
                    for interval in range(data.interval_count):
                        writer.writerow(
                            [
                                split_name,
                                data.state_ids[state_index],
                                *data.beta[state_index].tolist(),
                                interval,
                                data.q_edges[interval + 1],
                                data.at_risk[state_index, interval],
                                data.events[state_index, interval],
                                data.empirical_survival[state_index, interval],
                                hazard[local, interval],
                                np.exp(-hazard[local, interval]),
                                extinction[local, interval],
                            ]
                        )


def _write_coefficients(
    path: Path,
    model: CoefficientNet,
    data: CurveDataset,
    splits: dict[str, np.ndarray],
    device: torch.device,
) -> None:
    split_by_index = {
        int(index): name for name, indices in splits.items() for index in indices
    }
    model.eval()
    with torch.no_grad():
        coefficients = model(_tensor(data.beta, device)).cpu().numpy()
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["split", "state_id", "beta_0", "beta_a", "beta_g"]
            + [f"a_{index}" for index in range(coefficients.shape[1])]
        )
        for index, state_id in enumerate(data.state_ids):
            writer.writerow(
                [split_by_index[index], state_id, *data.beta[index].tolist(), *coefficients[index].tolist()]
            )


def train(config: dict[str, Any]) -> dict[str, Any]:
    seed = int(config["split"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    data = load_curve_dataset(config["data_csv"], config["kernel_id"])
    basis_config = config["basis"]
    basis = ISplineBasis.create(
        q_max=float(data.q_edges[-1]),
        basis_count=int(basis_config["count"]),
        degree=int(basis_config["degree"]),
        knot_power=float(basis_config["knot_power"]),
    )
    splits = split_state_indices(
        data.state_count,
        float(config["split"]["validation_fraction"]),
        float(config["split"]["test_fraction"]),
        seed,
    )

    transformed = transformed_beta(data.beta[splits["train"]])
    input_mean = transformed.mean(axis=0)
    input_std = transformed.std(axis=0)
    input_std[input_std < 1.0e-8] = 1.0

    model = CoefficientNet(
        input_mean=input_mean,
        input_std=input_std,
        hidden_sizes=config["model"]["hidden_sizes"],
        coefficient_count=basis.basis_count,
    )
    training = config["training"]
    device = _device(str(training["device"]))
    model.to(device)
    delta_i = _tensor(basis.delta_i(data.q_edges), device)
    i_end = _tensor(basis.i_spline(data.q_edges[1:]), device)

    train_indices = splits["train"]
    train_dataset = TensorDataset(
        torch.as_tensor(data.beta[train_indices], dtype=torch.float32),
        torch.as_tensor(data.at_risk[train_indices], dtype=torch.float32),
        torch.as_tensor(data.events[train_indices], dtype=torch.float32),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        train_dataset,
        batch_size=int(training["batch_size"]),
        shuffle=True,
        generator=generator,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )

    output = Path(config["output_directory"])
    output.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float | int]] = []
    best_validation = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    patience = int(training["patience"])
    minimum_improvement = float(training["minimum_improvement"])
    log_every = int(training.get("log_every", 25))

    validation_indices = splits["validation"]
    validation_beta = _tensor(data.beta[validation_indices], device)
    validation_risks = _tensor(data.at_risk[validation_indices], device)
    validation_events = _tensor(data.events[validation_indices], device)

    for epoch in range(1, int(training["epochs"]) + 1):
        model.train()
        weighted_loss = 0.0
        seen = 0
        for beta_batch, risk_batch, event_batch in loader:
            beta_batch = beta_batch.to(device)
            risk_batch = risk_batch.to(device)
            event_batch = event_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = interval_nll(model(beta_batch), delta_i, risk_batch, event_batch)
            loss.backward()
            optimizer.step()
            weighted_loss += float(loss.detach().cpu()) * beta_batch.shape[0]
            seen += beta_batch.shape[0]

        model.eval()
        with torch.no_grad():
            validation_loss = float(
                interval_nll(
                    model(validation_beta), delta_i, validation_risks, validation_events
                ).cpu()
            )
        train_loss = weighted_loss / seen
        history.append({"epoch": epoch, "train_nll": train_loss, "validation_nll": validation_loss})
        if log_every > 0 and (epoch == 1 or epoch % log_every == 0):
            print(f"epoch={epoch} train_nll={train_loss:.8g} validation_nll={validation_loss:.8g}")

        if validation_loss < best_validation - minimum_improvement:
            best_validation = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is None:
        raise RuntimeError("training produced no finite validation checkpoint")
    model.load_state_dict(best_state)

    metrics = {
        name: _split_metrics(model, data, indices, delta_i, i_end, device)
        for name, indices in splits.items()
    }
    result: dict[str, Any] = {
        "kernel_id": data.kernel_id,
        "kernel_type": data.kernel_type,
        "device": str(device),
        "state_count": data.state_count,
        "interval_count": data.interval_count,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "best_validation_nll": best_validation,
        "metrics": metrics,
    }

    with (output / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["epoch", "train_nll", "validation_nll"])
        writer.writeheader()
        writer.writerows(history)
    with (output / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")

    checkpoint = {
        "format_version": 1,
        "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "hidden_sizes": list(model.hidden_sizes),
        "coefficient_count": model.coefficient_count,
        "basis": basis.to_dict(),
        "kernel_id": data.kernel_id,
        "kernel_type": data.kernel_type,
        "q_edges": data.q_edges,
        "config": config,
        "splits": {name: [data.state_ids[index] for index in indices] for name, indices in splits.items()},
    }
    torch.save(checkpoint, output / "checkpoint.pt")
    _write_predictions(output / "predictions.csv", model, basis, data, splits, device)
    _write_coefficients(output / "coefficients.csv", model, data, splits, device)

    beta_min = data.beta.min(axis=0)
    beta_max = data.beta.max(axis=0)
    metadata = {
        "kernel_id": data.kernel_id,
        "kernel_type": data.kernel_type,
        "q_max": basis.q_max,
        "valid_range": {
            "beta_0": [float(beta_min[0]), float(beta_max[0])],
            "beta_a": [float(beta_min[1]), float(beta_max[1])],
            "beta_g": [float(beta_min[2]), float(beta_max[2])],
        },
        "training": {
            "data_csv": str(config["data_csv"]),
            "state_count": data.state_count,
            "base_step": data.base_step,
            "best_epoch": best_epoch,
            "metrics": metrics,
            "state_ids": checkpoint["splits"],
        },
    }
    export_model_bundle(output, model, basis, metadata)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the monotone first-passage I-spline surrogate")
    parser.add_argument("--config", required=True, help="training JSON file")
    parser.add_argument("--data", help="override data_csv")
    parser.add_argument("--output", help="override output_directory")
    parser.add_argument("--epochs", type=int, help="override training.epochs")
    parser.add_argument("--device", help="override training.device")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    with Path(args.config).open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    if args.data:
        config["data_csv"] = args.data
    if args.output:
        config["output_directory"] = args.output
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.device:
        config["training"]["device"] = args.device
    result = train(config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
