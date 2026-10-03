from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from .model import CoefficientNet
from .splines import ISplineBasis


def load_checkpoint(
    path: str | Path, device: str | torch.device = "cpu"
) -> tuple[CoefficientNet, ISplineBasis, dict[str, Any]]:
    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    basis = ISplineBasis.from_dict(checkpoint["basis"])
    state = checkpoint["model_state"]
    model = CoefficientNet(
        input_mean=state["input_mean"].numpy(),
        input_std=state["input_std"].numpy(),
        hidden_sizes=checkpoint["hidden_sizes"],
        coefficient_count=int(checkpoint["coefficient_count"]),
    )
    model.load_state_dict(state)
    model.to(device).eval()
    return model, basis, checkpoint


def predict_coefficients(model: CoefficientNet, beta: np.ndarray) -> np.ndarray:
    device = model.input_mean.device
    with torch.no_grad():
        values = model(torch.as_tensor(beta, dtype=torch.float32, device=device))
    return values.detach().cpu().numpy()


def evaluate_curves(
    model: CoefficientNet,
    basis: ISplineBasis,
    beta: np.ndarray,
    q: np.ndarray,
) -> dict[str, np.ndarray]:
    coefficients = np.atleast_2d(predict_coefficients(model, np.atleast_2d(beta)))
    i_values = basis.i_spline(q)
    m_values = basis.m_spline(q)
    cumulative = coefficients @ i_values.T
    extinction = coefficients @ m_values.T
    return {
        "coefficients": coefficients,
        "cumulative_hazard": cumulative,
        "transmittance": np.exp(-cumulative),
        "dimensionless_extinction": extinction,
    }


def invert_cumulative_hazard(
    coefficients: np.ndarray,
    basis: ISplineBasis,
    optical_depth: np.ndarray | float,
    iterations: int = 60,
) -> np.ndarray:
    """Invert H(q); return +inf when optical depth exceeds H(q_max)."""

    coeff = np.asarray(coefficients, dtype=np.float64)
    if coeff.ndim != 1 or coeff.size != basis.basis_count:
        raise ValueError("coefficients must contain one complete basis coefficient vector")
    targets = np.asarray(optical_depth, dtype=np.float64)
    if np.any(targets < 0.0):
        raise ValueError("optical_depth must be non-negative")
    flat = targets.reshape(-1)
    result = np.full(flat.shape, np.inf, dtype=np.float64)
    h_max = float(basis.i_spline(basis.q_max) @ coeff)
    for index, target in enumerate(flat):
        if target > h_max:
            continue
        if target == 0.0:
            result[index] = 0.0
            continue
        low = 0.0
        high = basis.q_max
        for _ in range(iterations):
            middle = 0.5 * (low + high)
            if float(basis.i_spline(middle) @ coeff) < target:
                low = middle
            else:
                high = middle
        result[index] = 0.5 * (low + high)
    return result.reshape(targets.shape)
