from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def transformed_beta(beta: np.ndarray) -> np.ndarray:
    values = np.asarray(beta, dtype=np.float64)
    if values.shape[-1] != 3:
        raise ValueError("beta must have final dimension 3")
    if np.any(values[..., 2] <= 0.0):
        raise ValueError("beta_g must be positive")
    result = values.copy()
    result[..., 2] = np.log(result[..., 2])
    return result


class CoefficientNet(nn.Module):
    """Map (beta_0, beta_a, beta_g) to non-negative I-spline coefficients."""

    def __init__(
        self,
        input_mean: np.ndarray | Sequence[float],
        input_std: np.ndarray | Sequence[float],
        hidden_sizes: Sequence[int] = (32, 32),
        coefficient_count: int = 32,
    ) -> None:
        super().__init__()
        if coefficient_count < 2:
            raise ValueError("coefficient_count must be at least two")
        if not hidden_sizes or any(size <= 0 for size in hidden_sizes):
            raise ValueError("hidden_sizes must contain positive values")
        mean = torch.as_tensor(input_mean, dtype=torch.float32)
        std = torch.as_tensor(input_std, dtype=torch.float32)
        if mean.shape != (3,) or std.shape != (3,) or torch.any(std <= 0.0):
            raise ValueError("input_mean and positive input_std must each have shape (3,)")
        self.register_buffer("input_mean", mean)
        self.register_buffer("input_std", std)
        self.hidden_sizes = tuple(int(value) for value in hidden_sizes)
        self.coefficient_count = int(coefficient_count)

        sizes = (3, *self.hidden_sizes, coefficient_count - 1)
        layers: list[nn.Module] = []
        for index in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[index], sizes[index + 1]))
            if index + 1 < len(sizes) - 1:
                layers.append(nn.SiLU())
        self.network = nn.Sequential(*layers)

    def normalized_input(self, beta: torch.Tensor) -> torch.Tensor:
        beta_log = torch.stack((beta[..., 0], beta[..., 1], torch.log(beta[..., 2])), dim=-1)
        return (beta_log - self.input_mean) / self.input_std

    def forward(self, beta: torch.Tensor) -> torch.Tensor:
        raw = self.network(self.normalized_input(beta))
        positive = F.softplus(raw)
        zero = torch.zeros_like(positive[..., :1])
        return torch.cat((zero, positive), dim=-1)
