from __future__ import annotations

import torch


def interval_delta_h(coefficients: torch.Tensor, delta_i: torch.Tensor) -> torch.Tensor:
    return coefficients @ delta_i.transpose(0, 1)


def per_state_interval_nll(
    coefficients: torch.Tensor,
    delta_i: torch.Tensor,
    at_risk: torch.Tensor,
    events: torch.Tensor,
) -> torch.Tensor:
    """Exact grouped interval likelihood, normalized by each state's initial count."""

    delta_h = interval_delta_h(coefficients, delta_i).clamp_min(0.0)
    event_probability = (-torch.expm1(-delta_h)).clamp_min(1.0e-12)
    event_term = -events * torch.log(event_probability)
    survivor_term = (at_risk - events) * delta_h
    return (event_term + survivor_term).sum(dim=-1) / at_risk[:, 0].clamp_min(1.0)


def interval_nll(
    coefficients: torch.Tensor,
    delta_i: torch.Tensor,
    at_risk: torch.Tensor,
    events: torch.Tensor,
) -> torch.Tensor:
    return per_state_interval_nll(coefficients, delta_i, at_risk, events).mean()


def cumulative_hazard(coefficients: torch.Tensor, i_values: torch.Tensor) -> torch.Tensor:
    return coefficients @ i_values.transpose(0, 1)


def transmittance(coefficients: torch.Tensor, i_values: torch.Tensor) -> torch.Tensor:
    return torch.exp(-cumulative_hazard(coefficients, i_values))
