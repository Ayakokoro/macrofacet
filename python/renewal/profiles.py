"""Deterministic input profiles and exact restrictions of their mean polynomials."""
from __future__ import annotations

import torch

from .model import mean_at


def validate_profile_mode(mode):
    if mode not in ("cubic", "point_linear"):
        raise ValueError("profile_mode must be cubic or point_linear")
    return mode


def with_profile_mode(data: dict, mode: str) -> dict:
    """Use starting tangents on the stored partition; keep the GP reference labels.

    This does not re-query the field or reproduce a different renderer partition.
    Adjacent tangents may be discontinuous, just as point-linear field queries are.
    The first value and one-sided derivative (and therefore A/B initials) agree
    with the original profile. Never modify the shared source dataset in place.
    """
    validate_profile_mode(mode)
    stored_mode = validate_profile_mode(data.get("profile_mode", "cubic"))
    if stored_mode == mode:
        return data
    if stored_mode != "cubic":
        raise ValueError("cannot recover cubic features from a point_linear dataset")
    features = data["features"].clone()
    features[:, 1] = features[:, 0] + features[:, 2]
    features[:, 3] = features[:, 2]
    return {**data, "features": features, "profile_mode": mode}


def validate_subdivision_factors(factors):
    if (not isinstance(factors, (list, tuple)) or not factors or
            any(type(n) is not int or n < 1 for n in factors) or
            len(set(factors)) != len(factors)):
        raise ValueError("subdivision_factors must be a nonempty list of distinct positive integers")
    return tuple(factors)


def subdivide_features(features: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """Split each stored segment into counts[i] equal pieces of the SAME cubic.

    Point-linear conversion must happen before this function: new subsegment
    starts are evaluations of that tangent, not new observations of the field.
    Segment boundaries, including mean/derivative jumps, are retained.
    """
    if (features.ndim != 2 or features.shape[1] != 5 or counts.ndim != 1 or
            len(counts) != len(features) or counts.dtype != torch.long or
            counts.device != features.device or (counts < 1).any()):
        raise ValueError("subdivision requires five-feature segments and positive integer counts")
    if (counts == 1).all():
        return features
    parent = torch.repeat_interleave(torch.arange(len(features), device=features.device), counts)
    local = torch.arange(len(parent), device=features.device) - torch.repeat_interleave(
        counts.cumsum(0) - counts, counts)
    source = features[parent]
    n = counts[parent].to(features.dtype)
    u0, u1 = local.to(features.dtype) / n, (local + 1).to(features.dtype) / n
    b0, db0 = mean_at(source, u0)
    b1, db1 = mean_at(source, u1)
    log_dx = source[:, 4] - n.log()
    dx = log_dx.exp()
    result = torch.stack((b0, b1, dx * db0, dx * db1, log_dx), dim=-1)
    # Keep unsplit segments bit-identical, including their one-sided derivatives.
    return torch.where((n == 1)[:, None], source, result)
