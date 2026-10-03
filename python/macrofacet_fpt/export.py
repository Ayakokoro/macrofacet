from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .model import CoefficientNet
from .splines import ISplineBasis


def _linear_layers(model: CoefficientNet) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for layer in model.network:
        if isinstance(layer, nn.Linear):
            result.append(
                {
                    "weight": layer.weight.detach().cpu().numpy().tolist(),
                    "bias": layer.bias.detach().cpu().numpy().tolist(),
                }
            )
    return result


def export_model_bundle(
    output_directory: str | Path,
    model: CoefficientNet,
    basis: ISplineBasis,
    metadata: dict[str, Any],
) -> None:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    cpu_model = model.to("cpu").eval()
    scripted = torch.jit.script(cpu_model)
    scripted.save(str(output / "coefficient_net.pt"))
    np.savez(
        output / "ispline_basis.npz",
        knots=basis.knots,
        degree=np.asarray(basis.degree, dtype=np.int64),
        q_max=np.asarray(basis.q_max, dtype=np.float64),
        knot_power=np.asarray(basis.knot_power, dtype=np.float64),
    )

    bundle = {
        "format": "macrofacet.fpt-ispline",
        "format_version": 1,
        "inputs": ["beta_0", "beta_a", "beta_g"],
        "input_transform": ["identity", "identity", "log"],
        "input_mean": cpu_model.input_mean.detach().numpy().tolist(),
        "input_std": cpu_model.input_std.detach().numpy().tolist(),
        "network": {
            "hidden_sizes": list(cpu_model.hidden_sizes),
            "output_coefficients": cpu_model.coefficient_count,
            "hidden_activation": "silu",
            "output_activation": "softplus",
            "coefficient_0": 0.0,
            "linear_layers": _linear_layers(cpu_model),
        },
        "basis": basis.to_dict(),
        "outputs": {
            "cumulative_hazard": "H(q) = sum_i a_i I_i(q)",
            "transmittance": "T(q) = exp(-H(q))",
            "dimensionless_extinction": "dH/dq = sum_i a_i M_i(q)",
        },
        **metadata,
    }
    with (output / "model_bundle.json").open("w", encoding="utf-8") as stream:
        json.dump(bundle, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
