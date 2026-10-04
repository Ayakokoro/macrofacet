from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from macrofacet_fpt.splines import ISplineBasis  # noqa: E402


class FixtureCoefficientNet(nn.Module):
    def forward(self, beta: torch.Tensor) -> torch.Tensor:
        zero = torch.zeros_like(beta[..., 0])
        return torch.stack(
            (zero, beta[..., 0] + 2.0, beta[..., 1] + 3.0, beta[..., 2] + 3.0),
            dim=-1,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    model = torch.jit.script(FixtureCoefficientNet().eval())
    model.save(str(output / "coefficient_net.pt"))
    basis = ISplineBasis.create(q_max=2.0, basis_count=4, degree=2, knot_power=1.5)
    bundle = {
        "format": "macrofacet.fpt-ispline",
        "format_version": 1,
        "kernel_id": "fixture",
        "kernel_type": "fixture_kernel_without_cpp_dispatch",
        "valid_range": {
            "beta_0": [-5.0, 5.0],
            "beta_a": [-5.0, 5.0],
            "beta_g": [0.1, 10.0],
        },
        "basis": basis.to_dict(),
    }
    with (output / "model_bundle.json").open("w", encoding="utf-8") as stream:
        json.dump(bundle, stream, indent=2)
        stream.write("\n")

    beta = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
    coefficients = np.asarray([0.0, 3.0, 5.0, 6.0], dtype=np.float64)
    q = np.asarray([0.0, 0.25, 1.0, 2.0], dtype=np.float64)
    i_values = basis.i_spline(q)
    m_values = basis.m_spline(q)
    golden = {
        "beta": beta.tolist(),
        "coefficients": coefficients.tolist(),
        "q": q.tolist(),
        "i_spline": i_values.tolist(),
        "m_spline": m_values.tolist(),
        "cumulative_hazard": (i_values @ coefficients).tolist(),
        "dimensionless_extinction": (m_values @ coefficients).tolist(),
    }
    with (output / "golden.json").open("w", encoding="utf-8") as stream:
        json.dump(golden, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
