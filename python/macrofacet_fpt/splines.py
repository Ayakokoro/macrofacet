from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.interpolate import BSpline


@dataclass(frozen=True)
class ISplineBasis:
    """Clamped I-splines obtained by integrating normalized M-splines."""

    knots: np.ndarray
    degree: int
    q_max: float
    knot_power: float

    @classmethod
    def create(
        cls,
        q_max: float,
        basis_count: int = 32,
        degree: int = 2,
        knot_power: float = 1.5,
    ) -> "ISplineBasis":
        if q_max <= 0.0:
            raise ValueError("q_max must be positive")
        if degree < 0:
            raise ValueError("degree must be non-negative")
        if basis_count <= degree + 1:
            raise ValueError("basis_count must exceed degree + 1")
        if knot_power <= 0.0:
            raise ValueError("knot_power must be positive")

        internal_count = basis_count - degree - 1
        fractions = np.arange(1, internal_count + 1, dtype=np.float64)
        fractions /= internal_count + 1
        internal = q_max * np.power(fractions, knot_power)
        knots = np.concatenate(
            (
                np.zeros(degree + 1, dtype=np.float64),
                internal,
                np.full(degree + 1, q_max, dtype=np.float64),
            )
        )
        return cls(knots=knots, degree=degree, q_max=float(q_max), knot_power=float(knot_power))

    @property
    def basis_count(self) -> int:
        return int(self.knots.size - self.degree - 1)

    def _m_spline_object(self) -> BSpline:
        widths = self.knots[self.degree + 1 :] - self.knots[: -self.degree - 1]
        if np.any(widths <= 0.0):
            raise ValueError("each M-spline support must have positive width")
        scales = (self.degree + 1.0) / widths
        coefficients = np.diag(scales)
        return BSpline(self.knots, coefficients, self.degree, extrapolate=False, axis=0)

    def m_spline(self, q: np.ndarray | float) -> np.ndarray:
        values = np.asarray(self._m_spline_object()(np.asarray(q, dtype=np.float64)))
        return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).clip(min=0.0)

    def i_spline(self, q: np.ndarray | float) -> np.ndarray:
        m_spline = self._m_spline_object()
        antiderivative = m_spline.antiderivative()
        q_array = np.asarray(q, dtype=np.float64)
        clipped = np.clip(q_array, 0.0, self.q_max)
        values = np.asarray(antiderivative(clipped) - antiderivative(0.0))
        values = np.nan_to_num(values, nan=0.0)
        values = np.clip(values, 0.0, 1.0)
        if q_array.ndim == 0:
            if q_array < 0.0:
                values[...] = 0.0
            elif q_array > self.q_max:
                values[...] = 1.0
        else:
            values[q_array < 0.0] = 0.0
            values[q_array > self.q_max] = 1.0
        return values

    def delta_i(self, q_edges: np.ndarray) -> np.ndarray:
        edges = np.asarray(q_edges, dtype=np.float64)
        if edges.ndim != 1 or edges.size < 2:
            raise ValueError("q_edges must be a one-dimensional array with at least two entries")
        if np.any(np.diff(edges) <= 0.0):
            raise ValueError("q_edges must be strictly increasing")
        values = self.i_spline(edges)
        return np.maximum(np.diff(values, axis=0), 0.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "degree": self.degree,
            "basis_count": self.basis_count,
            "q_max": self.q_max,
            "knot_power": self.knot_power,
            "knots": self.knots.tolist(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ISplineBasis":
        return cls(
            knots=np.asarray(value["knots"], dtype=np.float64),
            degree=int(value["degree"]),
            q_max=float(value["q_max"]),
            knot_power=float(value.get("knot_power", 1.0)),
        )
