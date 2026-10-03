"""Training tools for the collision-state first-passage surrogate."""

from .data import CurveDataset, load_curve_dataset, split_state_indices
from .model import CoefficientNet
from .splines import ISplineBasis

__all__ = [
    "CoefficientNet",
    "CurveDataset",
    "ISplineBasis",
    "load_curve_dataset",
    "split_state_indices",
]
