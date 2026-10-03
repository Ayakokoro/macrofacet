from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class CurveDataset:
    state_ids: tuple[str, ...]
    beta: np.ndarray
    q_edges: np.ndarray
    at_risk: np.ndarray
    events: np.ndarray
    empirical_survival: np.ndarray
    kernel_id: str
    kernel_type: str
    base_step: float

    @property
    def state_count(self) -> int:
        return len(self.state_ids)

    @property
    def interval_count(self) -> int:
        return int(self.events.shape[1])


def _require_columns(fieldnames: list[str] | None, required: set[str]) -> None:
    missing = required.difference(fieldnames or [])
    if missing:
        raise ValueError(f"curve CSV is missing columns: {', '.join(sorted(missing))}")


def load_curve_dataset(path: str | Path, kernel_id: str) -> CurveDataset:
    """Load the finest-resolution interval counts, grouping complete curves by state."""

    required = {
        "kernel_id",
        "kernel_type",
        "state_id",
        "beta_0",
        "beta_a",
        "beta_g",
        "base_step",
        "training_resolution",
        "bin",
        "q_begin",
        "q_end",
        "at_risk",
        "events",
    }
    grouped: dict[str, list[dict[str, str]]] = {}
    csv_path = Path(path)
    with csv_path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        _require_columns(reader.fieldnames, required)
        for row in reader:
            if row["kernel_id"] != kernel_id or int(row["training_resolution"]) != 1:
                continue
            grouped.setdefault(row["state_id"], []).append(row)

    if not grouped:
        raise ValueError(
            f"no training-resolution rows for kernel_id={kernel_id!r} in {csv_path}"
        )

    state_ids = tuple(sorted(grouped, key=lambda value: (value.rstrip("0123456789"), _numeric_suffix(value))))
    beta_rows: list[list[float]] = []
    at_risk_rows: list[list[int]] = []
    event_rows: list[list[int]] = []
    reference_edges: np.ndarray | None = None
    kernel_types: set[str] = set()
    base_steps: set[float] = set()

    for state_id in state_ids:
        rows = sorted(grouped[state_id], key=lambda row: int(row["bin"]))
        bins = np.asarray([int(row["bin"]) for row in rows], dtype=np.int64)
        if not np.array_equal(bins, np.arange(len(rows), dtype=np.int64)):
            raise ValueError(f"state {state_id!r} has missing or duplicated bins")

        begins = np.asarray([float(row["q_begin"]) for row in rows], dtype=np.float64)
        ends = np.asarray([float(row["q_end"]) for row in rows], dtype=np.float64)
        if not np.allclose(begins[1:], ends[:-1], rtol=1e-10, atol=1e-12):
            raise ValueError(f"state {state_id!r} has non-contiguous q intervals")
        edges = np.concatenate((begins[:1], ends))
        if reference_edges is None:
            reference_edges = edges
        elif not np.allclose(edges, reference_edges, rtol=1e-10, atol=1e-12):
            raise ValueError("all states must share the same q interval edges")

        beta = [float(rows[0][name]) for name in ("beta_0", "beta_a", "beta_g")]
        if beta[2] <= 0.0:
            raise ValueError(f"state {state_id!r} has non-positive beta_g")
        for row in rows[1:]:
            other = np.asarray([float(row[name]) for name in ("beta_0", "beta_a", "beta_g")])
            if not np.allclose(other, beta, rtol=1e-12, atol=1e-14):
                raise ValueError(f"state {state_id!r} changes beta parameters across bins")

        risks = np.asarray([int(row["at_risk"]) for row in rows], dtype=np.int64)
        events = np.asarray([int(row["events"]) for row in rows], dtype=np.int64)
        if risks[0] <= 0 or np.any(events < 0) or np.any(events > risks):
            raise ValueError(f"state {state_id!r} has invalid interval counts")
        if np.any(risks[1:] != risks[:-1] - events[:-1]):
            raise ValueError(
                f"state {state_id!r} loses trajectories before q_max; only horizon censoring is supported"
            )

        beta_rows.append(beta)
        at_risk_rows.append(risks.tolist())
        event_rows.append(events.tolist())
        kernel_types.update(row["kernel_type"] for row in rows)
        base_steps.update(float(row["base_step"]) for row in rows)

    if len(kernel_types) != 1 or len(base_steps) != 1:
        raise ValueError("selected training rows must use one kernel type and one base step")

    at_risk_array = np.asarray(at_risk_rows, dtype=np.int64)
    events_array = np.asarray(event_rows, dtype=np.int64)
    conditional = np.ones_like(events_array, dtype=np.float64)
    valid = at_risk_array > 0
    conditional[valid] -= events_array[valid] / at_risk_array[valid]
    empirical_survival = np.cumprod(conditional, axis=1)
    assert reference_edges is not None
    return CurveDataset(
        state_ids=state_ids,
        beta=np.asarray(beta_rows, dtype=np.float64),
        q_edges=reference_edges,
        at_risk=at_risk_array,
        events=events_array,
        empirical_survival=empirical_survival,
        kernel_id=kernel_id,
        kernel_type=next(iter(kernel_types)),
        base_step=next(iter(base_steps)),
    )


def _numeric_suffix(value: str) -> int:
    suffix = ""
    for character in reversed(value):
        if not character.isdigit():
            break
        suffix = character + suffix
    return int(suffix) if suffix else -1


def split_state_indices(
    state_count: int,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> dict[str, np.ndarray]:
    if state_count < 3:
        raise ValueError("at least three states are required for train/validation/test splitting")
    if validation_fraction <= 0.0 or test_fraction <= 0.0:
        raise ValueError("validation_fraction and test_fraction must be positive")
    if validation_fraction + test_fraction >= 1.0:
        raise ValueError("validation_fraction + test_fraction must be less than one")

    permutation = np.random.default_rng(seed).permutation(state_count)
    validation_count = max(1, int(round(state_count * validation_fraction)))
    test_count = max(1, int(round(state_count * test_fraction)))
    if validation_count + test_count >= state_count:
        validation_count = 1
        test_count = 1
    train_count = state_count - validation_count - test_count
    return {
        "train": np.sort(permutation[:train_count]),
        "validation": np.sort(permutation[train_count : train_count + validation_count]),
        "test": np.sort(permutation[train_count + validation_count :]),
    }
