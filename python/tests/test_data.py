import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from macrofacet_fpt.data import load_curve_dataset, split_state_indices


class CurveDatasetTest(unittest.TestCase):
    def test_loads_complete_training_curves(self) -> None:
        fieldnames = [
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
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "curves.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fieldnames)
                writer.writeheader()
                for state in range(4):
                    risks = [10, 9, 7]
                    events = [1, 2, state % 3]
                    for interval in range(3):
                        writer.writerow(
                            {
                                "kernel_id": "matern52",
                                "kernel_type": "matern_5_2",
                                "state_id": f"lhs_{state}",
                                "beta_0": state * 0.1,
                                "beta_a": state * -0.2,
                                "beta_g": 0.5 + state,
                                "base_step": 0.01,
                                "training_resolution": 1,
                                "bin": interval,
                                "q_begin": interval,
                                "q_end": interval + 1,
                                "at_risk": risks[interval],
                                "events": events[interval],
                            }
                        )
            data = load_curve_dataset(path, "matern52")
        self.assertEqual(data.state_count, 4)
        self.assertEqual(data.interval_count, 3)
        np.testing.assert_allclose(data.q_edges, [0.0, 1.0, 2.0, 3.0])
        np.testing.assert_allclose(data.empirical_survival[0], [0.9, 0.7, 0.7])

    def test_split_is_disjoint_and_reproducible(self) -> None:
        first = split_state_indices(20, 0.2, 0.2, 7)
        second = split_state_indices(20, 0.2, 0.2, 7)
        for name in first:
            np.testing.assert_array_equal(first[name], second[name])
        all_indices = np.concatenate(tuple(first.values()))
        np.testing.assert_array_equal(np.sort(all_indices), np.arange(20))


if __name__ == "__main__":
    unittest.main()
