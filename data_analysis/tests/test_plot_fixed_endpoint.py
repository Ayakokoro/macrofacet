from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from data_analysis.plot_fixed_endpoint import run


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class FixedEndpointPlotTests(unittest.TestCase):
    def _fixture(self, directory: Path) -> tuple[Path, dict[str, object]]:
        write_csv(directory / "fixed_endpoint_summary.csv", [{
            "kernel_id": "se", "state_id": "a", "target_distance": 0.004,
            "target_q": 2.0, "training_resolution": 1, "sum_weight_q": 4.0,
        }])
        rows = []
        for slope, weight, survived, resolution in (
            (-1.0, 1.0, 1, 1), (-3.0, 3.0, 1, 1),
            (-20.0, 0.0, 0, 1), (10.0, 0.0, 0, 1), (-50.0, 50.0, 1, 0),
        ):
            rows.append({"kernel_id": "se", "state_id": "a", "target_distance": 0.004,
                "training_resolution": resolution, "survived_to_endpoint": survived,
                "endpoint_slope": slope, "endpoint_derivative": 0.5 * slope,
                "flux_weight_q": weight})
        write_csv(directory / "fixed_endpoint_samples.csv", rows)
        config: dict[str, object] = {
            "schema_version": 1, "input_directory": str(directory),
            "comparison": "states", "kernel_id": "se", "state_ids": ["a"],
            "distances": [0.004], "bins": 4, "slope_units": "dimensionless", "slope_max": 4.0,
        }
        return directory / "analysis.json", config

    def test_flux_weighted_density_and_effective_sample_size(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path, config = self._fixture(directory)
            path.write_text(json.dumps(config), encoding="utf-8")
            generated = run(path)
            self.assertEqual(len(generated), 2)
            with (directory / "fixed_endpoint_slope_density.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            densities = [float(row["probability_density"]) for row in rows]
            self.assertEqual(densities, [0.0, 0.25, 0.0, 0.75])
            self.assertEqual(sum(int(row["count"]) for row in rows), 2)
            self.assertAlmostEqual(float(rows[0]["effective_samples"]), 1.6)
            self.assertAlmostEqual(sum(float(row["probability_density"]) *
                (float(row["slope_end"]) - float(row["slope_begin"])) for row in rows), 1.0)
            svg = (directory / "fixed_endpoint_slope_density_d0.svg").read_text(encoding="utf-8")
            self.assertIn("First-hit slope density at distance 0.004", svg)
            self.assertIn("negative-slope flux weights", svg)
            # Units change only the slope coordinate/PDF Jacobian, not weights.
            config["slope_units"], config["slope_max"] = "physical", 2.0
            path.write_text(json.dumps(config), encoding="utf-8")
            run(path)
            with (directory / "fixed_endpoint_slope_density.csv").open(newline="") as stream:
                physical = list(csv.DictReader(stream))
            self.assertEqual([float(row["probability_density"]) for row in physical], [0.0, 0.5, 0.0, 1.5])

    def test_range_truncation_and_inconsistent_runs_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path, config = self._fixture(directory)
            config["slope_max"] = 2.0
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "truncates"):
                run(path)
            config["slope_max"] = 4.0
            path.write_text(json.dumps(config), encoding="utf-8")
            write_csv(directory / "fixed_endpoint_summary.csv", [{
                "kernel_id": "se", "state_id": "a", "target_distance": 0.004,
                "target_q": 2.0, "training_resolution": 1, "sum_weight_q": 0.0,
            }])
            with self.assertRaisesRegex(ValueError, "different runs"):
                run(path)
            rows = [{"kernel_id": "se", "state_id": "a", "target_distance": 0.004,
                "training_resolution": 1, "survived_to_endpoint": 0,
                "endpoint_slope": -1.0, "endpoint_derivative": -0.5, "flux_weight_q": 0.0}]
            write_csv(directory / "fixed_endpoint_samples.csv", rows)
            with self.assertRaisesRegex(ValueError, "no positive-weight"):
                run(path)


if __name__ == "__main__":
    unittest.main()
