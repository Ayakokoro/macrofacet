from __future__ import annotations

import csv
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from data_analysis.plot_first_passage import run, _svg_line_chart


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class PlotFirstPassageTests(unittest.TestCase):
    def test_physical_distance_and_full_mean_profile(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            write_csv(directory / "first_passage_curves.csv", [{
                "kernel_id": "se", "state_id": "curved", "training_resolution": 1,
                "ell": 0.002, "q_end": 2.0, "at_risk": 100, "survival": 0.8,
                "hazard_mc": 0.25, "cumulative_hazard_product_limit": 0.2,
                "mean_profile_type": "full_field",
            }])
            write_csv(directory / "first_passage_mean_profiles.csv", [
                {"kernel_id": "se", "state_id": "curved", "q": 0.0,
                 "distance": 0.0, "mean": 0.0, "beta_mean": 0.0},
                {"kernel_id": "se", "state_id": "curved", "q": 2.0,
                 "distance": 0.004, "mean": 0.001, "beta_mean": 1.0},
            ])
            config = {"schema_version": 1, "input_directory": str(directory),
                "comparison": "states", "kernel_id": "se", "state_ids": ["curved"],
                "x_axis": "distance", "plots": {"mean_profile": True}}
            config_path = directory / "analysis.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with patch("data_analysis.plot_first_passage._svg_line_chart", wraps=_svg_line_chart) as charts:
                generated = run(config_path)
            hazard_chart = next(call.kwargs for call in charts.call_args_list
                                if call.kwargs["title"] == "First-passage hazard")
            self.assertEqual(hazard_chart["series"][0]["points"], [(0.004, 125.0)])
            self.assertEqual(len(generated), 4)
            hazard = (directory / "first_passage_hazard.svg").read_text(encoding="utf-8")
            self.assertIn("Distance t (scene units)", hazard)
            self.assertIn("1 / scene unit", hazard)
            profile = (directory / "first_passage_mean_profile.svg").read_text(encoding="utf-8")
            self.assertIn("SDF mean along the ray", profile)
            self.assertIn("curved", profile)

    def test_collision_plots_and_normalized_slope_density(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            curves = []
            for kernel_index, (kernel_id, kernel_type) in enumerate(
                (("se", "squared_exponential"), ("m52", "matern_5_2"))
            ):
                for bin_index in range(2):
                    curves.append(
                        {
                            "kernel_id": kernel_id,
                            "kernel_type": kernel_type,
                            "state_id": "state_a",
                            "beta_0": 0.1,
                            "beta_a": -0.2,
                            "beta_g": 1.3,
                            "training_resolution": 1,
                            "q_end": 0.5 * (bin_index + 1),
                            "at_risk": 20 - 3 * bin_index,
                            "survival": 1.0 - 0.2 * (bin_index + 1) - 0.05 * kernel_index,
                            "first_passage_density": 0.4,
                            "hazard_mc": 0.2 + 0.1 * bin_index,
                            "cumulative_hazard_product_limit": 0.1 + 0.2 * bin_index,
                        }
                    )
            write_csv(directory / "first_passage_curves.csv", curves)

            samples = []
            for kernel_id, kernel_type in (
                ("se", "squared_exponential"),
                ("m52", "matern_5_2"),
            ):
                for q, slope in ((0.2, 0.5), (0.4, 1.0), (0.7, 1.5)):
                    samples.append(
                        {
                            "kernel_id": kernel_id,
                            "kernel_type": kernel_type,
                            "state_id": "state_a",
                            "training_resolution": 1,
                            "event": 1,
                            "event_q": q,
                            "crossing_slope": slope,
                        }
                    )
            write_csv(directory / "first_passage_samples.csv", samples)
            config = {
                "schema_version": 1,
                "input_directory": str(directory),
                "output_directory": str(directory),
                "comparison": "kernels",
                "state_ids": ["state_a"],
                "minimum_risk_set": 2,
                "plots": {
                    "survival": True,
                    "hazard": True,
                    "cumulative_hazard": True,
                    "crossing_slope_density": {
                        "enabled": True,
                        "bins": 4,
                        "q_intervals": [[0.0, 0.5], [0.5, 1.0]],
                    },
                },
            }
            config_path = directory / "analysis.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            generated = run(config_path)

            self.assertIn(directory / "first_passage_survival.svg", generated)
            self.assertTrue((directory / "first_passage_hazard.svg").is_file())
            self.assertTrue((directory / "first_passage_cumulative_hazard.svg").is_file())
            self.assertTrue(
                (directory / "first_passage_crossing_slope_density_q1.svg").is_file()
            )
            with (directory / "first_passage_crossing_slope_density.csv").open(
                "r", encoding="utf-8", newline=""
            ) as stream:
                rows = list(csv.DictReader(stream))
            for kernel_id in ("se", "m52"):
                for interval in ("0", "1"):
                    selected = [
                        row
                        for row in rows
                        if row["kernel_id"] == kernel_id and row["q_interval"] == interval
                    ]
                    integral = sum(
                        float(row["probability_density"])
                        * (float(row["slope_end"]) - float(row["slope_begin"]))
                        for row in selected
                    )
                    self.assertAlmostEqual(integral, 1.0)

    def test_fixed_value_plots(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            rows = []
            for kernel_id in ("se", "m52"):
                for step in (0.2, 0.1):
                    for index in range(2):
                        rows.append(
                            {
                                "kernel_id": kernel_id,
                                "kernel_type": "matern_5_2",
                                "grid_step": step,
                                "bin_end": 0.5 * (index + 1),
                                "time": 0.5 * (index + 1),
                                "time_over_length_scale": 0.5 * (index + 1),
                                "at_risk": 20,
                                "survival": 0.9 - 0.2 * index,
                                "first_passage_density": 0.2 + 0.1 * index,
                                "hazard_mc": 0.1 + 0.1 * index,
                                "start_conditioned_sigma1_survival": 0.92 - 0.18 * index,
                                "start_conditioned_endpoint_survival": 0.95 - 0.1 * index,
                                "endpoint_conditioned_hazard_sigma1": 0.12 + 0.1 * index,
                                "rice_w1": 0.15 + 0.1 * index,
                                "rice_density_order2": 0.14 + 0.08 * index,
                            }
                        )
            write_csv(directory / "first_passage_curves.csv", rows)
            config_path = directory / "analysis.json"
            config_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "input_directory": str(directory),
                        "plots": {
                            "survival": True,
                            "hazard": True,
                            "rice_density": True,
                        },
                    }
                ),
                encoding="utf-8",
            )
            generated = run(config_path)
            self.assertEqual(len(generated), 3)
            for name in (
                "first_passage_survival.svg",
                "first_passage_hazard.svg",
                "first_passage_rice_density.svg",
            ):
                contents = (directory / name).read_text(encoding="utf-8")
                self.assertIn("<svg", contents)
                self.assertIn("se", contents)
                self.assertIn("m52", contents)


if __name__ == "__main__":
    unittest.main()
