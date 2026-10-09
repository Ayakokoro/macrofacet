import csv
import contextlib
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.nn import functional as F

from renewal import KERNEL
from renewal.data import prepare, load_data, SequenceDataset, collate
from renewal.evaluate import load_model, survival_calibration
from renewal.train import train
from renewal.model import (RenewalNetwork, ModelConfig, cumulative, log_hazard, mean_at,
                           mixture_log_prob, mixture_cdf)


def fixture(directory):
    sources = []
    for i, split in enumerate(("train", "validation", "test")):
        run = directory / split
        run.mkdir()
        mode = "collision_state" if i == 1 else "positive_exterior"
        config = {"first_passage": {"initial_condition": {"type": mode},
                  "kernels": [{"id": "m32", **KERNEL}],
                  "grid": {"max_time": 1.0, "step_sizes": [0.03125]}, "sampler": {}}}
        (run / "resolved_first_passage_config.json").write_text(json.dumps(config))
        with (run / "first_passage_mean_segments.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["kernel_id", "state_id", "start_mode", "segment", "x_begin", "x_end",
                             "b0", "b1", "dx_db0", "dx_db1", "log_dx"])
            for j in range(2):
                writer.writerow(["m32", "ray", mode, j, j*0.5, (j+1)*0.5, 0.3, 0.3, 0, 0, math.log(0.5)])
        with (run / "first_passage_samples.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["kernel_id", "state_id", "training_resolution", "sampler_type", "trajectory",
                             "beta_0", "beta_g", "event", "censored", "event_q", "crossing_slope", "max_q", "base_step"])
            writer.writerow(["m32", "ray", 1, "matern32_state_space", 0, 0.3,
                             0.7 if i == 1 else "", 1, 0, 0.25, 0.8, 1, 0.03125])
            writer.writerow(["m32", "ray", 1, "matern32_state_space", 1, 0.3,
                             0.7 if i == 1 else "", 0, 1, 1, "", 1, 0.03125])
        sources.append({"scene_id": split, "split": split, "directory": str(run), "geometry": {"unique": i}})
    path = directory / "manifest.json"
    path.write_text(json.dumps({"schema_version": 1, "kernel": KERNEL, "sources": sources}))
    return path


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(723)
        torch.set_num_threads(2)
        self.model = RenewalNetwork(ModelConfig(embedding=8, hidden=12, hazard_width=8, mixture_width=12, components=2))

    def test_cumulative_derivative_and_mass(self):
        logits = torch.tensor([[0.2, -2.0, 1.0, 0.1]], dtype=torch.double)
        u = torch.tensor([0.37], dtype=torch.double, requires_grad=True)
        dx = torch.tensor([0.21], dtype=torch.double)
        value = cumulative(logits, dx, u)
        derivative = torch.autograd.grad(value.sum(), u)[0]/dx
        torch.testing.assert_close(derivative, log_hazard(logits, u).exp())
        self.assertEqual(cumulative(logits, dx, torch.zeros_like(u)).item(), 0)
        torch.testing.assert_close(cumulative(logits, dx, torch.ones_like(u)), dx*F.softplus(logits).mean(-1))
        grid = torch.linspace(0, 1, 101, dtype=torch.double)
        values = cumulative(logits.expand(101, -1), dx.expand(101), grid)
        self.assertTrue((values.diff() >= 0).all())

    def test_no_hazard_floor(self):
        logits = torch.full((2, 4), -1000.0, requires_grad=True)
        u = torch.tensor([0.0, 1.0])
        logs = log_hazard(logits, u)
        torch.testing.assert_close(logs, torch.full((2,), -1000.0))
        logs.sum().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertEqual(cumulative(logits, torch.full((2,), 1e6), u).sum().item(), 0)

    def test_mixture_is_normalized_positive_half_axis(self):
        w = torch.linspace(0, 14, 30001, dtype=torch.double)
        weights = torch.tensor([[0.3, 0.7]], dtype=torch.double).log().expand(len(w), -1)
        means = torch.tensor([[-2.0, 1.0]], dtype=torch.double).expand(len(w), -1)
        scales = torch.tensor([[0.7, 1.3]], dtype=torch.double).expand(len(w), -1)
        pdf = mixture_log_prob(weights, means, scales, w).exp()
        self.assertAlmostEqual(torch.trapezoid(pdf, w).item(), 1.0, delta=0.0005)
        self.assertEqual(pdf[0].item(), 0)
        torch.testing.assert_close(mixture_cdf(weights[:1], means[:1], scales[:1], torch.tensor([14.])),
                                   torch.ones(1, dtype=torch.double))

    def test_extreme_tail_has_finite_likelihood_and_gradients(self):
        means = torch.tensor([[-1000.0]], dtype=torch.double, requires_grad=True)
        scale = torch.ones_like(means, requires_grad=True)
        value = mixture_log_prob(torch.zeros_like(means), means, scale, torch.tensor([0.001]))
        value.sum().backward()
        self.assertTrue(torch.isfinite(value).all() and torch.isfinite(means.grad).all() and torch.isfinite(scale.grad).all())
        self.assertAlmostEqual(value.item(), math.log(1000)-1, delta=0.00001)

    def test_short_cell_derivative_preserves_units(self):
        dx = 1e-7
        f = torch.tensor([[4., 4.+2*dx, 2*dx, 2*dx, math.log(dx)]], dtype=torch.double)
        _, derivative = mean_at(f, torch.tensor([0.37], dtype=torch.double))
        self.assertAlmostEqual(derivative.item(), 2.0, places=7)

    def test_causality_and_exterior_unknown_initials(self):
        f = torch.randn(2, 4, 5)
        initial = torch.tensor([[0., 0.2, 0., 0.], [1., 0.2, -0.2, 0.7]])
        lengths = torch.tensor([4, 4])
        _, original = self.model.encode(f, initial, lengths)
        f2 = f.clone(); f2[:, 2:] += 20
        altered = initial.clone(); altered[0, 2:] = torch.tensor([13., -17.])
        _, changed = self.model.encode(f2, altered, lengths)
        torch.testing.assert_close(original[:, :2], changed[:, :2])
        _, prefix = self.model.encode(f[:, :2], initial, torch.tensor([2, 2]))
        torch.testing.assert_close(original[:, :2], prefix)

    def test_censored_joint_loss_and_unused_speed(self):
        # A constant rate model must reproduce an exponential's censored NLL.
        for p in self.model.hazard_head.parameters():
            p.data.zero_()
        features = torch.tensor([[[0., 0., 0., 0., math.log(0.5)]]*2]*2)
        batch = {"features": features, "initial": torch.zeros(2, 4), "lengths": torch.tensor([2, 2]),
                 "u": torch.tensor([0.4, 1.]), "hit": torch.tensor([True, False]),
                 "speed": torch.tensor([0.7, math.nan])}
        values = self.model.losses(batch)
        rate = math.log(2)
        self.assertAlmostEqual(values["distance"][0].item(), rate*0.7-math.log(rate), places=6)
        self.assertAlmostEqual(values["joint"][1].item(), rate, places=6)
        values["joint"].mean().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in self.model.parameters() if p.grad is not None))

    def test_checkpoint_predictions_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/"model.pt"
            torch.save({"format_version": 1, "kernel": KERNEL, "model_config": self.model.configuration(),
                        "model_state": self.model.state_dict()}, path)
            loaded, _ = load_model(path, "cpu")
            for a, b in zip(self.model.parameters(), loaded.parameters()):
                torch.testing.assert_close(a, b)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_double_geometry_cuda_backward(self):
        model = self.model.cuda()
        batch = {"features": torch.tensor([[[0., 0., 0., 0., math.log(0.25)]]]*2,
                                           dtype=torch.double, device="cuda"),
                 "initial": torch.zeros(2, 4, dtype=torch.double, device="cuda"),
                 "lengths": torch.tensor([1, 1]), "u": torch.tensor([0.4, 1.], device="cuda"),
                 "hit": torch.tensor([True, False], device="cuda"),
                 "speed": torch.tensor([0.7, math.nan], device="cuda")}
        loss = model.losses(batch)["joint"].mean()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))


class DataTests(unittest.TestCase):
    def test_training_checkpoint_resume_and_evaluation(self):
        with tempfile.TemporaryDirectory() as d:
            directory = Path(d)
            manifest = fixture(directory)
            dataset = directory/"data.pt"
            prepare(manifest, dataset)
            config = {"schema_version": 1, "seed": 812, "dataset": str(dataset),
                      "output_directory": str(directory/"model"), "device": "cpu", "cpu_threads": 2,
                      "batch_size": 2, "epochs": 1, "patience": 1, "learning_rate": 0.0003,
                      "weight_decay": 1e-5, "gradient_clip": 1.0,
                      "model": {"embedding": 4, "hidden": 4, "hazard_width": 4,
                                "mixture_width": 4, "components": 2}}
            path = directory/"train.json"
            path.write_text(json.dumps(config))
            with contextlib.redirect_stdout(io.StringIO()):
                checkpoint = train(path)
                before, _ = load_model(checkpoint, "cpu")
                train(path, resume=True)
                after, _ = load_model(checkpoint, "cpu")
            for a, b in zip(before.parameters(), after.parameters()):
                torch.testing.assert_close(a, b)
            metrics = json.loads((directory/"model"/"test_metrics.json").read_text())
            self.assertEqual(metrics["likelihood"]["all"]["samples"], 2)
            self.assertIn("scene:test", metrics["likelihood"])

    def test_prefix_and_censoring_and_calibration(self):
        with tempfile.TemporaryDirectory() as d:
            directory = Path(d)
            manifest = fixture(directory)
            output = directory/"data.pt"
            report = prepare(manifest, output)
            data = load_data(output)
            self.assertEqual(report["counts"]["train"]["samples"], 2)
            dataset = SequenceDataset(data, "train")
            self.assertEqual(len(dataset[0]["features"]), 1)
            self.assertEqual(len(dataset[1]["features"]), 2)
            self.assertTrue(torch.isnan(dataset[1]["speed"]))
            self.assertEqual(data["initial"][0, 2:].abs().sum(), 0)
            self.assertAlmostEqual(data["initial"][1, 3].item(), 0.7)
            model = RenewalNetwork(ModelConfig(embedding=4, hidden=4, hazard_width=4, mixture_width=4, components=2))
            for p in model.hazard_head.parameters(): p.data.zero_()
            calibration = survival_calibration(model, data, "test", "cpu")
            self.assertAlmostEqual(calibration["all"][-1]["predicted_survival"], 0.5, places=6)
            self.assertAlmostEqual(calibration["all"][-1]["observed_survival"], 0.5, places=6)
            losses = model.losses(collate([dataset[0], dataset[1]]))
            self.assertTrue(torch.isfinite(losses["joint"]).all())

    def test_reject_split_leakage_and_old_kernel(self):
        with tempfile.TemporaryDirectory() as d:
            directory = Path(d)
            manifest = fixture(directory)
            source = json.loads(manifest.read_text())
            source["sources"][1]["scene_id"] = source["sources"][0]["scene_id"]
            manifest.write_text(json.dumps(source))
            with self.assertRaisesRegex(ValueError, "one split"):
                prepare(manifest, directory/"data.pt")
            source["sources"][1]["scene_id"] = "validation"
            manifest.write_text(json.dumps(source))
            config_path = directory/"train"/"resolved_first_passage_config.json"
            config = json.loads(config_path.read_text())
            del config["first_passage"]["kernels"][0]["parameterization"]
            config_path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "legacy"):
                prepare(manifest, directory/"data.pt")

    def test_reject_dropped_censoring_and_false_finest_flag(self):
        with tempfile.TemporaryDirectory() as d:
            directory = Path(d)
            manifest = fixture(directory)
            config_path = directory/"train"/"resolved_first_passage_config.json"
            config = json.loads(config_path.read_text())
            config["first_passage"]["monte_carlo"] = {"trajectories": 3}
            config_path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "incomplete trajectory"):
                prepare(manifest, directory/"data.pt")
            config["first_passage"]["monte_carlo"]["trajectories"] = 2
            config["first_passage"]["grid"]["step_sizes"].append(0.015625)
            config_path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "finest reference"):
                prepare(manifest, directory/"data.pt")


if __name__ == "__main__":
    unittest.main()
