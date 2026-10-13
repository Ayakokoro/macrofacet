import contextlib
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader

from renewal import SE_KERNEL
from renewal.data import SequenceDataset, collate
from renewal.evaluate import likelihood, load_model
from renewal.export import export_model
from renewal.model import ModelConfig, RenewalNetwork, mean_at
from renewal.profiles import subdivide_features, with_profile_mode
from renewal.train import train


def value(x):
    return 0.3 + 0.7*x + 0.2*x*x - 0.15*x*x*x


def derivative(x):
    return 0.7 + 0.4*x - 0.45*x*x


def curved_data():
    ends = [0., 0.5, 0.75, 1.]
    profile = [[value(a), value(b), (b-a)*derivative(a), (b-a)*derivative(b), math.log(b-a)]
               for a, b in zip(ends[:-1], ends[1:])]
    # Two start conditions per split, six outcomes per profile, including miss.
    return {
        "format_version": 1, "kernel": SE_KERNEL,
        "features": torch.tensor(profile * 6, dtype=torch.float64),
        "offsets": torch.arange(0, 19, 3),
        "initial": torch.tensor([[0., 0.3, 0., 0.], [1., 0.3, -0.3, 0.2]] * 3, dtype=torch.float64),
        "split": torch.tensor([0, 0, 1, 1, 2, 2]),
        "horizon": torch.ones(6, dtype=torch.float64),
        "scene_ids": ["train"]*2 + ["validation"]*2 + ["test"]*2,
        "sample_profile": torch.arange(6).repeat_interleave(6),
        "end_segment": torch.tensor([0, 0, 0, 1, 2, 2]*6),
        "end_u": torch.tensor([0.25, 0.75, 1., 0.25, 0.6, 1.]*6),
        "distance": torch.tensor([0.125, 0.375, 0.5, 0.5625, 0.9, 1.]*6, dtype=torch.float64),
        "hit": torch.tensor([True]*5 + [False]).repeat(6),
        "speed": torch.tensor([0.8, 0.9, 1., 1.1, 1.2, math.nan]*6),
    }


class ProfileTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_point_linear_uses_starting_tangents_without_mutating_labels(self):
        original = curved_data()
        saved = original["features"].clone()
        linear = with_profile_mode(original, "point_linear")
        for i, x in enumerate((0., 0.5, 0.75)):
            f = linear["features"][i]
            for u in (0., 0.37, 1.):
                b, db = mean_at(f, torch.tensor(u, dtype=torch.float64))
                self.assertAlmostEqual(b.item(), value(x) + derivative(x)*u*f[4].exp().item(), places=13)
                self.assertAlmostEqual(db.item(), derivative(x), places=12)
        torch.testing.assert_close(original["features"], saved, rtol=0, atol=0)
        for key in ("initial", "distance", "hit", "speed", "end_segment", "end_u"):
            self.assertIs(linear[key], original[key])
        self.assertIs(with_profile_mode(linear, "point_linear"), linear)
        # The next segment starts at its original field value, not the previous tangent's end.
        self.assertNotEqual(linear["features"][0, 1].item(), linear["features"][1, 0].item())

    def test_exact_subdivision_preserves_cubic_and_point_linear_means(self):
        for mode in ("cubic", "point_linear"):
            data = with_profile_mode(curved_data(), mode)
            features = data["features"][:3]
            counts = torch.tensor([2, 3, 5])
            pieces = subdivide_features(features, counts)
            position = 0
            for parent, (a, b, n) in enumerate(zip((0., .5, .75), (.5, .75, 1.), counts.tolist())):
                for j in range(n):
                    piece = pieces[position]
                    for v in (0., .173, .5, .91, 1.):
                        x = a + (b-a)*(j+v)/n
                        actual, slope = mean_at(piece, torch.tensor(v, dtype=torch.float64))
                        expected = value(x) if mode == "cubic" else value(a) + derivative(a)*(x-a)
                        expected_slope = derivative(x) if mode == "cubic" else derivative(a)
                        self.assertAlmostEqual(actual.item(), expected, places=12)
                        self.assertAlmostEqual(slope.item(), expected_slope, places=11)
                    position += 1
            self.assertAlmostEqual(pieces[:, 4].exp().sum().item(), 1., places=14)
            torch.testing.assert_close(subdivide_features(features, torch.ones(3, dtype=torch.long)),
                                       features, rtol=0, atol=0)

    def test_short_segments_keep_double_geometry(self):
        dx = 1e-7
        f = torch.tensor([[4., 4.+2*dx, 2*dx, 2*dx, math.log(dx)]], dtype=torch.float64)
        pieces = subdivide_features(f, torch.tensor([16]))
        _, slopes = mean_at(pieces, torch.full((16,), .37, dtype=torch.float64))
        self.assertEqual(pieces.dtype, torch.float64)
        torch.testing.assert_close(slopes, torch.full_like(slopes, 2.), rtol=0, atol=1e-6)

    def test_event_remapping_boundaries_censoring_and_shared_prefixes(self):
        for mode in ("cubic", "point_linear"):
            data = with_profile_mode(curved_data(), mode)
            dataset = SequenceDataset(data, "train", subdivision_factors=[4], seed=723)
            for i in range(len(dataset)):
                record = dataset[i]
                dx = record["features"][:, 4].exp()
                distance = dx[:-1].sum() + dx[-1]*record["u"]
                self.assertAlmostEqual(distance.item(), data["distance"][i].item(), places=14)
                torch.testing.assert_close(record["speed"], data["speed"][i], equal_nan=True)
                torch.testing.assert_close(record["initial"], data["initial"][data["sample_profile"][i]])
                self.assertEqual(record["hit"], data["hit"][i])
            for i, expected_length in ((0, 1), (1, 3), (2, 4), (3, 5), (5, 12)):
                self.assertEqual(len(dataset[i]["features"]), expected_length)
                self.assertEqual(dataset[i]["u"].item(), 1.)
            torch.testing.assert_close(dataset[1]["features"], dataset[5]["features"][:3])
            # Censoring/event distance still defines the correct constant-rate likelihood.
            model = RenewalNetwork(ModelConfig(embedding=4, hidden=4, hazard_width=4, mixture_width=4, components=2))
            for parameter in model.hazard_head.parameters():
                parameter.data.zero_()
            losses = model.losses(collate([dataset[i] for i in range(len(dataset))]))
            expected = math.log(2)*data["distance"][:12] - data["hit"][:12]*math.log(math.log(2))
            torch.testing.assert_close(losses["distance"], expected, rtol=1e-7, atol=1e-7)
            losses["joint"].mean().backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_epoch_randomness_is_reproducible_and_independent_of_sample_order(self):
        data = with_profile_mode(curved_data(), "point_linear")
        first = SequenceDataset(data, "train", subdivision_factors=[1, 2, 4, 8, 16], seed=912)
        second = SequenceDataset(data, "train", subdivision_factors=[1, 2, 4, 8, 16], seed=912)
        first.set_epoch(7)
        expected = [first[i] for i in range(len(first))]
        torch.manual_seed(9981)
        second.set_epoch(7)
        for i in reversed(range(len(second))):
            for key in ("features", "u", "initial", "speed"):
                torch.testing.assert_close(second[i][key], expected[i][key], rtol=0, atol=0, equal_nan=True)
        first.set_epoch(8)
        changed = first[5]["features"]
        self.assertFalse(torch.equal(changed, expected[5]["features"]))
        first.set_epoch(7)
        torch.testing.assert_close(first[5]["features"], expected[5]["features"], rtol=0, atol=0)

    def test_reject_invalid_modes_factors_and_validation_augmentation(self):
        data = curved_data()
        for factors in ([], [0], [-1], [1.5], [True], [1, 1], "1,2"):
            with self.subTest(factors=factors), self.assertRaisesRegex(ValueError, "subdivision_factors"):
                SequenceDataset(data, "train", subdivision_factors=factors)
        for split in ("validation", "test"):
            with self.assertRaisesRegex(ValueError, "train split"):
                SequenceDataset(data, split, subdivision_factors=[2])
        with self.assertRaisesRegex(ValueError, "profile_mode"):
            with_profile_mode(data, "typo")
        with self.assertRaisesRegex(ValueError, "cannot recover"):
            with_profile_mode(with_profile_mode(data, "point_linear"), "cubic")

    def test_training_evaluation_export_and_interrupted_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_path = root / "data.pt"
            data = curved_data()
            torch.save(data, dataset_path)
            config = {"schema_version": 1, "seed": 812, "kernel": SE_KERNEL,
                      "dataset": str(dataset_path), "device": "cpu", "cpu_threads": 2,
                      "batch_size": 4, "epochs": 3, "patience": 5,
                      "learning_rate": .0003, "weight_decay": 1e-5, "gradient_clip": 1.,
                      "profile_mode": "point_linear", "augmentation": {"subdivision_factors": [1, 2, 4]},
                      "model": {"embedding": 4, "hidden": 4, "hazard_width": 4, "mixture_width": 4, "components": 2}}
            config_path = root / "train.json"
            config["output_directory"] = str(root / "full")
            config_path.write_text(json.dumps(config))
            with contextlib.redirect_stdout(io.StringIO()):
                train(config_path)
            config["output_directory"] = str(root / "resumed")
            config_path.write_text(json.dumps(config))
            real_save = torch.save

            def save_then_interrupt(bundle, path):
                real_save(bundle, path)
                if Path(path).name == "last.pt" and bundle["epoch"] == 1:
                    raise RuntimeError("simulated interruption after checkpoint")

            with contextlib.redirect_stdout(io.StringIO()):
                with patch("renewal.train.torch.save", side_effect=save_then_interrupt):
                    with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                        train(config_path)
                checkpoint = train(config_path, resume=True)
            full = torch.load(root / "full/last.pt", weights_only=True)
            resumed = torch.load(root / "resumed/last.pt", weights_only=True)
            for name in full["model_state"]:
                torch.testing.assert_close(full["model_state"][name], resumed["model_state"][name], rtol=0, atol=0)
            self.assertEqual(full["validation"], resumed["validation"])
            self.assertEqual(resumed["profile_mode"], "point_linear")
            metrics = json.loads((root / "resumed/test_metrics.json").read_text())
            model, _ = load_model(checkpoint, "cpu")
            linear = with_profile_mode(data, "point_linear")
            loader = DataLoader(SequenceDataset(linear, "test"), batch_size=4, collate_fn=collate)
            self.assertEqual(metrics["likelihood"], likelihood(model, loader, "cpu", linear))
            self.assertEqual(metrics["profile_mode"], "point_linear")
            export_model(checkpoint, root / "export.json")
            exported = json.loads((root / "export.json").read_text())
            self.assertEqual(exported["training_profile_mode"], "point_linear")
            self.assertEqual(exported["training_augmentation"], config["augmentation"])


if __name__ == "__main__":
    unittest.main()
