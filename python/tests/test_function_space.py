"""Finite-memory teacher identity and C++ -> dataset -> network round-trip tests."""
import copy
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from renewal.collect import write_json
from renewal.reference_sampler import reference_sampler, configure_sampler, require_same_sampler
from renewal.reference_sampler import validate_sampler_row
from first_hit_se import EVENT, SE_KERNEL
from first_hit_se.data import collect, load_data, prepare
from first_hit_se.train import train, load_model, export_model, metrics
from first_hit_se.verify_cpp import verify_cpp


class FunctionSpaceTests(unittest.TestCase):
    def test_fixed_sampler_identity_and_step(self):
        identity = reference_sampler({"type": "gpis_function_space"})
        self.assertEqual(identity["history_points"], 1)
        self.assertEqual(identity["crossing_rule"], "tungsten_linear")
        fp = {"sampler": identity, "kernels": [SE_KERNEL], "grid": {"max_time": 1.013}}
        configure_sampler(fp, {})
        self.assertEqual(fp["grid"]["step_sizes"], [1/64])
        for key, value in (("history_points", 4), ("batch_size", 16), ("step_size", 1/128),
                           ("rank_tolerance", 1e-8), ("crossing_rule", "hermite_residual_cellwise_mean")):
            with self.assertRaisesRegex(ValueError, "reference_sampler mismatch"):
                require_same_sampler(identity, {**identity, key: value})
        with self.assertRaisesRegex(ValueError, "reference_sampler mismatch"):
            require_same_sampler(None, identity)
        with self.assertRaisesRegex(ValueError, "conflicts"):
            configure_sampler(fp, {"reference_steps": [1/128]})
        for change in ({"history_points": True}, {"history_points": 0}, {"batch_size": 129},
                       {"step_size": float("nan")}, {"rank_tolerance": 0}, {"version": 2},
                       {"history_derivatives": "all"}, {"batchsize": 3},
                       {"crossing_rule": "unknown"}, {"crossing_rule": []}):
            with self.assertRaises(ValueError):
                reference_sampler({**identity, **change})

    def test_crossing_labels_and_legacy_hermite_metadata(self):
        for rule, method in (("tungsten_linear", "total_field_linear_secant"),
                             ("hermite_residual_cellwise_mean", "analytic_derivative_hermite_cellwise_mean")):
            sampler = reference_sampler({"type": "gpis_function_space", "crossing_rule": rule})
            self.assertEqual(reference_sampler(sampler), sampler)
            validate_sampler_row(sampler, {"crossing_slope_method": method})
            for wrong in ({}, {"crossing_slope_method": "grid_cubic_hermite"}):
                with self.assertRaisesRegex(ValueError, "crossing_slope_method"):
                    validate_sampler_row(sampler, wrong)

    @unittest.skipUnless(os.environ.get("MACROFACET_RENEWAL_EXE"), "C++ reference executable not provided")
    def test_reference_dataset_training_export_and_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            executable = os.environ["MACROFACET_RENEWAL_EXE"]
            identity = reference_sampler({"type": "gpis_function_space", "history_points": 4})
            sources = []
            for i, split in enumerate(("train", "validation", "test")):
                request = root/f"{split}.json"
                write_json(request, {"schema_version": 1, "first_passage": {
                    "initial_condition": {"type": EVENT, "profiles": [
                        {"id": split, "beta_0": -.2+i*.3, "beta_a": -.5}]},
                    "grid": {"max_time": 1.013}, "curve": {"bins": 8},
                    "kernels": [{"id": "se", **SE_KERNEL, "variance": 1., "length_scale": 1.}]}})
                sources.append({"scene_id": split, "split": split, "reference_config": str(request)})
            settings = {"output_directory": str(root/"data"), "executable": executable,
                        "reference_sampler": identity, "realizations_per_profile": 64,
                        "threads": 2, "sources": sources}
            write_json(root/"collect.json", settings)
            collect(root/"collect.json")
            dataset = root/"data/dataset.pt"
            data = load_data(dataset)
            self.assertEqual(data["reference_sampler"], identity)
            self.assertEqual(len(data["profiles"]), 3)
            settings["reference_sampler"] = {**identity, "crossing_rule": "hermite_residual_cellwise_mean"}
            write_json(root/"different_collect.json", settings)
            with self.assertRaisesRegex(ValueError, "reference_sampler mismatch"):
                collect(root/"different_collect.json")
            self.assertEqual(load_data(dataset)["reference_sampler"], identity)
            # Existing explicitly marked Hermite sources still collect and load.
            settings["output_directory"] = str(root/"hermite_data")
            write_json(root/"hermite_collect.json", settings)
            collect(root/"hermite_collect.json")
            self.assertEqual(load_data(root/"hermite_data/dataset.pt")["reference_sampler"],
                             settings["reference_sampler"])
            for p in data["profiles"]:
                self.assertTrue(torch.all((p["distance"]>0) & (p["distance"]<=1.013)))
            source = root/"data/source_00000/reference"
            with (source/"first_passage_samples.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual({r["crossing_slope_method"] for r in rows}, {"total_field_linear_secant"})
            request = json.loads((root/"data/source_00000/collection_reference.json").read_text())
            request["output_directory"] = str(root/"single_thread")
            request["first_passage"]["monte_carlo"]["thread_count"] = 1
            # The C++ default must agree with Python's explicit linear convention.
            request["first_passage"]["sampler"].pop("crossing_rule")
            write_json(root/"single.json", request)
            subprocess.run([executable, "first-passage", "--config", str(root/"single.json")],
                           check=True, capture_output=True)
            self.assertEqual((source/"first_passage_samples.csv").read_bytes(),
                             (root/"single_thread/first_passage_samples.csv").read_bytes())
            resolved_default = json.loads((root/"single_thread/resolved_first_passage_config.json").read_text())
            self.assertEqual(resolved_default["first_passage"]["sampler"], identity)
            # Invalid steps must fail before creating output, including non-divisible horizons.
            for step in ([1/32], [1/64, 1/128]):
                request["output_directory"] = str(root/"invalid")
                request["first_passage"]["grid"]["step_sizes"] = step
                write_json(root/"invalid.json", request)
                failed = subprocess.run([executable, "first-passage", "--config", str(root/"invalid.json")],
                                        capture_output=True)
                self.assertNotEqual(failed.returncode, 0)
                self.assertFalse((root/"invalid").exists())
            request["first_passage"]["grid"]["step_sizes"] = [1/64]
            request["first_passage"]["sampler"]["crossing_rule"] = "unknown"
            write_json(root/"invalid.json", request)
            failed = subprocess.run([executable, "first-passage", "--config", str(root/"invalid.json")],
                                    capture_output=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertFalse((root/"invalid").exists())
            for architecture in ("gru", "rno"):
                config = {"dataset": str(dataset), "output_directory": str(root/architecture),
                    "reference_sampler": identity, "epochs": 1, "profiles_per_batch": 1,
                    "cpu_threads": 1, "model": {"architecture": architecture, "hidden": 4,
                        "width": 8, "mixture_width": 8, "components": 2, "maximum_step": .25}}
                write_json(root/"train.json", config)
                trained = train(root/"train.json")
                checkpoint = Path(trained["checkpoint"])
                model, bundle = load_model(checkpoint)
                self.assertEqual(bundle["reference_sampler"], identity)
                export_model(checkpoint, root/f"{architecture}.json")
                exported = json.loads((root/f"{architecture}.json").read_text())
                self.assertEqual(exported["reference_sampler"], identity)
                incompatible = copy.deepcopy(data["profiles"])
                incompatible[0]["reference_sampler"]["crossing_rule"] = "hermite_residual_cellwise_mean"
                with self.assertRaisesRegex(ValueError, "reference_sampler mismatch"):
                    metrics(model, incompatible)
                if os.environ.get("MACROFACET_FIRST_HIT_EXE"):
                    verify_cpp(checkpoint, dataset, os.environ["MACROFACET_FIRST_HIT_EXE"],
                               root/f"verify_{architecture}", limit=1, trials=128)
                config["reference_sampler"] = {**identity, "history_points": 1}
                write_json(root/"train.json", config)
                with self.assertRaisesRegex(ValueError, "reference_sampler mismatch"):
                    train(root/"train.json")
            # CSV slope conventions must agree with the declared rule, even if
            # all metadata agrees. Editing metadata cannot convert old labels.
            sample_path = source/"first_passage_samples.csv"
            original = sample_path.read_bytes()
            sample_path.write_bytes(original.replace(b"total_field_linear_secant",
                                                    b"analytic_derivative_hermite_cellwise_mean"))
            with self.assertRaisesRegex(ValueError, "crossing_slope_method"):
                prepare(root/"data/manifest.json", root/"wrong_slope.pt")
            sample_path.write_bytes(original)
            # Sources with different crossing rules cannot be packed together.
            resolved = root/"data/source_00001/reference/resolved_first_passage_config.json"
            changed = json.loads(resolved.read_text())
            changed["first_passage"]["sampler"]["crossing_rule"] = "hermite_residual_cellwise_mean"
            write_json(resolved, changed)
            with self.assertRaisesRegex(ValueError, "reference_sampler mismatch"):
                prepare(root/"data/manifest.json", root/"mixed.pt")

    @unittest.skipUnless(os.environ.get("MACROFACET_RENEWAL_EXE"), "C++ reference executable not provided")
    def test_renewal_ab_reference_and_model_identity(self):
        from renewal.data import prepare as prepare_ab, load_data as load_ab
        from renewal.train import train as train_ab
        from renewal.export import export_model as export_ab
        from renewal.evaluate import evaluate as evaluate_ab
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            identity = reference_sampler({"type": "gpis_function_space"})
            sources = []
            for i, split in enumerate(("train", "validation", "test")):
                state = {"id": split, "beta_0": .2+i*.1, "beta_a": -.3}
                initial = {"type": "positive_exterior", "profiles": [state]}
                if i == 1:
                    initial = {"type": "collision_state", "parameter_space": {
                        "type": "explicit", "states": [{**state, "beta_g": .7}]}}
                write_json(root/f"{split}.json", {"schema_version": 1, "seed": 814+i,
                    "output_directory": str(root/split), "first_passage": {
                        "initial_condition": initial, "grid": {"max_time": 1.013},
                        "sampler": identity, "curve": {"bins": 8},
                        "monte_carlo": {"trajectories": 32, "write_raw_samples": True, "thread_count": 2},
                        "kernels": [{"id": "se", **SE_KERNEL, "variance": 1., "length_scale": 1.}]}})
                subprocess.run([os.environ["MACROFACET_RENEWAL_EXE"], "first-passage", "--config",
                                str(root/f"{split}.json")], check=True, capture_output=True)
                sources.append({"scene_id": split, "split": split, "directory": str(root/split)})
            write_json(root/"manifest.json", {"schema_version": 1, "kernel": SE_KERNEL,
                        "reference_sampler": identity, "sources": sources})
            prepare_ab(root/"manifest.json", root/"dataset.pt")
            self.assertEqual(load_ab(root/"dataset.pt")["reference_sampler"], identity)
            config = {"schema_version": 1, "seed": 91, "dataset": str(root/"dataset.pt"),
                      "output_directory": str(root/"model"), "epochs": 1, "batch_size": 16,
                      "learning_rate": .001, "weight_decay": 0., "gradient_clip": 1., "patience": 1,
                      "device": "cpu", "cpu_threads": 1, "reference_sampler": identity,
                      "model": {"embedding": 4, "hidden": 4, "hazard_width": 8,
                                "mixture_width": 8, "components": 2}}
            write_json(root/"train.json", config)
            checkpoint = train_ab(root/"train.json")
            export_ab(checkpoint, root/"export.json")
            self.assertEqual(json.loads((root/"export.json").read_text())["reference_sampler"], identity)
            wrong = torch.load(checkpoint, weights_only=True)
            wrong["reference_sampler"]["batch_size"] = 16
            torch.save(wrong, root/"wrong.pt")
            with self.assertRaisesRegex(ValueError, "reference_sampler mismatch"):
                evaluate_ab(root/"wrong.pt", root/"dataset.pt", root/"wrong.json")


if __name__ == "__main__":
    unittest.main()
