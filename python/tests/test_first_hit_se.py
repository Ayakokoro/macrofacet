import math
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from first_hit_se.model import FirstHitNetwork, ModelConfig
from first_hit_se.geometry import sample_gradient
from first_hit_se import FORMAT, VERSION, EVENT, SE_KERNEL
from first_hit_se.data import prepare, load_data
from first_hit_se.train import export_model, train
from first_hit_se.batch import make_plan, query_many, losses_many, CudaIntegrator
from renewal.data import FEATURES
from renewal.profiles import subdivide_features


class FirstHitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def profile(self):
        # The same non-affine polynomial in one cell, b(u)=.4-1.2u+.7u^2-.3u^3.
        return torch.tensor([[.4, -.4, -1.2, -.7, 0.]], dtype=torch.float64)

    def model(self, architecture="rno", step=.125):
        torch.manual_seed(137)
        return FirstHitNetwork(ModelConfig(architecture=architecture, hidden=4, width=8,
            mixture_width=8, components=2, maximum_step=step)).double()

    def test_constant_rate_closed_form_and_both_signs(self):
        for architecture in ("rno", "gru"):
            model = self.model(architecture)
            for p in model.parameters():
                p.data.zero_()
            q = torch.tensor([0., .3, .8, 1.], dtype=torch.float64)
            prediction = model.query(self.profile(), q)
            torch.testing.assert_close(prediction["T"], torch.exp(-math.log(2)*q), atol=1e-12, rtol=1e-12)
            prior = torch.special.ndtr(torch.tensor(.4, dtype=torch.float64))
            torch.testing.assert_close(prediction["hit_branch_probability"][0], prior.expand(4))
            self.assertEqual(float(prediction["T"][0].detach()), 1.)

    def test_sign_symmetry_and_positive_gradient_branch(self):
        features = self.profile()
        opposite = features.clone(); opposite[:, :4] *= -1
        q = torch.linspace(0, 1, 7, dtype=torch.float64)
        for architecture in ("rno", "gru"):
            model = self.model(architecture)
            a, b = model.query(features, q), model.query(opposite, q)
            torch.testing.assert_close(a["T"], b["T"])
            torch.testing.assert_close(a["H"], b["H"].flip(0))
            torch.testing.assert_close(a["means"], b["means"].flip(0))
            self.assertTrue(torch.all(a["T"][1:] <= a["T"][:-1]))

    def test_rno_subdivision_converges_and_gru_is_retained(self):
        features = self.profile()
        split = subdivide_features(features, torch.tensor([7]))
        q = torch.tensor([0., .233, .711, 1.], dtype=torch.float64)
        model = self.model(step=.3)
        errors = []
        for step in (.3, .15, .075):
            a = model.query(features, q, maximum_step=step)
            b = model.query(split, q, maximum_step=step)
            errors.append(float((a["log_T"]-b["log_T"]).abs().max().detach()))
        self.assertLess(errors[-1], errors[0]/4)
        gru = self.model("gru")
        difference = (gru.query(features,q)["T"]-gru.query(split,q)["T"]).abs().max()
        self.assertGreater(float(difference.detach()), 1e-5)

    def test_censoring_and_signed_labels_have_finite_gradients(self):
        for architecture in ("rno", "gru"):
            model = self.model(architecture)
            labels = (torch.tensor([.25,.7,1.], dtype=torch.float64), torch.tensor([True,True,False]),
                      torch.tensor([1,-1,-1]), torch.tensor([.4,.8,math.nan], dtype=torch.float64))
            losses = model.losses(self.profile(), *labels)
            self.assertTrue(torch.isfinite(losses["complete_joint"]).all())
            self.assertAlmostEqual(float(losses["joint"][-1].detach()), -float(losses["prediction"]["log_T"][-1].detach()))
            losses["complete_joint"].mean().backward()
            for p in model.parameters():
                self.assertIsNotNone(p.grad)
                self.assertTrue(torch.isfinite(p.grad).all())

    def test_no_point_linear_discontinuities_or_invalid_queries(self):
        model = self.model()
        features = self.profile().repeat(2,1)
        with self.assertRaises(ValueError): model.query(features, torch.tensor([.1]))
        with self.assertRaises(ValueError): model.query(self.profile(), torch.tensor([1.1]))

    def test_batched_solver_preserves_values_and_parameter_gradients(self):
        profiles = [self.profile(), subdivide_features(self.profile(), torch.tensor([7])),
                    torch.tensor([[1., -.2, -.6, -.6, math.log(2.)]], dtype=torch.float64)]
        queries = [torch.tensor([.73, 1., 0., .73], dtype=torch.float64),
                   torch.tensor([.233, .711, .97], dtype=torch.float64),
                   torch.tensor([0., .8, 2.], dtype=torch.float64)]
        for architecture in ("rno", "gru"):
            model = self.model(architecture)
            plans = [make_plan(p, model.config.maximum_step) for p in profiles]
            batched = query_many(model, profiles, queries, plans=plans)
            individual = [model.query(f, q) for f, q in zip(profiles, queries)]
            for a, b in zip(batched, individual):
                for key in a:
                    torch.testing.assert_close(a[key], b[key], atol=1e-10, rtol=1e-10)
            def objective(predictions):
                return sum((p["H"]**2).sum()+p["log_hazard"].sum()+p["means"].sum()+p["scales"].sum() for p in predictions)
            g1 = torch.autograd.grad(objective(batched), tuple(model.parameters()))
            g2 = torch.autograd.grad(objective(individual), tuple(model.parameters()))
            for a, b in zip(g1, g2):
                torch.testing.assert_close(a, b, atol=1e-9, rtol=1e-9)

    def test_query_locations_do_not_change_integration_history(self):
        model = self.model()
        features = self.profile()
        sparse = torch.tensor([.73, 1., .73, 0.], dtype=torch.float64)
        crowded = torch.cat((torch.linspace(0, 1, 103, dtype=torch.float64), sparse))
        a, b = model.query(features, sparse), model.query(features, crowded)
        for name in ("H", "log_hazard", "means", "scales"):
            torch.testing.assert_close(a[name], b[name][:, -4:], atol=1e-12, rtol=1e-12)

    def test_hidden_expansion_preserves_predictions_and_can_learn_new_connections(self):
        original = self.model()
        expanded = FirstHitNetwork(ModelConfig(**{**original.configuration(), 'hidden': 9})).double()
        expanded.expand_hidden_from(original)
        q = torch.tensor([0., .13, .81, 1.], dtype=torch.float64)
        expected, actual = original.query(self.profile(), q), expanded.query(self.profile(), q)
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], atol=1e-12, rtol=1e-12)
        actual['T'].sum().backward()
        # Extra state coordinates are active, despite initially having no effect
        # on the old predictions; their output connections must receive gradients.
        extra = expanded.hazard_head[0].weight.grad[:, 4:9]
        self.assertGreater(float(extra.abs().sum()), 1e-8)
        with self.assertRaisesRegex(ValueError, 'increase'):
            original.expand_hidden_from(expanded)

    def test_joint_curve_batch_matches_separate_values_and_gradients(self):
        for architecture in ('rno', 'gru'):
            model = self.model(architecture)
            p = {'features': self.profile(),
                 'distance': torch.tensor([.25, .7, 1.], dtype=torch.float64),
                 'hit': torch.tensor([True, True, False]), 'initial_sign': torch.tensor([1, -1, -1]),
                 'speed': torch.tensor([.4, .8, math.nan], dtype=torch.float64),
                 'curve_queries': torch.linspace(0, 1, 9, dtype=torch.float64)}
            result = losses_many(model, [p])[0]
            reference = model.losses(p['features'], p['distance'], p['hit'], p['initial_sign'], p['speed'])
            curve = model.query(p['features'], p['curve_queries'], with_mixture=False)['T']
            torch.testing.assert_close(result['complete_joint'], reference['complete_joint'], atol=1e-11, rtol=1e-11)
            torch.testing.assert_close(result['curve_prediction'], curve, atol=1e-11, rtol=1e-11)
            a = result['complete_joint'].mean()+10*(result['curve_prediction']-.5).square().mean()
            b = reference['complete_joint'].mean()+10*(curve-.5).square().mean()
            ga = torch.autograd.grad(a, tuple(model.parameters()))
            gb = torch.autograd.grad(b, tuple(model.parameters()))
            for x, y in zip(ga, gb):
                torch.testing.assert_close(x, y, atol=1e-9, rtol=1e-9)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA graph test requires CUDA')
    def test_captured_joint_curve_updates_match_eager(self):
        models = [self.model().float().cuda() for _ in range(2)]
        p = {'features': self.profile().cuda(),
             'distance': torch.tensor([.25, .7, 1.], dtype=torch.float64, device='cuda'),
             'hit': torch.tensor([True, True, False], device='cuda'),
             'initial_sign': torch.tensor([1, -1, -1], device='cuda'),
             'speed': torch.tensor([.4, .8, math.nan], dtype=torch.float64, device='cuda'),
             'curve_queries': torch.linspace(0, 1, 9, dtype=torch.float64, device='cuda')}
        plans = [make_plan(p['features'], .125, 'cuda')]
        captured = CudaIntegrator(models[1], len(plans[0]['dt'])+1, 2)
        optimizers = [torch.optim.AdamW(m.parameters(), lr=.001) for m in models]
        for _ in range(3):
            for i, (model, optimizer) in enumerate(zip(models, optimizers)):
                optimizer.zero_grad(set_to_none=True)
                result = losses_many(model, [p], plans=plans, integrator=captured if i else None)[0]
                objective = result['complete_joint'].mean()+10*(result['curve_prediction']-.5).square().mean()
                objective.backward()
                optimizer.step()
                del result, objective
            for a, b in zip(models[0].parameters(), models[1].parameters()):
                torch.testing.assert_close(a, b, atol=3e-6, rtol=3e-6)
                torch.testing.assert_close(a.grad, b.grad, atol=3e-6, rtol=3e-6)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA graph test requires CUDA")
    def test_captured_solver_preserves_optimizer_updates_and_validation(self):
        models = [self.model().float().cuda() for _ in range(2)]
        profiles = [self.profile().cuda(), subdivide_features(self.profile(), torch.tensor([3])).cuda()]
        plans = [make_plan(f, .125, "cuda") for f in profiles]
        # The last batch can contain fewer profiles than the captured capacity.
        integrator = CudaIntegrator(models[1], max(len(p["dt"]) for p in plans)+2, 3)
        optimizers = [torch.optim.AdamW(m.parameters(), lr=.001) for m in models]
        queries = [torch.tensor([.19, .55, 1.], device="cuda", dtype=torch.float64) for _ in profiles]
        for count in (2, 1, 2):
            predictions = []
            for i, (model, optimizer) in enumerate(zip(models, optimizers)):
                optimizer.zero_grad(set_to_none=True)
                result = query_many(model, profiles[:count], queries[:count], plans=plans[:count],
                                    integrator=integrator if i else None)
                predictions.append([{k: v.detach().clone() for k, v in p.items()} for p in result])
                objective = sum(p["H"].square().mean()+p["means"].square().mean()+p["scales"].mean()
                                +p["log_weights"].square().mean()+p["log_hazard"].square().mean() for p in result)
                objective.backward()
                optimizer.step()
                del result, objective
            for a, b in zip(*predictions):
                for key in a:
                    torch.testing.assert_close(a[key], b[key], atol=3e-6, rtol=3e-6)
            for a, b in zip(models[0].parameters(), models[1].parameters()):
                torch.testing.assert_close(a.grad, b.grad, atol=3e-6, rtol=3e-6)
                torch.testing.assert_close(a, b, atol=3e-6, rtol=3e-6)
        for model in models:
            model.eval()
        with torch.no_grad():
            a = query_many(models[0], profiles, queries, plans=plans)
            b = query_many(models[1], profiles, queries, plans=plans, integrator=integrator)
            for p, q in zip(a, b):
                for key in p:
                    torch.testing.assert_close(p[key], q[key], atol=3e-6, rtol=3e-6)

    def test_anisotropic_full_gradient_for_up_and_down_crossings(self):
        rng = np.random.default_rng(19)
        direction = np.array([1.,2.,-1.]); direction /= np.linalg.norm(direction)
        C = np.array([[1.,.2,.1],[.2,2.,.3],[.1,.3,.7]])
        mean = np.array([.3,-.5,.7])
        geometry = {"direction": direction, "sigma": 1., "ell": 1., "gradient_covariance": C}
        cw = C@direction
        covariance = C-np.outer(cw,cw)/(direction@cw)
        for derivative in (-.8,.6):
            samples = np.array([sample_gradient(geometry, mean, derivative, rng) for _ in range(6000)])
            np.testing.assert_allclose(samples@direction, derivative, atol=1e-12)
            target = mean+cw/(direction@cw)*(derivative-direction@mean)
            np.testing.assert_allclose(samples.mean(0), target, atol=.06)
            np.testing.assert_allclose(np.cov(samples.T), covariance, atol=.08)

    def test_data_rejects_conditioned_labels_and_invalid_signed_gradients(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root/"reference"; source.mkdir()
            def write(name, value):
                (source/name).write_text(json.dumps(value), encoding="utf-8")
            config = {"first_passage": {"initial_condition": {"type": EVENT},
                "kernels": [{"id": "se", "type": "squared_exponential", "parameterization": "unit_length"}],
                "grid": {"max_time": 1., "step_sizes": [.125, .0625]}, "monte_carlo": {"trajectories": 2}}}
            write("resolved_first_passage_config.json", config)
            segment = dict(kernel_id="se", state_id="p", start_mode=EVENT, segment=0, x_begin=0., x_end=1.)
            segment.update(zip(FEATURES, self.profile()[0].tolist()))
            def table(name, records):
                with (source/name).open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(records[0]))
                    writer.writeheader(); writer.writerows(records)
            table("first_passage_mean_segments.csv", [segment])
            write("first_hit_geometry.json", [{"kernel_id": "se", "state_id": "p", "sigma": 1., "ell": 1.,
                "origin": [0,0,0], "direction": [0,0,1],
                "mean_gradient_polynomials": [[[0,0,0],[0,0,0],[-.9,1.4,-1.2]]]}])
            base = dict(kernel_id="se", state_id="p", training_resolution="1", initial_sign=-1,
                sigma=1., ell=1., base_step=.0625, sampler_type="unconditioned_grid_circulant")
            records = [dict(base, trajectory=0, event="1", censored="0", event_q=.3, event_distance=.3,
                            crossing_slope=.7, crossing_derivative=.7),
                       dict(base, trajectory=1, event="0", censored="1", event_q=1., event_distance=1.,
                            crossing_slope="", crossing_derivative="")]
            table("first_passage_samples.csv", records)
            manifest = root/"manifest.json"
            manifest.write_text(json.dumps({"format": FORMAT, "version": VERSION, "event": EVENT, "kernel": SE_KERNEL,
                "sources": [{"scene_id": "s", "split": "train", "directory": "reference"}]}), encoding="utf-8")
            prepare(manifest, root/"data.pt")
            p = load_data(root/"data.pt")["profiles"][0]
            self.assertEqual(p["initial_sign"].tolist(), [-1, -1])
            self.assertTrue(math.isnan(float(p["speed"][1])))
            records[0]["crossing_derivative"] = -.7
            table("first_passage_samples.csv", records)
            with self.assertRaisesRegex(ValueError, "signed derivative"):
                prepare(manifest, root/"invalid.pt")
            config["first_passage"]["initial_condition"]["type"] = "positive_exterior"
            write("resolved_first_passage_config.json", config)
            with self.assertRaisesRegex(ValueError, "conditioned A/B"):
                prepare(manifest, root/"invalid.pt")

    def test_training_resume_preserves_optimizer_scheduler_and_sampling(self):
        from renewal.collect import write_json
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profiles = []
            for i, split in enumerate(("train", "train", "validation")):
                features = self.profile().clone(); features[:, :2] += i*.2
                profiles.append({"features": features, "horizon": 1., "split": split,
                    "distance": torch.tensor([.2, .3, .6, 1., 1.], dtype=torch.float64),
                    "hit": torch.tensor([True, True, True, False, False]),
                    "initial_sign": torch.tensor([1, -1, 1, 1, -1]),
                    "speed": torch.tensor([.4, .8, .6, math.nan, math.nan], dtype=torch.float64)})
            torch.save({"format": FORMAT, "version": VERSION, "event": EVENT, "kernel": SE_KERNEL,
                        "profile_mode": "cubic", "profiles": profiles}, root/"dataset.pt")
            for architecture in ("rno", "gru"):
                config = {"dataset": str(root/"dataset.pt"), "output_directory": str(root/f"{architecture}_whole"),
                    "device": "cpu", "cpu_threads": 1, "epochs": 3, "profiles_per_batch": 2,
                    "samples_per_profile_per_epoch": 3, "batched_solver": True,
                    "lr_scheduler": {"factor": .5, "patience": 0, "threshold": 100., "threshold_mode": "abs"},
                    "model": self.model(architecture).configuration()}
                config.update(curve_loss_weight=3., curve_points=7)
                if architecture == 'rno':
                    config['lr_scheduler'] = {'type': 'cosine', 'T_max': 3, 'eta_min': .00003}
                write_json(root/"config.json", config)
                train(root/"config.json")
                whole = torch.load(root/f"{architecture}_whole"/"last.pt", weights_only=True)
                config["output_directory"] = str(root/f"{architecture}_resumed")
                write_json(root/"config.json", config)
                def interrupt(path, value):
                    if Path(path).name == "history.json":
                        raise RuntimeError("simulated interruption after checkpoint")
                    return write_json(path, value)
                with mock.patch("first_hit_se.train.write_json", side_effect=interrupt):
                    with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                        train(root/"config.json")
                train(root/"config.json", resume=True)
                resumed = torch.load(root/f"{architecture}_resumed"/"last.pt", weights_only=True)
                self.assertEqual(whole["optimizer_updates"], resumed["optimizer_updates"])
                self.assertEqual(whole["scheduler_state"], resumed["scheduler_state"])
                for key, expected in whole["model_state"].items():
                    torch.testing.assert_close(resumed["model_state"][key], expected, atol=0., rtol=0.)

    @unittest.skipUnless(os.environ.get("MACROFACET_FIRST_HIT_EXE"), "C++ query executable supplied by CTest")
    @torch.no_grad()
    def test_cpp_export_query_and_unconditioned_sampling(self):
        executable = os.environ["MACROFACET_FIRST_HIT_EXE"]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            features = self.profile()
            q = torch.linspace(0, 1, 33, dtype=torch.float64)
            request = root/"profile.json"
            request.write_text(json.dumps({"features": features.tolist(), "queries": q.tolist()}), encoding="utf-8")
            for architecture in ("rno", "gru"):
                model = self.model(architecture).float()
                bundle = {"format": FORMAT, "version": VERSION, "event": EVENT, "kernel": SE_KERNEL,
                          "model_config": model.configuration(), "model_state": model.state_dict()}
                torch.save(bundle, root/"checkpoint.pt")
                export_model(root/"checkpoint.pt", root/"model.json")
                subprocess.run([executable, str(root/"model.json"), str(request), str(root/"result.json"), "512"],
                               check=True, capture_output=True, text=True)
                result = json.loads((root/"result.json").read_text())
                expected = model.query(features, q)
                for name in ("H", "log_hazard", "hit_branch_probability"):
                    actual = torch.tensor([p[name] for p in result["predictions"]], dtype=torch.float64).T
                    torch.testing.assert_close(actual, expected[name].double(), atol=2e-5, rtol=2e-5)
                for name in ("weights", "means", "scales"):
                    actual = torch.tensor([[m[name] for m in p["mixtures"]] for p in result["predictions"]], dtype=torch.float64).transpose(0,1)
                    target = expected["log_weights"].exp() if name == "weights" else expected[name]
                    torch.testing.assert_close(actual, target.double(), atol=2e-5, rtol=2e-5)
                samples = result["samples"]
                observed = torch.tensor([sum(not s["hit"] or s["q"] > x for s in samples)/len(samples) for x in q])
                self.assertLess(float((observed-expected["T"]).abs().max()), .1)
                for sign in (1, -1):
                    self.assertGreater(sum(s["hit"] and s["initial_sign"] == sign and s["speed"] > 0 for s in samples), 10)
                invalid = json.loads((root/"model.json").read_text())
                invalid["distance_convention"] = "from_camera"
                (root/"model.json").write_text(json.dumps(invalid), encoding="utf-8")
                rejected = subprocess.run([executable, str(root/"model.json"), str(request), str(root/"bad.json")], capture_output=True)
                self.assertNotEqual(rejected.returncode, 0)


if __name__ == "__main__":
    unittest.main()
