import contextlib
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from first_hit_se.causal_gradient import GradientConfig
from first_hit_se.causal_residual import CausalConfig
from first_hit_se.distance_study import alter_suffix, restrict_profile
from first_hit_se.gradient_cdf import cdf, log_prob, sample
from first_hit_se.hazard_study import parameter_hash
from renewal.causal_data import ProfileDataset, query_model
from renewal.causal_gradient import HitGradient, cache_profile, load_joint, loss, points, query_points, raw_profiles
from renewal.causal_gradient_train import evaluate, evaluate_profiles, train
from renewal.causal_residual import RenewalResidual, load_model
from renewal.causal_train import train as train_distance
from renewal.collect import digest, write_json
from test_renewal_causal import fixture as distance_fixture


def fixture(path):
    data = distance_fixture(path)
    data['speed'] = torch.where(data['hit'], .3+data['distance']+.2*data['sample_profile'], math.nan).float()
    torch.save(data, path)
    return data


class HitGradientTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)
        self.backbone = RenewalResidual(CausalConfig(hidden=4, embedding=4, width=8)).requires_grad_(False).eval()
        self.head = HitGradient(GradientConfig(hidden=4, width=8, size=4), 71)
        self.f = torch.tensor([[.6, -.4, -1., -.5, math.log(.7)],
                               [-.4, .8, .3, 1., math.log(.6)]], dtype=torch.float64)
        self.a = torch.tensor([0., .6, 0., 0.], dtype=torch.float64)
        self.b = torch.tensor([1., .6, -.6, 2.], dtype=torch.float64)

    def raw(self, initial):
        return {'features': self.f, 'initial': initial, 'horizon': torch.tensor(1.3, dtype=torch.float64),
                'distance': torch.tensor([.125, .213, 1.3, 1.3], dtype=torch.float64),
                'hit': torch.tensor([True, True, True, False]),
                'speed': torch.tensor([.3, 1.2, 4.1, math.nan]), 'profile': torch.tensor(int(initial[0]))}

    def test_cached_and_direct_queries_match_for_both_conditions(self):
        for initial in (self.a, self.b):
            raw = self.raw(initial)
            row = cache_profile(self.backbone, raw)
            cached = points([row])
            direct = query_points(self.backbone, self.f, initial, raw['distance'][raw['hit']])
            for key in direct:
                torch.testing.assert_close(cached[key], direct[key], atol=0, rtol=0)
            predicted = query_model(self.backbone, self.f, initial, raw['distance'])
            expected = predicted['H']-torch.where(raw['hit'], predicted['log_hazard'], 0.)
            torch.testing.assert_close(row['distance_nll_sum'], expected.sum(), atol=0, rtol=0)
            self.assertGreater(float(cached['w'].max()), 3.)

    def test_start_derivative_is_used_and_exterior_latent_values_are_masked(self):
        q = torch.tensor([.1, .5, 1.], dtype=torch.float64)
        w = torch.ones_like(q)
        a, b = self.a.clone(), self.b.clone()
        a[2:] = torch.tensor([math.nan, -1e6])
        b[3] = 4.
        def predicted(initial):
            return cdf('mixture', self.head(query_points(self.backbone, self.f, initial, q)), w)
        torch.testing.assert_close(predicted(a), predicted(self.a), atol=0, rtol=0)
        self.assertGreater(float((predicted(b)-predicted(self.b)).abs().max().detach()), 1e-8)

    def test_prefix_consistency_including_recurrent_boundaries(self):
        for initial in (self.a, self.b):
            for cutoff in (.5, .65, 1.):
                q = torch.tensor([0., .125, .213, cutoff], dtype=torch.float64)
                full = self.head(query_points(self.backbone, self.f, initial, q))
                for f in (restrict_profile(self.f, cutoff/1.3), alter_suffix(self.f, cutoff/1.3)):
                    changed = self.head(query_points(self.backbone, f, initial, q))
                    torch.testing.assert_close(cdf('mixture', full, q+1), cdf('mixture', changed, q+1), atol=2e-7, rtol=0)

    def test_only_hit_labels_train_and_backbone_remains_frozen(self):
        raw = self.raw(self.b)
        row = cache_profile(self.backbone, raw)
        before = parameter_hash(self.backbone)
        batch = points([row], [torch.tensor([0, 3])])
        self.assertEqual(batch['samples'], 2)
        self.assertEqual(len(batch['w']), 1)
        value = loss(self.head, batch)
        value.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in self.head.parameters()))
        torch.optim.SGD(self.head.parameters(), lr=.01).step()
        self.assertEqual(before, parameter_hash(self.backbone))
        self.assertTrue(all(p.grad is None for p in self.backbone.parameters()))
        self.assertIsNone(loss(self.head, points([row], [torch.tensor([3])])))
        altered = {**row, 'w': torch.tensor([.3, 1.2, 4.1, 999.], dtype=torch.float64)}
        torch.testing.assert_close(loss(self.head, points([row])), loss(self.head, points([altered])))
        for w in (0., -1., math.nan, math.inf):
            bad = raw['speed'].clone()
            bad[0] = w
            with self.assertRaisesRegex(ValueError, 'labels'):
                cache_profile(self.backbone, {**raw, 'speed': bad})

    def test_joint_nll_and_zero_hit_mode_accounting(self):
        row = cache_profile(self.backbone, self.raw(self.b))
        empty = {**row, 'mode': torch.tensor(0), 'hit': torch.zeros(4, dtype=torch.bool),
                 'distance_nll_sum': torch.tensor(2., dtype=torch.float64)}
        report = evaluate_profiles(self.head, [row, empty])
        batch = points([row])
        gradient_sum = -log_prob('mixture', self.head(batch), batch['w']).sum()
        expected = (row['distance_nll_sum']+2.+gradient_sum)/8
        self.assertAlmostEqual(report['overall']['joint_nll'], float(expected.detach()), places=12)
        self.assertEqual(report['overall']['hits'], 3)
        self.assertEqual(report['overall']['samples'], 8)
        self.assertIsNone(report['by_mode']['positive_exterior']['speed_nll'])
        self.assertIsNone(report['by_mode']['positive_exterior']['pit_ks'])
        self.assertEqual(sum(report['overall']['pit_counts']), 3)

    def test_positive_distribution_normalization_and_sampling(self):
        grid = torch.linspace(0, 30., 30001, dtype=torch.float64)
        context = query_points(self.backbone, self.f, self.b, torch.full_like(grid, .5))
        params = self.head(context)
        density = log_prob('mixture', params, grid).exp()
        self.assertAlmostEqual(float(torch.trapezoid(density, grid).detach()), 1., delta=.001)
        self.assertAlmostEqual(float(cdf('mixture', params, grid)[-1].detach()), 1., places=10)
        uniforms = torch.rand((len(grid), 2), dtype=torch.float64).clamp(1e-8, 1-1e-8)
        w = sample('mixture', params, uniforms)
        self.assertTrue(torch.isfinite(w).all() and (w > 0).all())
        pit = cdf('mixture', params, w)
        self.assertLess(abs(float(pit.mean().detach())-.5), .01)

    def test_shuffled_packed_speed_labels_keep_profile_association(self):
        with tempfile.TemporaryDirectory() as temp:
            data = fixture(Path(temp)/'data.pt')
            order = torch.randperm(len(data['hit']))
            for key in ('sample_profile', 'distance', 'hit', 'speed'):
                data[key] = data[key][order]
            rows = raw_profiles(ProfileDataset(data), 'train')
            for raw in rows:
                wanted = (.3+raw['distance']+.2*raw['profile']).float()
                torch.testing.assert_close(raw['speed'][raw['hit']], wanted[raw['hit']])
            self.assertEqual([int(p['profile']) for p in raw_profiles(ProfileDataset(data), 'test')], [4, 5])

    def test_training_resume_and_self_contained_joint_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temp)
            dataset = root/'data.pt'
            fixture(dataset)
            config_root = Path(__file__).resolve().parents[1]/'configs'
            distance_spec = json.loads((config_root/'renewal_shader_ball_sigma002_r05_residual.json').read_text())
            distance_spec.update(dataset=str(dataset), dataset_sha256=digest(dataset), output_directory=str(root/'distance'),
                                 device='cpu', epochs=2, validation_every=1, profiles_per_batch=2,
                                 samples_per_profile_per_epoch=5, curve_points=17,
                                 model={'hidden': 4, 'embedding': 4, 'width': 8})
            config = root/'config.json'
            write_json(config, distance_spec)
            distance = train_distance(config)
            before, _ = load_model(distance['checkpoint'])
            original_hash = parameter_hash(before)
            spec = json.loads((config_root/'renewal_shader_ball_sigma002_r05_gradient.json').read_text())
            spec.update(dataset=str(dataset), dataset_sha256=digest(dataset), backbone_checkpoint=distance['checkpoint'],
                        output_directory=str(root/'full'), device='cpu', epochs=3, validation_every=1,
                        profiles_per_batch=2, samples_per_profile_per_epoch=5, head={'width': 8, 'size': 4})
            write_json(config, spec)
            full = train(config)
            self.assertTrue(full['backbone_unchanged'])
            self.assertFalse(full['test_metrics_used'])
            spec['output_directory'] = str(root/'resumed')
            write_json(config, spec)

            def stop(path, value):
                write_json(path, value)
                if path.name == 'history.json' and len(value) == 1:
                    raise RuntimeError('simulated interruption')

            with mock.patch('renewal.causal_gradient_train.write_json', side_effect=stop):
                with self.assertRaisesRegex(RuntimeError, 'simulated interruption'):
                    train(config)
            resumed = train(config, resume=True)
            backbone, head, _ = load_joint(resumed['checkpoint'])
            _, expected_head, _ = load_joint(full['checkpoint'])
            self.assertEqual(parameter_hash(backbone), original_hash)
            for key, value in head.state_dict().items():
                torch.testing.assert_close(value, expected_head.state_dict()[key], atol=0, rtol=0)
            q = torch.linspace(0, 1.3, 13, dtype=torch.float64)
            for initial in (self.a, self.b):
                torch.testing.assert_close(query_model(backbone, self.f, initial, q)['T'],
                                           query_model(before, self.f, initial, q)['T'], atol=0, rtol=0)
            # Loading/evaluation must use the embedded backbone, not an external file.
            Path(distance['checkpoint']).unlink()
            report = evaluate(resumed['checkpoint'], dataset, root/'test.json')
            self.assertEqual(report['overall']['profiles'], 2)
            self.assertGreater(report['by_mode']['positive_exterior']['hits'], 0)
            self.assertGreater(report['by_mode']['collision_state']['hits'], 0)


if __name__ == '__main__':
    unittest.main()
