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
from first_hit_se.causal_residual import CausalConfig
from first_hit_se.distance_study import alter_suffix, restrict_profile
from renewal import SE_KERNEL
from renewal.causal_data import ProfileDataset, batch_profiles, geometry, prepare_profile, query_model
from renewal.causal_residual import RenewalResidual, load_model, objective, prediction
from renewal.causal_train import evaluate, train
from renewal.collect import digest, write_json
from renewal.data import FEATURES


def fixture(path):
    features, initial, ids, distances, hits = [], [], [], [], []
    for p in range(6):
        b = .1*p
        features.append([b, b+.4, .4, .4, 0.])
        initial.append([0., b, 0., 0.] if p % 2 == 0 else [1., b, -b, .8])
        n = 3+p % 3
        ids.extend([p]*n)
        distances.extend(torch.linspace(.15, 1., n).double().tolist())
        hits.extend([True]*(n-1)+[False])
    data = {'format_version': 1, 'kernel': SE_KERNEL, 'feature_names': list(FEATURES),
            'features': torch.tensor(features, dtype=torch.float64), 'offsets': torch.arange(7),
            'initial': torch.tensor(initial, dtype=torch.float64), 'horizon': torch.ones(6, dtype=torch.float64),
            'split': torch.tensor([0, 0, 1, 1, 2, 2]), 'sample_profile': torch.tensor(ids),
            'distance': torch.tensor(distances, dtype=torch.float64), 'hit': torch.tensor(hits)}
    torch.save(data, path)
    return data


class ConditionalResidualTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)
        self.config = CausalConfig(hidden=4, embedding=4, width=8)
        self.model = RenewalResidual(self.config)
        self.f = torch.tensor([[.6, -.4, -1., -.5, math.log(.7)],
                               [-.4, .8, .3, 1., math.log(.6)]], dtype=torch.float64)
        self.a = torch.tensor([0., .6, 0., 0.], dtype=torch.float64)
        self.b = torch.tensor([1., .6, -.6, 2.], dtype=torch.float64)

    def raw(self, initial, n=4):
        return {'features': self.f, 'initial': initial, 'horizon': torch.tensor(1.3, dtype=torch.float64),
                'distance': torch.linspace(.2, 1.3, n, dtype=torch.float64),
                'hit': torch.tensor([True]*(n-1)+[False]), 'profile': torch.tensor(int(initial[0]))}

    def randomize(self):
        with torch.no_grad():
            self.model.head[-1].weight.normal_(0, .3)

    def test_positive_condition_is_not_a_sign_mixture(self):
        b = 1.
        f = torch.tensor([[b, b, 0., 0., 0.]], dtype=torch.float64)
        initial = torch.tensor([0., b, 0., 0.], dtype=torch.float64)
        q = torch.linspace(0, 1., 101, dtype=torch.float64)
        rate = math.exp(-b*b/2)/(2*math.pi*(.5+.5*math.erf(b/math.sqrt(2))))
        result = query_model(self.model, f, initial, q)
        torch.testing.assert_close(result['T'], (-q*rate).exp(), atol=1e-14, rtol=1e-14)
        geo = geometry(f, initial, self.config)
        self.assertEqual(geo['tokens'].shape[0], 1)
        self.assertNotIn('log_prior', geo)

    def test_surface_derivative_is_used_and_unknown_exterior_values_are_masked(self):
        self.randomize()
        q = torch.linspace(0, 1.3, 101, dtype=torch.float64)
        changed_a, changed_b = self.a.clone(), self.b.clone()
        changed_a[2:] = torch.tensor([float('nan'), 1000.])
        changed_b[3] = 4.  # Inference is not clipped to the training interval.
        original_a = query_model(self.model, self.f, self.a, q)['T']
        torch.testing.assert_close(original_a, query_model(self.model, self.f, changed_a, q)['T'], atol=0, rtol=0)
        original_b = query_model(self.model, self.f, self.b, q)['T']
        self.assertGreater(float((original_b-query_model(self.model, self.f, changed_b, q)['T']).abs().max()), 1e-7)
        self.assertGreater(float((original_a-original_b).abs().max()), 1e-7)
        entry = torch.stack((self.a, self.b))[:, None].requires_grad_()
        derivative = torch.autograd.grad(self.model.initial_state(entry).square().sum(), entry)[0]
        self.assertEqual(float(derivative[0, :, 2:].abs().max()), 0.)
        self.assertGreater(float(derivative[1, :, 3].abs().max()), 0.)

    def test_invalid_surface_conditions_and_negative_branch_rejected(self):
        for slot, value in ((0, -1), (2, .6), (3, -5), (3, float('nan'))):
            initial = self.b.clone()
            initial[slot] = value
            with self.assertRaises(ValueError):
                geometry(self.f, initial, self.config)
        zero_outward = self.b.clone()
        zero_outward[3] = -self.f[0, 2]/self.f[0, 4].exp()
        with self.assertRaises(ValueError):
            geometry(self.f, zero_outward, self.config)

    def test_censored_likelihood_variable_counts_and_full_curve_targets(self):
        rows = [prepare_profile(self.raw(self.a, 3), self.config, 33),
                prepare_profile(self.raw(self.b, 5), self.config, 33)]
        batch = batch_profiles(rows)
        p = prediction(self.model, batch)
        self.assertEqual(batch['valid'].sum().item(), 8)
        for i, n in enumerate((3, 5)):
            expected = p['H'][i, :n].clone()
            expected[:-1] -= p['log_hazard'][i, :n-1]
            torch.testing.assert_close(p['distance_nll'][i, :n], expected)
            self.assertAlmostEqual(float(batch['reference_T'][i, -1]), 1/n)
        weights = {'residual_l2_weight': .01, 'transmittance_weight': 100.}
        terms = objective(self.model, batch, weights)
        single = [objective(self.model, batch_profiles([row]), weights) for row in rows]
        for key in terms:
            torch.testing.assert_close(terms[key], sum(s[key] for s in single)/2)
        subset = batch_profiles(rows, [torch.tensor([0]), torch.tensor([0])])
        torch.testing.assert_close(subset['reference_T'], batch['reference_T'], atol=0, rtol=0)
        terms['loss'].backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in self.model.parameters()))

    def test_prefix_causality_and_density_consistency_for_both_modes(self):
        self.randomize()
        for initial in (self.a, self.b):
            q = torch.linspace(0, .65, 41, dtype=torch.float64)
            full = query_model(self.model, self.f, initial, q)['T']
            for f in (restrict_profile(self.f, .5), alter_suffix(self.f, .5)):
                torch.testing.assert_close(full, query_model(self.model, f, initial, q)['T'], atol=2e-7, rtol=0)
            q = torch.tensor([.213, .619, 1.119], dtype=torch.float64)
            p = query_model(self.model, self.f, initial, q)
            epsilon = 1e-5
            derivative = (query_model(self.model, self.f, initial, q-epsilon)['T']
                          - query_model(self.model, self.f, initial, q+epsilon)['T'])/(2*epsilon)
            torch.testing.assert_close((-p['H']+p['log_hazard']).exp(), derivative, atol=1e-7, rtol=1e-7)

    def test_adapter_preserves_splits_and_unsorted_label_associations(self):
        with tempfile.TemporaryDirectory() as temp:
            data = fixture(Path(temp)/'data.pt')
            expected = list(ProfileDataset(data).profiles('train'))
            order = torch.randperm(len(data['hit']))
            for key in ('sample_profile', 'distance', 'hit'):
                data[key] = data[key][order]
            actual = list(ProfileDataset(data).profiles('train'))
            self.assertEqual([int(p['profile']) for p in actual], [0, 1])
            self.assertEqual([int(p['profile']) for p in ProfileDataset(data).profiles('test')], [4, 5])
            for a, b in zip(actual, expected):
                index_a, index_b = a['distance'].argsort(), b['distance'].argsort()
                torch.testing.assert_close(a['distance'][index_a], b['distance'][index_b])
                torch.testing.assert_close(a['hit'][index_a], b['hit'][index_b])

    def test_training_resume_and_conditional_evaluation(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temp)
            dataset = root/'data.pt'
            fixture(dataset)
            spec = {'schema_version': 1, 'dataset': str(dataset), 'dataset_sha256': digest(dataset),
                    'output_directory': str(root/'full'), 'device': 'cpu', 'cpu_threads': 1,
                    'seed': 17, 'epochs': 3, 'patience': 4, 'profiles_per_batch': 2,
                    'samples_per_profile_per_epoch': 2, 'learning_rate': .001,
                    'minimum_learning_rate': .00002, 'weight_decay': .00001, 'gradient_clip': 1.,
                    'validation_every': 1, 'curve_points': 17,
                    'objective': {'residual_l2_weight': .01, 'transmittance_weight': 10.},
                    'model': {'hidden': 4, 'embedding': 4, 'width': 8}}
            config = root/'config.json'
            write_json(config, spec)
            full = train(config)
            self.assertFalse(full['test_metrics_used'])
            spec['output_directory'] = str(root/'resumed')
            write_json(config, spec)

            def stop_after_checkpoint(path, value):
                write_json(path, value)
                if path.name == 'history.json' and len(value) == 1:
                    raise RuntimeError('simulated interruption')

            with mock.patch('renewal.causal_train.write_json', side_effect=stop_after_checkpoint):
                with self.assertRaisesRegex(RuntimeError, 'simulated interruption'):
                    train(config)
            resumed = train(config, resume=True)
            full_model, _ = load_model(full['checkpoint'])
            resumed_model, _ = load_model(resumed['checkpoint'])
            for key, value in full_model.state_dict().items():
                torch.testing.assert_close(value, resumed_model.state_dict()[key], atol=0, rtol=0)
            report = evaluate(resumed['checkpoint'], dataset, root/'test.json')
            self.assertEqual(report['overall']['profiles'], 2)
            self.assertEqual(report['by_mode']['positive_exterior']['profiles'], 1)
            self.assertEqual(report['by_mode']['collision_state']['profiles'], 1)
            with self.assertRaisesRegex(ValueError, 'already exists'):
                train(config)
            self.assertEqual(train(config, resume=True)['epochs_completed'], 3)


if __name__ == '__main__':
    unittest.main()
