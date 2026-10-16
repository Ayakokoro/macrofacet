import math
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from renewal.profiles import subdivide_features
from first_hit_se.causal_data import batch_profiles, geometry, prepare_profile, query_model
from first_hit_se.causal_residual import CausalConfig, CausalResidual, objective, prediction
from first_hit_se.distance_study import alter_suffix, restrict_profile


class CausalResidualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(17)
        self.config = CausalConfig(hidden=4, embedding=4, width=8)
        self.model = CausalResidual(self.config)
        self.f = torch.tensor([[.6, -.4, -1., -.5, math.log(.7)],
                               [-.4, .8, .3, 1., math.log(.6)]], dtype=torch.float64)
        self.raw = {'features': self.f, 'distance': torch.tensor([.2, .8, 1.3, 1.3], dtype=torch.float64),
            'hit': torch.tensor([True, True, True, False]), 'initial_sign': torch.tensor([1., -1., 1., 1.])}
        self.row = prepare_profile(self.raw, self.config, 33)
        self.batch = batch_profiles([self.row])

    def randomize(self):
        with torch.no_grad():
            self.model.head[-1].weight.normal_(0, .3)

    def test_zero_residual_recovers_local_and_constant_hazard(self):
        f = torch.tensor([[0., 0., 0., 0., math.log(1.3)]], dtype=torch.float64)
        q = torch.linspace(0, 1.3, 91, dtype=torch.float64)
        predicted = query_model(self.model, f, q)
        torch.testing.assert_close(predicted['T'], torch.exp(-q/math.pi), atol=1e-14, rtol=1e-14)
        torch.testing.assert_close(predicted['T'], query_model(None, f, q, self.config)['T'], atol=0, rtol=0)
        with torch.no_grad():
            self.model.head[-1].bias.fill_(.5)
        torch.testing.assert_close(query_model(self.model, f, q)['T'], torch.exp(-math.exp(.5)*q/math.pi), atol=1e-14, rtol=1e-14)

    def test_partial_prefix_future_change_and_repartition(self):
        self.randomize()
        for fraction in (.17, .5, .75):
            q = torch.linspace(0, 1.3*fraction, 57, dtype=torch.float64)
            full = query_model(self.model, self.f, q)['T']
            cut = query_model(self.model, restrict_profile(self.f, fraction), q)['T']
            future = query_model(self.model, alter_suffix(self.f, fraction), q)['T']
            torch.testing.assert_close(full, cut, atol=2e-7, rtol=0)
            torch.testing.assert_close(full, future, atol=2e-7, rtol=0)
        subdivided = subdivide_features(self.f, torch.full((2,), 4, dtype=torch.long))
        q = torch.linspace(0, 1.3, 101, dtype=torch.float64)
        torch.testing.assert_close(query_model(self.model, self.f, q)['T'],
                                   query_model(self.model, subdivided, q)['T'], atol=2e-7, rtol=0)

    def test_future_tokens_have_exactly_zero_prefix_gradient(self):
        self.randomize()
        self.batch['tokens'].requires_grad_()
        delta = self.model(self.batch)
        gradient = torch.autograd.grad(delta[:, :, 3].sum(), self.batch['tokens'])[0]
        self.assertEqual(float(gradient[:, :, 3:].abs().max()), 0.)
        self.assertGreater(float(gradient[:, :, :3].abs().max()), 0.)

    def test_streaming_state_matches_batched_gru(self):
        self.randomize()
        geo = geometry(self.f, self.config)
        state = self.model.initial_state(geo['entry'])
        deltas = []
        for j in range(len(geo['widths'])):
            deltas.append(self.model.correction(state, geo['left'][:, j]))
            state = self.model.advance(state, geo['tokens'][:, j])
        torch.testing.assert_close(torch.stack(deltas, -1), self.model(self.batch)[0], atol=2e-7, rtol=0)

    def test_padding_and_sign_symmetry(self):
        self.randomize()
        longer = {**self.raw, 'features': alter_suffix(self.f, .5)}
        pair = batch_profiles([self.row, prepare_profile(longer, self.config, 33)])
        torch.testing.assert_close(prediction(self.model, pair)['T'][0],
                                   prediction(self.model, self.batch)['T'][0], atol=2e-7, rtol=0)
        short = {**self.raw, 'features': restrict_profile(self.f, .5),
            'distance': torch.tensor([.1, .3, .65, .65], dtype=torch.float64)}
        short_row = prepare_profile(short, self.config, 33)
        mixed = prediction(self.model, batch_profiles([short_row, self.row]))['T'][0]
        torch.testing.assert_close(mixed, prediction(self.model, batch_profiles([short_row]))['T'][0], atol=2e-7, rtol=0)
        negated = self.f.clone()
        negated[:, :4] *= -1
        q = torch.linspace(0, 1.3, 71, dtype=torch.float64)
        torch.testing.assert_close(query_model(self.model, self.f, q)['T'], query_model(self.model, negated, q)['T'], atol=1e-14, rtol=1e-14)

    def test_full_targets_and_width_weighted_regularizer(self):
        with torch.no_grad():
            self.model.head[-1].bias.fill_(.5)
        weights = {'residual_l2_weight': .01, 'transmittance_weight': 100.}
        terms = objective(self.model, self.batch, weights)
        self.assertAlmostEqual(float(terms['residual_l2'].detach()), .25, places=13)
        subsample = batch_profiles([self.row], [torch.tensor([0])])
        self.assertEqual(float(subsample['reference_T'][0, -1]), .25)
        torch.testing.assert_close(objective(self.model, subsample, weights)['transmittance_mse'], terms['transmittance_mse'], atol=0, rtol=0)
        terms['loss'].backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in self.model.parameters()))

    def test_local_head_ignores_completed_segments_but_gru_uses_them(self):
        self.randomize()
        altered = {**self.batch, 'tokens': self.batch['tokens']+3.}
        self.assertGreater(float((self.model(self.batch)-self.model(altered)).abs().max().detach()), 1e-4)
        self.model.config.history = False
        torch.testing.assert_close(self.model(self.batch), self.model(altered), atol=0, rtol=0)

    def test_density_matches_survival_derivative_away_from_boundaries(self):
        self.randomize()
        q = torch.tensor([.213, .619, 1.119], dtype=torch.float64)
        p = query_model(self.model, self.f, q)
        prior = self.row['log_prior']
        density = torch.logsumexp(prior[:, None]-p['H']+p['log_hazard'], 0).exp()
        epsilon = 1e-5
        derivative = (query_model(self.model, self.f, q-epsilon)['T']-
                      query_model(self.model, self.f, q+epsilon)['T'])/(2*epsilon)
        torch.testing.assert_close(density, derivative, atol=1e-7, rtol=1e-7)


if __name__ == '__main__':
    unittest.main()
