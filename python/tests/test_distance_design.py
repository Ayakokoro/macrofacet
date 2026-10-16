import math
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from first_hit_se.distance_models import (MODES, make_model, predictions, sample_distance)
from first_hit_se.distance_probe import GradientHead, probe_predictions
from first_hit_se.distance_study import alter_suffix, restrict_profile
from first_hit_se.hazard import local_bin_log_rates
from first_hit_se.spline import local_profile, resample_profile


class DistanceDesignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.config = {'model': {'profile_points': 16, 'bins': 32, 'controls': 8,
                                'hidden': 4, 'width': 8, 'quadrature_order': 8}, 'spline_bins': 8}
        self.f = torch.tensor([[.4, -.4, -1.2, -.7, 0.]], dtype=torch.float64)
        q = torch.tensor([[.2, .6, 1.]], dtype=torch.float64)
        b, db = local_profile(self.f, q[0])
        self.batch = {'fixed': resample_profile(self.f, 16)[None],
                      'baseline': local_bin_log_rates(self.f, 32)[None],
                      'distance': q, 'horizon': torch.ones(1, dtype=torch.float64),
                      'b0': self.f[:1, 0], 'hit': torch.tensor([[True, True, False]]),
                      'initial_sign': torch.tensor([[1., -1., -1.]]),
                      'speed': torch.tensor([[.4, 1.1, math.nan]], dtype=torch.float64),
                      'b': b[None], 'db': db[None]}

    def test_shared_encoder_and_no_gradient_head(self):
        models = [make_model(self.config, mode, 37) for mode in MODES]
        for other in models[1:]:
            for a, b in zip(models[0].encoder.parameters(), other.encoder.parameters()):
                torch.testing.assert_close(a, b, atol=0, rtol=0)
        self.assertFalse(any('mixture' in name for name, _ in models[-1].named_parameters()))

    def test_conditional_and_marginal_likelihood_match_single_queries(self):
        for mode in MODES:
            model = make_model(self.config, mode, 17)
            with torch.no_grad():
                last = model.cdf_head if mode == 'cdf' else model.hazard_head
                last.weight.normal_(0, .15)
            result = predictions(model, self.batch)
            single = model.query(self.f, self.batch['distance'][0])
            expected = torch.where(self.batch['hit'][0], -single['log_density'], -single['log_T'])
            torch.testing.assert_close(result['distance_nll'][0], expected, atol=1e-12, rtol=1e-12)
            branch = torch.tensor([0, 1, 1])
            H = single['H'][branch, torch.arange(3)]
            rate = single['log_hazard'][branch, torch.arange(3)]
            torch.testing.assert_close(result['training_nll'][0],
                                      H-torch.where(self.batch['hit'][0], rate, 0.))
            result['training_nll'].mean().backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                for p in model.parameters()))

    def test_inverse_matches_each_sign_cdf_and_escape(self):
        u = (torch.arange(20000, dtype=torch.float64)+.5)/20000
        q = torch.linspace(0, 1, 101, dtype=torch.float64)
        for mode in MODES:
            model = make_model(self.config, mode, 17)
            prediction = model.query(self.f, q)
            for k, sign in enumerate((1., -1.)):
                draws, hit = sample_distance(model, self.batch['fixed'], torch.tensor([sign]),
                    self.batch['baseline'][0, k:k+1], self.batch['horizon'], u[None])
                empirical = 1-torch.searchsorted(draws[0, hit[0]].sort().values, q, right=True).double()/len(u)
                torch.testing.assert_close(empirical, (-prediction['H'][k]).exp(), atol=1/len(u), rtol=0)

    def test_prefix_restriction_preserves_original_mean_and_derivative(self):
        for fraction in (.17, .5, .93):
            cut = restrict_profile(self.f, fraction)
            q = torch.linspace(0, fraction, 57, dtype=torch.float64)
            for a, b in zip(local_profile(cut, q), local_profile(self.f, q)):
                torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)

    def test_frozen_probe_censors_missing_speed_and_blocks_backbone_gradient(self):
        model = make_model(self.config, 'cdf', 17)
        head = GradientHead(4, 8, 2)
        loss, pit = probe_predictions(model, head, self.batch, frozen=True)
        self.assertTrue(torch.isfinite(loss).all())
        self.assertEqual(float(loss[0, -1].detach()), 0.)
        self.assertTrue(((pit[self.batch['hit']] >= 0) & (pit[self.batch['hit']] <= 1)).all())
        loss.mean().backward()
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in head.parameters()))

    def test_future_perturbation_preserves_prefix_and_horizon(self):
        altered = alter_suffix(self.f, .5)
        self.assertAlmostEqual(float(altered[:, 4].exp().sum()), 1.)
        q = torch.linspace(0, .5, 51, dtype=torch.float64)
        for a, b in zip(local_profile(altered, q), local_profile(self.f, q)):
            torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)
        q = torch.tensor([1.], dtype=torch.float64)
        self.assertAlmostEqual(float(local_profile(altered, q)[0]-local_profile(self.f, q)[0]), 2.)


if __name__ == '__main__':
    unittest.main()
