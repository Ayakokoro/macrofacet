import math
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.integrate import quad
from scipy.special import log_ndtr
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from first_hit_se.hazard import (HazardConfig, HazardNetwork, batch_predictions,
    inverse_piecewise, local_bin_log_rates, local_log_hazard,
    log_negative_part_mean, piecewise_distribution)
from first_hit_se.spline import resample_profile
from renewal.profiles import subdivide_features


class HazardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def profile(self):
        return torch.tensor([[.4, -.4, -1.2, -.7, 0.]], dtype=torch.float64)

    def model(self, mode):
        torch.manual_seed(137)
        return HazardNetwork(HazardConfig(mode=mode, profile_points=16, bins=32,
                                         controls=8, hidden=4, width=8))

    def test_local_rate_against_integrated_gaussian_flux(self):
        for slope in (-40., -3., 0., 3., 8., 15., 26., 40.):
            # Factoring out phi(slope) keeps the reference integral meaningful
            # even when the probability itself is below floating-point range.
            if slope >= 0:
                integral = quad(lambda w: w*np.exp(-slope*w-.5*w*w), 0, np.inf,
                                epsabs=1e-14, epsrel=1e-12)[0]
                expected = -.5*slope*slope-.5*math.log(2*math.pi)+math.log(integral)
            else:
                area = quad(lambda w: w*np.exp(-.5*(w+slope)**2)/math.sqrt(2*math.pi),
                            0, max(12., -slope+12), epsabs=1e-12)[0]
                expected = math.log(area)
            actual = float(log_negative_part_mean(torch.tensor(slope, dtype=torch.float64)))
            self.assertAlmostEqual(actual, expected, delta=2e-10)
            for b in (-30., 0., 30.):
                truth = -.5*b*b-.5*math.log(2*math.pi)-log_ndtr(b)+expected
                rate = float(local_log_hazard(torch.tensor(b), torch.tensor(slope)))
                self.assertAlmostEqual(rate, truth, delta=2e-10)

    def test_inverse_sampling_matches_survival_including_escape(self):
        rates = torch.tensor([[.1, 4., .01, .8], [3., .02, .3, .1]], dtype=torch.float64).log()
        horizon = torch.tensor([1., 2.], dtype=torch.float64)
        rng = torch.Generator().manual_seed(28)
        uniforms = torch.rand(2, 60000, dtype=torch.float64, generator=rng)
        q, hit = inverse_piecewise(rates, horizon, uniforms)
        grid = torch.linspace(0, 1, 51, dtype=torch.float64)[None]*horizon[:, None]
        H, _ = piecewise_distribution(rates, grid, horizon)
        empirical = torch.stack([torch.stack([((~hit[j]) | (q[j] > t)).double().mean()
                                              for t in grid[j]]) for j in range(2)])
        self.assertLess(float((empirical-(-H).exp()).abs().max()), .008)
        sampled_H, _ = piecewise_distribution(rates, q, horizon)
        torch.testing.assert_close(sampled_H[hit], -torch.log1p(-uniforms[hit]), atol=1e-13, rtol=1e-13)
        self.assertTrue(((q >= 0) & (q <= horizon[:, None])).all())

    def test_quadrature_respects_discontinuous_mean_derivatives(self):
        widths, values = [.371, .629], [.3, .2, .8]
        f = torch.tensor([[values[i], values[i+1], values[i+1]-values[i], values[i+1]-values[i],
                           math.log(widths[i])] for i in range(2)], dtype=torch.float64)
        rates = local_bin_log_rates(f, bins=7, quadrature_order=16)
        q = torch.linspace(0, 1, 8, dtype=torch.float64)
        H, _ = piecewise_distribution(rates, q[None], 1.)
        # Each affine piece has an independent closed-form integrated rate:
        # A(a)/a * [log Phi(b_end)-log Phi(b_start)]. The jump is not on a bin edge.
        for branch, sign in enumerate((1., -1.)):
            expected = []
            for end in q.tolist():
                total, left = 0., 0.
                for i, width in enumerate(widths):
                    length = min(max(end-left, 0.), width)
                    slope = sign*(values[i+1]-values[i])/width
                    b0 = sign*values[i]
                    area = math.exp(float(log_negative_part_mean(torch.tensor(slope, dtype=torch.float64))))
                    total += area/slope*(log_ndtr(b0+slope*length)-log_ndtr(b0))
                    left += width
                expected.append(total)
            torch.testing.assert_close(H[branch], torch.tensor(expected, dtype=torch.float64), atol=1e-12, rtol=1e-12)

    def test_zero_residual_and_paired_parameters(self):
        direct, residual = self.model('direct'), self.model('residual')
        for x, y in zip(direct.parameters(), residual.parameters()):
            torch.testing.assert_close(x, y, atol=0, rtol=0)
        f = self.profile()
        fixed = resample_profile(f, 16).expand(2, -1)
        signs = torch.tensor([1., -1.])
        baseline = local_bin_log_rates(f, 32)
        torch.testing.assert_close(residual.log_rates(fixed, signs, baseline), baseline, atol=0, rtol=0)
        torch.testing.assert_close(direct.log_rates(fixed, signs), torch.zeros_like(baseline), atol=0, rtol=0)

    def test_sign_symmetry_partition_invariance_and_density(self):
        f = self.profile()
        partition = subdivide_features(f, torch.tensor([19]))
        torch.testing.assert_close(local_bin_log_rates(f, 32), local_bin_log_rates(partition, 32),
                                   atol=1e-12, rtol=1e-12)
        opposite = f.clone(); opposite[:, :4] *= -1
        for mode in ('direct', 'residual'):
            model = self.model(mode).double()
            with torch.no_grad():
                model.hazard_head.weight.normal_(0, .3)
            q = torch.linspace(.002, .998, 103, dtype=torch.float64, requires_grad=True)
            a, b = model.query(f, q), model.query(opposite, q)
            torch.testing.assert_close(a['T'], b['T'], atol=1e-14, rtol=1e-14)
            derivative = torch.autograd.grad((1-a['T']).sum(), q)[0]
            torch.testing.assert_close(derivative, a['log_density'].exp(), atol=1e-11, rtol=1e-11)
            endpoints = model.query(f, torch.tensor([0., 1.], dtype=torch.float64))
            self.assertAlmostEqual(float(endpoints['T'][0].detach()), 1., places=14)

    def test_censored_batch_likelihood_and_training_gradients(self):
        f = self.profile()
        batch = {'fixed': resample_profile(f, 16)[None].expand(2, -1),
                 'baseline': local_bin_log_rates(f, 32)[None].expand(2, -1, -1),
                 'distance': torch.tensor([[.25, .7, 1.]]*2, dtype=torch.float64),
                 'hit': torch.tensor([[True, True, False]]*2),
                 'initial_sign': torch.tensor([[1., -1., -1.]]*2),
                 'horizon': torch.ones(2, dtype=torch.float64), 'b0': torch.tensor([.4, .4], dtype=torch.float64)}
        model = self.model('direct')
        result = batch_predictions(model, batch)
        torch.testing.assert_close(result['training_nll'], batch['distance'])
        torch.testing.assert_close(result['distance_nll'], batch['distance'])
        result['training_nll'].mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertGreater(float(model.hazard_head.bias.grad.abs().sum()), 0.)
        for mode in ('direct', 'residual'):
            model = self.model(mode)
            with torch.no_grad():
                model.hazard_head.weight.normal_(0, .1)
            batched = batch_predictions(model, batch)
            single = model.query(f, batch['distance'][0])
            expected = torch.where(batch['hit'][0], -single['log_density'], -single['log_T'])
            torch.testing.assert_close(batched['distance_nll'][0], expected, atol=2e-7, rtol=2e-7)


if __name__ == '__main__':
    unittest.main()
