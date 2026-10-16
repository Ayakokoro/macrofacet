from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from first_hit_se.gradient_cdf import (GradientCDF, cdf, log_prob, inverse_log_ndtr, sample_mixture_components,
                                      spline_cdf, spline_log_prob, spline_quantile)
from renewal.model import mixture_cdf


class GradientCDFTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def parameters(self, count=1, random=False):
        torch.manual_seed(17)
        head = GradientCDF(4, 8, 8)
        if random:
            with torch.no_grad():
                head.net[-1].weight.normal_(0, .4)
                head.net[-1].bias.normal_(0, .6)
        z = torch.ones(count, 4)
        q = torch.full((count,), .4, dtype=torch.float64)
        one = torch.ones_like(q)
        return head, head(z, q, one, one, one*.3, one*-.8)

    def test_inverse_and_support_with_nonuniform_spline(self):
        _, params = self.parameters(401, True)
        u = torch.linspace(1e-8, 1-1e-8, 401, dtype=torch.float64)
        w = spline_quantile(params, u)
        self.assertTrue((w > 0).all() and (w[1:] > w[:-1]).all())
        torch.testing.assert_close(spline_cdf(params, w), u, atol=2e-13, rtol=2e-13)
        torch.testing.assert_close(spline_cdf(params, torch.zeros_like(w)), torch.zeros_like(w))
        self.assertTrue(torch.isneginf(spline_log_prob(params, -torch.ones_like(w))).all())

    def test_pdf_is_cdf_derivative_and_likelihood_backpropagates(self):
        head, params = self.parameters(37, True)
        w = torch.linspace(.01, 8., 37, dtype=torch.float64, requires_grad=True)
        F = spline_cdf(params, w)
        slope = torch.autograd.grad(F.sum(), w, retain_graph=True)[0]
        density = spline_log_prob(params, w).exp()
        torch.testing.assert_close(slope, density, atol=1e-12, rtol=1e-11)
        (-spline_log_prob(params, w).mean()).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters()))

    def test_density_integrates_to_one_and_tail_log_density_is_finite(self):
        _, params = self.parameters(random=True)
        limit = float(spline_quantile(params, torch.tensor([1-1e-9], dtype=torch.float64))[0].detach())
        w = torch.linspace(0, limit, 40001, dtype=torch.float64)
        integral = torch.trapezoid(spline_log_prob(params, w).exp(), w)
        self.assertAlmostEqual(float(integral.detach()), 1., places=5)
        self.assertTrue(torch.isfinite(spline_log_prob(params, torch.tensor([1e6], dtype=torch.float64))).all())

    def test_normal_inverse_and_single_component_sampler_handle_underflow_tails(self):
        logp = torch.tensor([-1., -100., -800., -50000.], dtype=torch.float64)
        torch.testing.assert_close(torch.special.log_ndtr(inverse_log_ndtr(logp)), logp, atol=1e-8, rtol=1e-12)
        u = torch.tensor([1e-5, .1, .5, .9, 1-1e-5], dtype=torch.float64)
        for mean in (-40., -3., 2.):
            params = (torch.zeros(len(u), 1, dtype=torch.float64),
                      torch.full((len(u), 1), mean, dtype=torch.float64),
                      torch.ones(len(u), 1, dtype=torch.float64))
            values = sample_mixture_components(params, torch.full_like(u, .5), u)
            torch.testing.assert_close(mixture_cdf(*params, values), u, atol=2e-10, rtol=2e-10)

    def test_component_sampling_matches_mixture_cdf(self):
        n = 40000
        generator = torch.Generator().manual_seed(58)
        uniforms = torch.rand(n, 2, dtype=torch.float64, generator=generator)
        params = (torch.tensor([[.2, .8]], dtype=torch.float64).log().expand(n, -1),
                  torch.tensor([[-2., 3.]], dtype=torch.float64).expand(n, -1),
                  torch.tensor([[.5, 1.]], dtype=torch.float64).expand(n, -1))
        draws = sample_mixture_components(params, uniforms[:, 0], uniforms[:, 1]).sort().values
        q = torch.linspace(0, 7, 50, dtype=torch.float64)
        expected = mixture_cdf(*(p[:1].expand(len(q), -1) for p in params), q)
        observed = torch.searchsorted(draws, q, right=True).double()/n
        self.assertLess(float((observed-expected).abs().max()), .012)

    def test_float32_mixture_weights_remain_normalized_in_probability_arithmetic(self):
        params = (torch.tensor([[-.7, -.7]], dtype=torch.float32),
                  torch.tensor([[.4, 1.1]], dtype=torch.float32),
                  torch.tensor([[.6, .7]], dtype=torch.float32))
        w = torch.linspace(1e-8, 16, 32001, dtype=torch.float64)
        density = log_prob('mixture', params, w).exp()
        self.assertAlmostEqual(float(torch.trapezoid(density, w)), 1., places=6)
        tail = cdf('mixture', params, w[-1:])
        self.assertTrue((tail <= 1).all())
        torch.testing.assert_close(tail, torch.ones_like(tail), atol=1e-14, rtol=0)


if __name__ == '__main__':
    unittest.main()
