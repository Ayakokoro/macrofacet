import math
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from first_hit_se.causal_data import query_model
from first_hit_se.causal_gradient import CausalGradient, GradientConfig
from first_hit_se.causal_residual import CausalConfig, CausalResidual
from first_hit_se.causal_sampling import sample_profile


class CausalSamplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.backbone = CausalResidual(CausalConfig(hidden=4, embedding=4, width=8)).eval()
        self.head = CausalGradient(GradientConfig(hidden=4, width=8, size=2), 11).eval()
        with torch.no_grad():
            for p in self.head.parameters():
                p.zero_()
        self.features = torch.tensor([[0., 0., 0., 0., math.log(2.03)]], dtype=torch.float64)
        self.uniforms = torch.rand((2048, 4), generator=torch.Generator().manual_seed(81), dtype=torch.float64)

    def test_zero_residual_recovers_analytic_exponential_and_half_normal(self):
        result = sample_profile(self.backbone, self.head, self.features, self.uniforms)
        q = -math.pi*self.uniforms[:, 1].log()
        hit = q < 2.03
        self.assertTrue(torch.equal(result['hit'], hit))
        self.assertTrue(torch.equal(result['initial_sign'], 1-2*(self.uniforms[:, 0] >= .5).long()))
        torch.testing.assert_close(result['q'], q.clamp_max(2.03), atol=1e-9, rtol=0)
        scale = float(torch.nn.functional.softplus(torch.tensor(0.))+.01)
        speed = scale*torch.special.ndtri((1+self.uniforms[hit, 3])/2)
        torch.testing.assert_close(result['speed'][hit], speed, atol=1e-11, rtol=1e-11)
        self.assertTrue((result['speed'][~hit] == 0).all())

    def test_constant_log_residual_scales_integrated_hazard(self):
        with torch.no_grad():
            self.backbone.head[-1].bias.fill_(math.log(2))
        scale = math.exp(float(self.backbone.head[-1].bias.detach()))
        result = sample_profile(self.backbone, self.head, self.features, self.uniforms)
        expected = (-math.pi*self.uniforms[:, 1].log()/scale).clamp_max(2.03)
        torch.testing.assert_close(result['q'], expected, atol=1e-9, rtol=0)
        q = torch.tensor([0., .125, .7, 2.03], dtype=torch.float64)
        torch.testing.assert_close(query_model(self.backbone, self.features, q)['T'],
                                  torch.exp(-q*scale/math.pi), atol=2e-15, rtol=1e-14)

    def test_open_uniform_contract_and_short_interval(self):
        with self.assertRaises(ValueError):
            sample_profile(self.backbone, self.head, self.features, torch.zeros(1, 4))
        features = self.features.clone(); features[:, 4] = math.log(.03)
        uniforms = torch.tensor([[.2, .999, .3, .7], [.8, .01, .5, .5]], dtype=torch.float64)
        result = sample_profile(self.backbone, self.head, features, uniforms)
        self.assertEqual(result['hit'].tolist(), [True, False])
        torch.testing.assert_close(result['q'], (-math.pi*uniforms[:, 1].log()).clamp_max(.03), atol=1e-9, rtol=0)


if __name__ == '__main__':
    unittest.main()
