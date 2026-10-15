import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from first_hit_se import FORMAT, VERSION, EVENT, SE_KERNEL
from first_hit_se.spline import (SplineConfig, SplineFirstHitNetwork, spline_parameters,
                                 rational_quadratic, resample_profile)
from first_hit_se.spline_train import prepare_profiles, batch_rows, batch_losses
from first_hit_se.train import export_model
from renewal.profiles import subdivide_features


class SplineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def model(self):
        torch.manual_seed(137)
        model = SplineFirstHitNetwork(SplineConfig(profile_points=16, bins=8, hidden=4,
                                                   width=8, mixture_width=8, components=2))
        with torch.no_grad():
            model.cdf_head.weight.normal_(0, .3)
        return model

    def profile(self):
        return torch.tensor([[.4, -.4, -1.2, -.7, 0.]], dtype=torch.float64)

    def test_identity_endpoints_and_analytic_inverse(self):
        c = SplineConfig(bins=8)
        raw = torch.zeros(2, 3*c.bins+2, dtype=torch.float64)
        raw[:, 1+2*c.bins:] = math.log(math.expm1(1-c.minimum_derivative))
        params = spline_parameters(raw, c)
        u = torch.linspace(0, 1, 1001, dtype=torch.float64).expand(2, -1)
        forward, log_pdf, tail = rational_quadratic(u, *(p[:, None] for p in params[:3]))
        torch.testing.assert_close(forward, u, atol=1e-14, rtol=1e-14)
        torch.testing.assert_close(tail, 1-u, atol=1e-14, rtol=1e-14)
        torch.testing.assert_close(log_pdf, torch.zeros_like(u), atol=1e-14, rtol=0)
        torch.manual_seed(4)
        raw.normal_(0, 2)
        params = spline_parameters(raw, c)
        forward, log_pdf, _ = rational_quadratic(u, *(p[:, None] for p in params[:3]))
        inverse, log_inverse, _ = rational_quadratic(forward, *(p[:, None] for p in params[:3]), inverse=True)
        self.assertTrue((forward[:, 1:] >= forward[:, :-1]).all())
        torch.testing.assert_close(inverse, u, atol=2e-10, rtol=2e-10)
        torch.testing.assert_close(log_pdf+log_inverse, torch.zeros_like(u), atol=2e-8, rtol=0)

    def test_cdf_derivative_and_probability_mass(self):
        model = self.model().double()
        q = torch.linspace(0, 1, 10001, dtype=torch.float64, requires_grad=True)
        result = model.query(self.profile(), q, with_mixture=False)
        derivative = torch.autograd.grad((1-result['T']).sum(), q)[0]
        torch.testing.assert_close(derivative[1:-1], result['log_density'].exp()[1:-1], atol=1e-10, rtol=1e-9)
        self.assertLess(abs(float((torch.trapezoid(result['log_density'].exp(), q)+result['T'][-1]-1).detach())), 2e-6)
        self.assertAlmostEqual(float(result['T'][0].detach()), 1., places=14)

    def test_fixed_input_partition_invariance_and_sign_symmetry(self):
        model = self.model()
        f = self.profile()
        partition = subdivide_features(f, torch.tensor([19]))
        torch.testing.assert_close(resample_profile(f, 16), resample_profile(partition, 16), atol=1e-12, rtol=1e-12)
        opposite = f.clone(); opposite[:, :4] *= -1
        q = torch.tensor([1., 0., .7, .2, .7], dtype=torch.float64)
        a, b = model.query(f, q), model.query(opposite, q)
        torch.testing.assert_close(a['T'], b['T'])
        torch.testing.assert_close(a['means'], b['means'].flip(0))

    def test_censored_labels_batched_loss_and_gradients(self):
        model = self.model()
        p = {'features': self.profile(), 'horizon': 1.,
             'distance': torch.tensor([.25, .7, 1.], dtype=torch.float64),
             'hit': torch.tensor([True, True, False]), 'initial_sign': torch.tensor([1, -1, -1]),
             'speed': torch.tensor([.4, .8, math.nan], dtype=torch.float64)}
        batch = batch_rows(prepare_profiles([p, p], 16, 'cpu'))
        result = batch_losses(model, batch)
        expected = model.losses(p['features'], *(p[k] for k in ('distance', 'hit', 'initial_sign', 'speed')))
        for key in ('joint', 'complete_joint', 'distance', 'speed'):
            torch.testing.assert_close(result[key][0], expected[key], atol=2e-6, rtol=2e-6)
        result['complete_joint'].mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    @unittest.skipUnless(os.environ.get('MACROFACET_FIRST_HIT_EXE'), 'native executable not supplied')
    def test_cpp_query_and_inverse_sampling(self):
        model = self.model()
        q = torch.linspace(0, 1, 101, dtype=torch.float64)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            torch.save({'format': FORMAT, 'version': VERSION, 'event': EVENT, 'kernel': SE_KERNEL,
                        'model_config': model.configuration(), 'model_state': model.state_dict()}, root/'model.pt')
            export_model(root/'model.pt', root/'model.json')
            (root/'profile.json').write_text(json.dumps({'features': self.profile().tolist(), 'queries': q.tolist()}))
            subprocess.run([os.environ['MACROFACET_FIRST_HIT_EXE'], str(root/'model.json'), str(root/'profile.json'),
                            str(root/'result.json'), '8192'], check=True)
            result = json.loads((root/'result.json').read_text())
            expected = model.query(self.profile(), q)
            for key in ('H', 'log_hazard', 'hit_branch_probability'):
                actual = torch.tensor([r[key] for r in result['predictions']], dtype=torch.float64).T
                torch.testing.assert_close(actual, expected[key], atol=2e-5, rtol=2e-5)
            for key in ('weights', 'means', 'scales'):
                actual = torch.tensor([[m[key] for m in r['mixtures']] for r in result['predictions']], dtype=torch.float64).transpose(0, 1)
                truth = expected['log_weights'].exp().double() if key == 'weights' else expected[key].double()
                torch.testing.assert_close(actual, truth, atol=2e-5, rtol=2e-5)
            draws = result['samples']
            empirical = torch.tensor([sum(not s['hit'] or s['q'] > x for s in draws)/len(draws) for x in q], dtype=torch.float64)
            self.assertLess(float((empirical-expected['T']).abs().max().detach()), .025)
            self.assertTrue(all(s['speed'] > 0 and 0 < s['q'] <= 1 for s in draws if s['hit']))
            self.assertEqual({s['initial_sign'] for s in draws if s['hit']}, {-1, 1})


if __name__ == '__main__':
    unittest.main()
