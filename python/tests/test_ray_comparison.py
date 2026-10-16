"""Independent quadrature checks for the analytic local comparison law."""
import unittest

import numpy as np
import torch

from first_hit_se.ray_comparison import local_speed_cdf, local_speed_log_prob


class LocalCrossingLawTests(unittest.TestCase):
    def test_zero_mean_is_rayleigh(self):
        w = torch.linspace(0, 8, 97, dtype=torch.float64)
        expected = -torch.expm1(-w.square()/2)
        torch.testing.assert_close(local_speed_cdf(w, w*0), expected, rtol=1e-11, atol=1e-12)

    def test_cdf_matches_integrated_density_and_normalizes(self):
        nodes, weights = [torch.from_numpy(x.copy()) for x in np.polynomial.legendre.leggauss(256)]
        for mu_value in (-50., -8., -1., 0., 1., 8., 50.):
            mu = torch.tensor(mu_value, dtype=torch.float64)
            maximum = max(mu_value, 0)+12
            for end in (maximum*.01, maximum*.1, maximum*.5, maximum):
                w = (nodes+1)*end/2
                integral = (local_speed_log_prob(w, mu).exp()*weights).sum()*end/2
                actual = local_speed_cdf(torch.tensor(end, dtype=torch.float64), mu)
                self.assertAlmostEqual(float(integral), float(actual), delta=2e-9)
            self.assertAlmostEqual(float(actual), 1., delta=2e-9)

    def test_support_monotonicity_and_extreme_directions(self):
        w = torch.linspace(-1, 110, 2001, dtype=torch.float64)[None]
        mu = torch.tensor([-100., -26., 0., 26., 100.], dtype=torch.float64)[:, None]
        values = local_speed_cdf(w, mu)
        self.assertTrue(torch.isfinite(values).all())
        self.assertTrue(((values >= 0) & (values <= 1)).all())
        self.assertTrue((values[:, 1:] >= values[:, :-1]-1e-12).all())
        self.assertTrue((values[:, w[0] <= 0] == 0).all())


if __name__ == '__main__':
    unittest.main()
