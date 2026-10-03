import unittest

import numpy as np

from macrofacet_fpt.splines import ISplineBasis


class ISplineBasisTest(unittest.TestCase):
    def test_basis_is_normalized_and_monotone(self) -> None:
        basis = ISplineBasis.create(8.0, basis_count=32, degree=2, knot_power=1.5)
        q = np.linspace(0.0, 8.0, 1001)
        m_values = basis.m_spline(q)
        i_values = basis.i_spline(q)
        self.assertEqual(m_values.shape, (1001, 32))
        self.assertEqual(i_values.shape, (1001, 32))
        self.assertTrue(np.all(m_values >= 0.0))
        np.testing.assert_allclose(i_values[0], 0.0, atol=1.0e-12)
        np.testing.assert_allclose(i_values[-1], 1.0, atol=1.0e-12)
        self.assertTrue(np.all(np.diff(i_values, axis=0) >= -1.0e-12))

    def test_interval_increments_telescope(self) -> None:
        basis = ISplineBasis.create(8.0)
        edges = np.linspace(0.0, 8.0, 201)
        increments = basis.delta_i(edges)
        self.assertTrue(np.all(increments >= 0.0))
        np.testing.assert_allclose(increments.sum(axis=0), 1.0, atol=1.0e-12)


if __name__ == "__main__":
    unittest.main()
