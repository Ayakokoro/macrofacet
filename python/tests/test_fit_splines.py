import unittest

import numpy as np

from macrofacet_fpt.fit_splines import fit_state_coefficients
from macrofacet_fpt.splines import ISplineBasis


class IndependentFitTest(unittest.TestCase):
    def test_recovers_a_smooth_empirical_curve(self) -> None:
        basis = ISplineBasis.create(5.0, basis_count=12)
        edges = np.linspace(0.0, 5.0, 51)
        delta_i = basis.delta_i(edges)
        truth = np.linspace(0.0, 1.2, basis.basis_count)
        delta_h = delta_i @ truth
        risks = np.empty(50, dtype=np.int64)
        events = np.empty(50, dtype=np.int64)
        remaining = 100000
        for index, probability in enumerate(-np.expm1(-delta_h)):
            risks[index] = remaining
            events[index] = int(round(remaining * probability))
            remaining -= events[index]
        fitted, _, _ = fit_state_coefficients(delta_i, risks, events)
        truth_survival = np.exp(-(basis.i_spline(edges[1:]) @ truth))
        fitted_survival = np.exp(-(basis.i_spline(edges[1:]) @ fitted))
        self.assertLess(np.max(np.abs(fitted_survival - truth_survival)), 2.0e-3)


if __name__ == "__main__":
    unittest.main()
