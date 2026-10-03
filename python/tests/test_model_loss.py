import unittest

import numpy as np
import torch

from macrofacet_fpt.inference import invert_cumulative_hazard
from macrofacet_fpt.loss import interval_nll
from macrofacet_fpt.model import CoefficientNet
from macrofacet_fpt.splines import ISplineBasis


class ModelAndLossTest(unittest.TestCase):
    def test_coefficients_and_likelihood_are_well_behaved(self) -> None:
        model = CoefficientNet(np.zeros(3), np.ones(3))
        beta = torch.tensor([[0.0, 1.0, 0.5], [1.0, -1.0, 2.0]])
        coefficients = model(beta)
        self.assertEqual(tuple(coefficients.shape), (2, 32))
        torch.testing.assert_close(coefficients[:, 0], torch.zeros(2))
        self.assertTrue(bool(torch.all(coefficients[:, 1:] > 0.0)))

        basis = ISplineBasis.create(2.0)
        delta_i = torch.tensor(basis.delta_i(np.linspace(0.0, 2.0, 5)), dtype=torch.float32)
        risks = torch.tensor([[16, 15, 13, 12], [16, 16, 14, 10]], dtype=torch.float32)
        events = torch.tensor([[1, 2, 1, 0], [0, 2, 4, 1]], dtype=torch.float32)
        loss = interval_nll(coefficients, delta_i, risks, events)
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_cumulative_hazard_inverse(self) -> None:
        basis = ISplineBasis.create(4.0, basis_count=8)
        coefficients = np.linspace(0.0, 2.0, basis.basis_count)
        q = np.asarray([0.2, 1.0, 3.5])
        target = basis.i_spline(q) @ coefficients
        recovered = invert_cumulative_hazard(coefficients, basis, target)
        np.testing.assert_allclose(recovered, q, atol=1.0e-8)
        self.assertTrue(np.isinf(invert_cumulative_hazard(coefficients, basis, target[-1] + 100.0)))

    def test_model_is_torchscript_exportable(self) -> None:
        model = CoefficientNet(np.zeros(3), np.ones(3)).eval()
        scripted = torch.jit.script(model)
        beta = torch.tensor([[0.1, -0.2, 1.5]])
        torch.testing.assert_close(scripted(beta), model(beta))


if __name__ == "__main__":
    unittest.main()
