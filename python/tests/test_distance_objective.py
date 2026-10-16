import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from first_hit_se.distance_models import make_model, predictions, transmittance
from first_hit_se.distance_objective import distance_objective, objective_weights
from first_hit_se.hazard_study import make_batch, prepare
from first_hit_se.residual_supervision import choose_candidate, ray_metrics


class DistanceObjectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.config = {'curve_points': 5, 'model': {'profile_points': 16, 'bins': 32,
            'controls': 8, 'hidden': 4, 'width': 8, 'quadrature_order': 8}}
        self.raw = {'features': torch.tensor([[.4, -.4, -1.2, -.7, 0.]], dtype=torch.float64),
            'horizon': 1., 'split': 'train', 'distance': torch.tensor([.2, .5, 1., 1.], dtype=torch.float64),
            'hit': torch.tensor([True, True, True, False]),
            'initial_sign': torch.tensor([1., -1., 1., 1.], dtype=torch.float64)}
        self.profiles = prepare({'profiles': [self.raw]}, self.config, 'cpu')['train']
        self.batch = make_batch(self.profiles)
        self.model = make_model(self.config, 'residual', 41)

    def test_disabled_objective_exactly_preserves_original_loss_and_gradients(self):
        with torch.no_grad():
            self.model.hazard_head.weight.normal_(0, .2)
        p = predictions(self.model, self.batch)
        original = p['training_nll'].mean()
        revised = distance_objective(self.model, self.batch, p, objective_weights())['loss']
        torch.testing.assert_close(original, revised, atol=0, rtol=0)
        a = torch.autograd.grad(original, self.model.parameters(), retain_graph=True)
        b = torch.autograd.grad(revised, self.model.parameters())
        for x, y in zip(a, b):
            torch.testing.assert_close(x, y, atol=0, rtol=0)

    def test_residual_energy_weights_analytic_sign_prior_and_has_expected_gradient(self):
        delta = torch.tensor([[[1., 2., 3.], [4., 5., 6.]]], dtype=torch.float64, requires_grad=True)
        baseline = torch.full_like(delta, -100.)
        prior = torch.tensor([[.75, .25]], dtype=torch.float64)
        p = {'training_nll': delta.new_zeros((1, 4)), 'encoded': baseline+delta, 'log_prior': prior.log()}
        weights = objective_weights({'residual_l2_weight': 1.})
        loss = distance_objective(self.model, {'baseline': baseline}, p, weights)['residual_l2']
        self.assertAlmostEqual(float(loss.detach()), .75*14/3+.25*77/3)
        gradient = torch.autograd.grad(loss, delta)[0]
        torch.testing.assert_close(gradient, 2*delta*prior[..., None]/3)
        p['encoded'] = baseline+10000+delta
        shifted = distance_objective(self.model, {'baseline': baseline+10000}, p, weights)['residual_l2']
        torch.testing.assert_close(loss, shifted, atol=0, rtol=0)

    def test_curve_targets_include_escapes_and_ignore_likelihood_subsampling(self):
        expected = torch.tensor([[1., .75, .5, .5, .25]], dtype=torch.float64)
        torch.testing.assert_close(self.batch['reference_T'], expected)
        sub = make_batch(self.profiles, [torch.tensor([0])])
        self.assertEqual(sub['hit'].numel(), 1)
        torch.testing.assert_close(sub['reference_T'], expected)
        weights = objective_weights({'transmittance_weight': 1.})
        full_loss = distance_objective(self.model, self.batch, predictions(self.model, self.batch), weights)
        sub_loss = distance_objective(self.model, sub, predictions(self.model, sub), weights)
        torch.testing.assert_close(full_loss['transmittance_mse'], sub_loss['transmittance_mse'], atol=0, rtol=0)

    def test_curve_square_error_differs_from_empirical_brier_only_by_constant(self):
        p = predictions(self.model, self.batch)
        q = torch.linspace(0, 1, 5, dtype=torch.float64)[None]
        T = transmittance(self.model, p['encoded'], q, self.batch['horizon'], p['log_prior'])
        alive = (~self.batch['hit'][..., None]) | (self.batch['distance'][..., None] > q[:, None])
        brier = (T[:, None]-alive.double()).square().mean()
        mse = (T-self.batch['reference_T']).square().mean()
        variance = (self.batch['reference_T']*(1-self.batch['reference_T'])).mean()
        torch.testing.assert_close(brier-mse, variance, atol=1e-14, rtol=0)

    def test_auxiliary_losses_backpropagate_without_extra_network_forward(self):
        with torch.no_grad():
            self.model.hazard_head.bias.fill_(.3)
        target = self.batch['reference_T'].requires_grad_()
        with patch.object(self.model.encoder, 'forward', wraps=self.model.encoder.forward) as forward:
            p = predictions(self.model, self.batch)
            terms = distance_objective(self.model, self.batch, p,
                objective_weights({'residual_l2_weight': .01, 'transmittance_weight': 10.}))
            terms['loss'].backward()
            self.assertEqual(forward.call_count, 1)
        self.assertIsNone(target.grad)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in self.model.parameters()))
        self.assertGreater(float(terms['residual_l2'].detach()), 0.)
        self.assertGreater(float(terms['transmittance_mse'].detach()), 0.)

    def test_all_escape_curve_penalizes_positive_hazard_correction(self):
        raw = {**self.raw, 'hit': torch.zeros(4, dtype=torch.bool),
               'distance': torch.ones(4, dtype=torch.float64)}
        batch = make_batch(prepare({'profiles': [raw]}, self.config, 'cpu')['train'])
        p = predictions(self.model, batch)
        terms = distance_objective(self.model, batch, p, objective_weights({'transmittance_weight': 1.}))
        derivative = torch.autograd.grad(terms['transmittance_mse'], self.model.hazard_head.bias)[0]
        self.assertGreater(float(derivative.sum()), 0.)

    def test_objective_rejects_invalid_weights_and_direct_residual_penalty(self):
        for value in (-1., math.nan, math.inf):
            with self.assertRaises(ValueError):
                objective_weights({'residual_l2_weight': value})
        with self.assertRaises(ValueError):
            objective_weights({'unknown': 1.})
        with self.assertRaises(ValueError):
            objective_weights({'residual_l2_weight': 1.}, 'direct')

    def test_tail_and_validation_selection_obey_nll_guard(self):
        rows = [{'max_abs_T_error': float(i)/100, 'distance_nll': .5, 'mse_T': .001,
                 'escape_error': -.01} for i in range(20)]
        self.assertAlmostEqual(ray_metrics(rows)['worst_decile_mean'], .185)
        candidates = [{'name': name, 'eta': .01, 'beta': 10., 'validation': {
            'distance_nll': nll, 'worst_decile_mean': tail, 'mean_ray_max_abs_T_error': .02}}
            for name, nll, tail in (('too_costly', .51, .01), ('best', .503, .04), ('other', .502, .05))]
        result = choose_candidate(candidates, {'distance_nll': .5}, {'maximum_validation_nll_increase': .005})
        self.assertEqual(result['name'], 'best')
        self.assertNotIn('too_costly', result['eligible_names'])
        rejected = choose_candidate(candidates[:1], {'distance_nll': .5}, {'maximum_validation_nll_increase': .005})
        self.assertFalse(rejected['nll_guard_passed'])


if __name__ == '__main__':
    unittest.main()
