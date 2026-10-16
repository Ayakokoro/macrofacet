import math
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from first_hit_se.causal_gradient import CausalGradient, GradientConfig, points
from first_hit_se.causal_gradient_step import (assert_same_observations, conditional_metrics,
    contexts, hybrid_contexts, maximum_numeric_difference, recurrent_step)
from first_hit_se.causal_gradient_study import evaluate
from first_hit_se.causal_residual import CausalConfig, CausalResidual
from first_hit_se.gradient_cdf import cdf
from first_hit_se.hazard_study import parameter_hash


class GradientStepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(19)
        self.backbone = CausalResidual(CausalConfig(hidden=4, embedding=4, width=8)).requires_grad_(False).eval()
        self.raw = [{'features': torch.tensor([[.4, -.3, -.2, .4, math.log(1.3)]], dtype=torch.float64),
            'distance': torch.tensor([.2, .8, 1.3, 1.3], dtype=torch.float64),
            'initial_sign': torch.tensor([1., -1., 1., 1.]),
            'speed': torch.tensor([.4, 1.7, .8, math.nan], dtype=torch.float64),
            'hit': torch.tensor([True, True, True, False])}]
        self.base = contexts(self.backbone, self.raw, .125, 'cpu')
        self.fine = contexts(self.backbone, self.raw, .0625, 'cpu')

    def test_step_restoration_and_weights_including_failed_operation(self):
        original, before = self.backbone.config, parameter_hash(self.backbone)
        with self.assertRaisesRegex(RuntimeError, 'interrupted'):
            with recurrent_step(self.backbone, .25):
                self.assertEqual(self.backbone.config.step, .25)
                self.assertEqual(self.backbone.config.integration_step, original.integration_step)
                raise RuntimeError('interrupted')
        self.assertIs(self.backbone.config, original)
        with self.assertRaises(ValueError):
            with recurrent_step(self.backbone, .001):
                pass
        self.assertIs(self.backbone.config, original)
        self.assertEqual(parameter_hash(self.backbone), before)

    def test_changes_context_without_changing_observations(self):
        assert_same_observations(self.base, self.fine)
        base, fine = points(self.base), points(self.fine)
        self.assertGreater(float((base['phase']-fine['phase']).abs().max()), .1)
        self.assertGreater(float((base['z']-fine['z']).abs().max()), 1e-4)
        broken = [{**self.fine[0], 'q': self.fine[0]['q']+.001}]
        with self.assertRaises(AssertionError):
            assert_same_observations(self.base, broken)

    def test_hybrids_use_matching_state_and_index(self):
        base, fine = points(self.base), points(self.fine)
        state = points(hybrid_contexts(self.base, self.fine, 'state_only'))
        phase = points(hybrid_contexts(self.base, self.fine, 'phase_only'))
        for key in base:
            if not isinstance(base[key], torch.Tensor):
                self.assertEqual(state[key], base[key])
                continue
            torch.testing.assert_close(state[key], fine[key] if key == 'z' else base[key], atol=0, rtol=0)
            torch.testing.assert_close(phase[key], fine[key] if key == 'phase' else base[key], atol=0, rtol=0)
        local = CausalGradient(GradientConfig(hidden=4, width=8, size=4, history=False), 33)
        with torch.no_grad():
            torch.testing.assert_close(cdf('mixture', local(base), base['w']),
                                       cdf('mixture', local(state), state['w']), atol=0, rtol=0)
            torch.testing.assert_close(cdf('mixture', local(fine), fine['w']),
                                       cdf('mixture', local(phase), phase['w']), atol=0, rtol=0)

    def test_identical_step_reproduces_conditional_metrics(self):
        head = CausalGradient(GradientConfig(hidden=4, width=8, size=4), 33)
        a = conditional_metrics(evaluate(head, self.base))
        b = conditional_metrics(evaluate(head, contexts(self.backbone, self.raw, .125, 'cpu')))
        self.assertEqual(maximum_numeric_difference(a, b), 0.)
        self.assertNotIn('joint_nll', a)
        self.assertNotIn('joint_nll', a['per_ray'][0])
        self.assertEqual(a['hits'], 3)
        self.assertEqual(a['samples'], 4)


if __name__ == '__main__':
    unittest.main()
