import math
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from first_hit_se.causal_data import geometry
from first_hit_se.causal_gradient import (CausalGradient, GradientConfig, cache_profile,
                                        entering_states, points, query_points, state_indices)
from first_hit_se.causal_gradient_study import evaluate
from first_hit_se.causal_residual import CausalConfig, CausalResidual
from first_hit_se.distance_study import alter_suffix, restrict_profile
from first_hit_se.gradient_cdf import cdf, log_prob
from first_hit_se.hazard_study import parameter_hash


class CausalGradientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(17)
        self.backbone = CausalResidual(CausalConfig(hidden=4, embedding=4, width=8)).requires_grad_(False).eval()
        with torch.no_grad():
            self.backbone.head[-1].weight.normal_(0, .3)
        self.features = torch.tensor([[.6, -.4, -1., -.5, math.log(.7)],
                                      [-.4, .8, .3, 1., math.log(.6)]], dtype=torch.float64)
        self.raw = {'features': self.features, 'distance': torch.tensor([.2, .8, 1.3, 1.3], dtype=torch.float64),
            'hit': torch.tensor([True, True, True, False]), 'initial_sign': torch.tensor([1., -1., 1., 1.]),
            'speed': torch.tensor([.3, 1.2, 2.1, math.nan], dtype=torch.float64)}
        self.row = cache_profile(self.backbone, self.raw)

    def head(self, kind='mixture', history=True):
        h = CausalGradient(GradientConfig(hidden=4, width=8, size=4, kind=kind, history=history), 71)
        if kind == 'cdf':
            with torch.no_grad():
                h.net[-1].weight.normal_(0, .2)
        return h

    def test_cached_context_matches_distance_entering_states(self):
        geo = geometry(self.features, self.backbone.config)
        states, _ = entering_states(self.backbone, geo)
        delta = self.backbone.correction(states, geo['left'])
        batch = {k: geo[k][None] for k in ('tokens', 'left', 'entry')}
        torch.testing.assert_close(delta, self.backbone(batch)[0], atol=0, rtol=0)
        selected = self.raw['hit']
        direct = query_points(self.backbone, self.features, self.raw['distance'][selected], self.raw['initial_sign'][selected])
        cached = points([self.row])
        for k in direct:
            torch.testing.assert_close(cached[k], direct[k], atol=0, rtol=0)

    def test_cdf_preserves_prefix_including_exact_recurrent_boundaries(self):
        for kind in ('mixture', 'cdf'):
            head = self.head(kind)
            for cutoff in (.5, .65, 1.):
                q = torch.tensor([0., .125, .213, cutoff], dtype=torch.float64)
                sign = torch.tensor([1., -1., 1., -1.], dtype=torch.float64)
                full = query_points(self.backbone, self.features, q, sign)
                for f in (restrict_profile(self.features, cutoff/1.3), alter_suffix(self.features, cutoff/1.3)):
                    changed = query_points(self.backbone, f, q, sign)
                    torch.testing.assert_close(cdf(kind, head(full), q+1.),
                                               cdf(kind, head(changed), q+1.), atol=2e-7, rtol=0)

    def test_left_limit_index_does_not_read_next_interval(self):
        q = torch.tensor([0., .124, .125, .126, .5], dtype=torch.float64)
        cell, phase = state_indices(q, .125, 4)
        self.assertEqual(cell.tolist(), [0, 0, 0, 1, 3])
        self.assertEqual(phase[[0, 2, 4]].tolist(), [0., 1., 1.])

    def test_only_hits_train_and_backbone_is_unchanged(self):
        before = parameter_hash(self.backbone)
        head = self.head()
        p = points([self.row], [torch.tensor([0, 3])])
        self.assertEqual(p['samples'], 2)
        self.assertEqual(len(p['w']), 1)
        loss = -log_prob('mixture', head(p), p['w']).sum()/p['samples']
        loss.backward()
        self.assertTrue(all(p.grad is None for p in self.backbone.parameters()))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters()))
        torch.optim.SGD(head.parameters(), lr=.01).step()
        self.assertEqual(before, parameter_hash(self.backbone))
        p = points([self.row], [torch.tensor([3])])
        self.assertEqual(len(p['w']), 0)

    def test_same_capacity_ablation_and_sign_symmetry(self):
        history, local = self.head(), self.head(history=False)
        self.assertEqual(parameter_hash(history), parameter_hash(local))
        p = points([self.row])
        changed = {**p, 'z': p['z']+5.}
        w = p['w']
        self.assertGreater(float((cdf('mixture', history(p), w)-cdf('mixture', history(changed), w)).abs().max().detach()), 1e-6)
        torch.testing.assert_close(cdf('mixture', local(p), w), cdf('mixture', local(changed), w), atol=0, rtol=0)
        f = self.features.clone()
        f[:, :4] *= -1
        flipped = query_points(self.backbone, f, p['q'], -p['sign'])
        torch.testing.assert_close(cdf('mixture', history(p), w), cdf('mixture', history(flipped), w), atol=0, rtol=0)

    def test_evaluation_handles_zero_hit_ray_and_joint_accounting(self):
        head = self.head()
        empty = {**self.row, 'hit': torch.zeros(4, dtype=torch.bool), 'joint_distance_sum': torch.tensor(2.)}
        score = evaluate(head, [self.row, empty])
        self.assertEqual(score['hits'], 3)
        self.assertEqual(score['samples'], 8)
        self.assertIsNone(score['per_ray'][1]['pit_ks'])
        p = points([self.row])
        loss = -log_prob('mixture', head(p), p['w']).sum()
        expected = (self.row['joint_distance_sum']+2.+loss)/8
        self.assertAlmostEqual(score['joint_nll'], float(expected.detach()), places=12)


if __name__ == '__main__':
    unittest.main()
