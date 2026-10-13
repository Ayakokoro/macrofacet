import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from plot_collision_gradients import histogram, weighted_summary, transverse_parameters
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
from compare_se_collision_gradients import conditional_target


class GradientStatisticsTests(unittest.TestCase):
    def test_weighted_statistics_are_scale_invariant(self):
        first=weighted_summary([1.,2.,8.],[2.,5.,1.])
        second=weighted_summary([1.,2.,8.],[20.,50.,10.])
        self.assertEqual(first['mean'],2.5)
        self.assertEqual(first['q50'],2.)
        for key in ('mean','stddev','q10','q50','q90','effective_samples'):
            self.assertAlmostEqual(first[key],second[key])
        self.assertAlmostEqual(first['effective_samples'],64/30)

    def test_histogram_keeps_right_boundary_and_tail_mass(self):
        h=histogram([0.,1.,2.],[1.,2.,3.],0.,2.,2)
        self.assertEqual(h['cdf'],[0.,1/6,1.])
        self.assertTrue(math.isclose(sum(h['pdf']),1.))
        with self.assertRaises(ValueError):
            histogram([0.,3.],[1.,1.],0.,2.,2)

    def test_empty_weights_do_not_fabricate_distribution(self):
        with self.assertRaises(ValueError):
            weighted_summary([],[])
        with self.assertRaises(ValueError):
            weighted_summary([1.],[0.])

    def test_conditional_distance_uses_hit_mass_not_unconditional_median(self):
        meta=dict(ell=2.,replayed_sample_hit=False)
        segments=[dict(begin_x=0.,end_x=1.,rates=[.1]*4)]
        result=conditional_target(meta,segments,.5)
        p=-math.expm1(-.1)
        expected=-math.log1p(-.5*p)/.1*2
        self.assertAlmostEqual(result['distance'],expected,places=13)
        self.assertAlmostEqual(result['hit_probability'],p)
        self.assertFalse(result['source_replayed_sample_hit'])
        self.assertIsNone(conditional_target(meta,[dict(begin_x=0.,end_x=1.,rates=[0.]*4)],.5))

    def test_surface_transverse_gradient_preserves_birth_then_decorrelates(self):
        for q in (0.,.1,1.,10.):
            rho=math.exp(-.5*q*q)
            meta=dict(start_condition='collision_state',target_q=q,
                mean_gradient=[1.,2.,3.],transverse_variance=.125,e1=[1.,0.,0.],e2=[0.,1.,0.],
                source_hit=dict(gradient=[4.,5.,6.]),birth_mean_gradient=[1.,1.,1.],
                gp_transverse_mean=[1.+3*rho,2.+4*rho,3.+5*rho],
                gp_transverse_variance=.125*(-math.expm1(-q*q)))
            means,variance=transverse_parameters(meta)
            self.assertEqual(means,[1.+3*rho,2.+4*rho])
            self.assertAlmostEqual(variance,.125*(1-rho*rho))
            if q==0.:self.assertEqual(variance,0.)


if __name__=='__main__':
    unittest.main()
