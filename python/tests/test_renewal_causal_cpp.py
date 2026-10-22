import json
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from first_hit_se.causal_gradient import GradientConfig
from first_hit_se.causal_residual import CausalConfig
from first_hit_se.distance_study import restrict_profile
from renewal.causal_export import export_joint
from renewal.causal_gradient import FORMAT, VERSION, TARGET, HitGradient, load_joint
from renewal.causal_residual import CONDITIONING, RenewalResidual
from renewal.causal_verify import compare_profile

ROOT = Path(__file__).resolve().parents[2]
EXE = Path(os.environ.get('MACROFACET_CAUSAL_RENEWAL_EXE',
                          ROOT/'build'/'Release'/'macrofacet_query_causal_renewal.exe'))


class JointDeploymentTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(132)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        backbone = RenewalResidual(CausalConfig(hidden=4, embedding=4, width=8))
        with torch.no_grad():
            backbone.head[-1].weight.normal_(std=.7)
            backbone.head[-1].bias.fill_(.1)
        head = HitGradient(GradientConfig(hidden=4, width=8, size=3), 42)
        self.saved = {'format': FORMAT, 'version': VERSION, 'conditioning': CONDITIONING, 'target': TARGET,
                      'backbone_config': backbone.configuration(), 'backbone_state': backbone.state_dict(),
                      'head_config': head.configuration(), 'head_state': head.state_dict()}
        self.checkpoint = self.root/'joint.pt'
        torch.save(self.saved, self.checkpoint)
        self.bundle = self.root/'model.json'

    def test_export_is_self_contained_and_rejects_invalid_weights(self):
        report = export_joint(self.checkpoint, self.bundle)
        bundle = json.loads(self.bundle.read_text())
        self.assertEqual(bundle['format'], 'macrofacet.causal_renewal')
        self.assertEqual(bundle['backbone']['initial.0.weight']['shape'], [4, 4])
        self.assertEqual(bundle['model_config']['step'], .125)
        self.assertGreater(report['parameters'], 0)
        self.saved['head_state']['net.0.weight'][0, 0] = math.nan
        torch.save(self.saved, self.checkpoint)
        with self.assertRaisesRegex(ValueError, 'invalid joint tensor'):
            export_joint(self.checkpoint, self.bundle)

    def test_export_rejects_unrelated_checkpoints(self):
        torch.save({'format': 'renewal_causal_residual'}, self.checkpoint)
        with self.assertRaises(ValueError):
            export_joint(self.checkpoint, self.bundle)

    @unittest.skipUnless(EXE.exists(), 'build macrofacet_query_causal_renewal for deployment parity')
    def test_cpp_matches_nonzero_network_for_A_B_and_prefixes(self):
        export_joint(self.checkpoint, self.bundle)
        backbone, head, _ = load_joint(self.checkpoint)
        features = torch.tensor([[.6, -.4, -1., -.5, math.log(.7)],
                                 [-.4, .8, .3, 1., math.log(.6)]], dtype=torch.float64)
        initial_conditions = [[0, .6, 0, 0], [1, .6, -.6, 2.], [1, .6, -.6, 4.]]
        for i, initial in enumerate(initial_conditions):
            initial = torch.tensor(initial, dtype=torch.float64)
            for cutoff in (1.3, .5, .65):
                f = features if cutoff == 1.3 else restrict_profile(features, cutoff/1.3)
                report = compare_profile(backbone, head, f, initial, EXE, self.bundle, self.root,
                                         f'mode_{i}_cutoff_{cutoff}', trials=12)
                self.assertLess(report['max_T_error'], 2e-5)


if __name__ == '__main__':
    unittest.main()
