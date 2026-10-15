"""Reproducible parallel references and independent collision-window diagnostics."""
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from first_hit_se import EVENT, SE_KERNEL
from first_hit_se.camera_study import gradient_windows, sample_mixture
from first_hit_se.data import collect, load_data
from renewal.collect import write_json
from renewal.model import mixture_cdf


class CameraStudyTests(unittest.TestCase):
    def test_collision_windows_use_independent_pilot_realizations(self):
        q = torch.linspace(.1, 2., 4096, dtype=torch.float64)
        p = {'distance': torch.cat((q, q+.00001)), 'hit': torch.ones(8192, dtype=torch.bool)}
        windows = gradient_windows(p, [.2, .5, .8], .075)
        for lo, at, hi, indices in windows:
            self.assertLess(lo, at)
            self.assertLess(at, hi)
            self.assertTrue((q == at).any())
            self.assertTrue((indices >= 4096).all())
        # Changing evaluation labels may change membership, never window selection.
        changed = {**p, 'distance': torch.cat((q, q+.02))}
        other = gradient_windows(changed, [.2, .5, .8], .075)
        self.assertEqual([w[:3] for w in windows], [w[:3] for w in other])

    def test_truncated_mixture_sampling_handles_extreme_negative_means(self):
        u = np.array([1e-8, .01, .25, .5, .9, 1-1e-8])
        means = torch.tensor([[-30., 1.]], dtype=torch.float64).expand(len(u), -1)
        scales = torch.ones_like(means)
        weights = torch.tensor([[.8, .2]], dtype=torch.float64).log().expand(len(u), -1)
        sampled = sample_mixture((weights, means, scales), u)
        self.assertTrue(np.all(sampled > 0))
        np.testing.assert_allclose(mixture_cdf(weights, means, scales, torch.from_numpy(sampled)).numpy(), u, atol=1e-10)

    @unittest.skipUnless(os.environ.get('MACROFACET_RENEWAL_EXE'), 'C++ reference executable not provided')
    def test_parallel_collection_preserves_seeds_labels_and_profile_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sources = []
            for i, split in enumerate(('train', 'validation', 'test')):
                reference = root/f'{split}.json'
                write_json(reference, {'schema_version': 1, 'first_passage': {
                    'initial_condition': {'type': EVENT, 'profiles': [{'id': split, 'beta_0': .2*i, 'beta_a': -.5}]},
                    'grid': {'max_time': .413}, 'curve': {'bins': 8},
                    'kernels': [{'id': 'se', **SE_KERNEL, 'variance': 1., 'length_scale': 1.}]}})
                sources.append({'scene_id': split, 'split': split, 'reference_config': str(reference)})
            for workers in (1, 3):
                config = root/f'collect_{workers}.json'
                write_json(config, {'output_directory': str(root/str(workers)),
                    'executable': os.environ['MACROFACET_RENEWAL_EXE'], 'seed': 20261014,
                    'collection_workers': workers, 'threads': 2, 'realizations_per_profile': 64,
                    'reference_sampler': {'type': 'gpis_function_space'}, 'sources': sources})
                collect(config)
            a, b = (load_data(root/str(w)/'dataset.pt') for w in (1, 3))
            for first, second in zip(a['profiles'], b['profiles']):
                self.assertEqual(first['id'], second['id'])
                self.assertEqual(first['split'], second['split'])
                for key in ('features', 'distance', 'hit', 'initial_sign', 'speed'):
                    torch.testing.assert_close(first[key], second[key], atol=0, rtol=0, equal_nan=True)
                self.assertEqual((Path(first['source'])/'first_passage_samples.csv').read_bytes(),
                                 (Path(second['source'])/'first_passage_samples.csv').read_bytes())


if __name__ == '__main__':
    unittest.main()
