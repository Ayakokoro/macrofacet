"""Checks for the independent-realization diagnostic and conditional averaging."""
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from first_hit_se.single_ray import (averaged_cdf, bin_edges, bin_indices,
                                    ks_uniform, validate_realization_sets)
from renewal.model import mixture_cdf


class SingleRayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_bins_use_only_hits_and_cover_boundaries_once(self):
        profile = {'distance': torch.tensor([1., 2., 3., 4., 10.], dtype=torch.float64),
                   'hit': torch.tensor([True, True, True, True, False]), 'horizon': 10.}
        edges = bin_edges(profile, 2)
        torch.testing.assert_close(edges, torch.tensor([0., 2.5, 10.], dtype=torch.float64))
        labels = bin_indices(torch.tensor([0., 2.49, 2.5, 10.], dtype=torch.float64), edges)
        self.assertEqual(labels.tolist(), [0, 0, 1, 1])
        with self.assertRaises(ValueError):
            bin_indices(torch.tensor([10.1]), edges)

    def test_independent_realization_split_rejects_shared_gp_seed(self):
        validate_realization_sets({'train': {'1', '2'}, 'validation': {'3'}, 'test': {'4', '5'}})
        with self.assertRaisesRegex(ValueError, 'leakage'):
            validate_realization_sets({'train': {'1', '2'}, 'validation': {'3'}, 'test': {'2', '4'}})

    def test_conditional_cdfs_are_averaged_per_reference_event(self):
        # Widely separated conditional laws must be mixed at their real queries,
        # not replaced by a distribution at the mean query/mean parameters.
        mixture = [torch.zeros((3, 1), dtype=torch.float64),
                   torch.tensor([[.3], [1.5], [3.]], dtype=torch.float64),
                   torch.tensor([[.2], [.5], [.3]], dtype=torch.float64)]
        grid = torch.tensor([0., .1, .5, 1., 2., 4.], dtype=torch.float64)
        expected = torch.stack([mixture_cdf(*(v[i] for v in mixture), grid) for i in range(3)]).mean(0).numpy()
        for chunk_size in (1, 2, 10):
            np.testing.assert_allclose(averaged_cdf(mixture, grid, chunk_size), expected, rtol=1e-13, atol=1e-13)
        wrong = mixture_cdf(*(v.mean(0) for v in mixture), grid).numpy()
        self.assertGreater(np.max(np.abs(expected-wrong)), .1)

    def test_pit_ks_uses_both_empirical_sides(self):
        self.assertAlmostEqual(ks_uniform([.1, .3, .5, .7, .9]), .1)
        self.assertAlmostEqual(ks_uniform([.9, .95]), .9)


if __name__ == '__main__':
    unittest.main()
