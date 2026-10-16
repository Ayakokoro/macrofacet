"""Geometry selection invariants protecting the expanded held-out dataset."""
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from first_hit_se.camera_holdout import (camera_position, candidates, describe,
                                         diverse_indices, nearest_distance, ray_key)


class CameraHoldoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_camera_orbit_preserves_radius_and_original_view_exactly(self):
        position, target = [2.27, -1.09, 1.42], [0., 0., .35]
        self.assertEqual(camera_position(position, target, 0), position)
        radius = np.linalg.norm(np.array(position)-target)
        for yaw, elevation in ((60, None), (180, None), (0, 5), (0, 60)):
            moved = np.array(camera_position(position, target, yaw, elevation))-target
            self.assertAlmostEqual(np.linalg.norm(moved), radius, places=12)
            if elevation is not None:
                self.assertAlmostEqual(np.degrees(np.arcsin(moved[2]/radius)), elevation, places=12)
        opposite = np.array(camera_position(position, target, 180))-target
        np.testing.assert_allclose(opposite[:2], -(np.array(position)-target)[:2], atol=1e-12)

    def test_stratified_candidates_exclude_prior_pixels_and_are_reproducible(self):
        original = candidates(32, 24, 4, 17)
        extra = candidates(32, 24, 4, 17, original)
        self.assertFalse(set(original) & set(extra))
        self.assertEqual(extra, candidates(32, 24, 4, 17, original))
        self.assertEqual(len({(x*4//32, y*4//24) for x, y in extra}), 16)

    def test_diverse_selection_covers_new_shapes_instead_of_duplicate_anchors(self):
        values = np.array([[0., 0.], [10., 0.], [10., 0.], [0., 8.], [.1, 0.]])
        anchors = np.array([[0., 0.]])
        selected = diverse_indices(values, anchors, 2)
        self.assertEqual(selected, [1, 3])
        np.testing.assert_allclose(nearest_distance(values, anchors), [0., 10., 10., 8., .1])
        self.assertEqual(len(set(diverse_indices(values, anchors, 5))), 5)
        with self.assertRaises(ValueError):
            diverse_indices(values, anchors, 6)

    def test_descriptor_uses_shape_and_length_and_ray_identity_uses_geometry(self):
        profile = torch.tensor([[2., -2., -4., -4., 0.]], dtype=torch.float64)
        descriptor, stats = describe(profile, 1.)
        self.assertEqual(stats['sampled_zero_crossings'], 1)
        self.assertAlmostEqual(stats['max_abs_slope'], 4.)
        changed = profile.clone()
        changed[:, 4] += np.log(2)
        other, _ = describe(changed, 2.)
        self.assertGreater(np.linalg.norm(descriptor-other), .1)
        ray = {'entry_origin': [1., 2., 3.], 'direction': [0., 0., -1.], 'pixel': [1, 2]}
        self.assertEqual(ray_key(ray), ray_key({**ray, 'pixel': [20, 30]}))
        self.assertNotEqual(ray_key(ray), ray_key({**ray, 'direction': [0., -1., 0.]}))


if __name__ == '__main__':
    unittest.main()
