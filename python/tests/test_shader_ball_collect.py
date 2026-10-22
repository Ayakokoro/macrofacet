import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from renewal import SE_KERNEL
from renewal.collect import collision_seeds
from renewal.data import prepare, load_data
from renewal.shader_ball_collect import stratified_residuals, status
from test_renewal import fixture


class ShaderBallCollectionTests(unittest.TestCase):
    @unittest.skipUnless(os.name == 'nt', 'Windows file sharing semantics')
    def test_status_survives_a_reader_temporarily_blocking_rename(self):
        import ctypes
        from ctypes import wintypes
        import threading
        create = ctypes.windll.kernel32.CreateFileW
        create.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
                           wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
        create.restype = wintypes.HANDLE
        close = ctypes.windll.kernel32.CloseHandle
        close.argtypes = [wintypes.HANDLE]
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            status(directory, 'planned')
            path = directory/'generation_status.json'
            handle = create(str(path), 0x80000000, 3, None, 3, 0x80, None)
            self.assertNotEqual(handle, wintypes.HANDLE(-1).value)
            release = threading.Timer(.2, close, [handle])
            release.start()
            try:
                status(directory, 'sampling', completed_jobs=1)
            finally:
                release.join()
            self.assertEqual(json.loads(path.read_text())['completed_jobs'], 1)

    def test_residual_coverage_and_positive_total_derivative(self):
        slopes = np.tile([-2.9, -1.0, 0.0, 1.0, 5.0], 200)
        residual, quantile = stratified_residuals(slopes, 3.0, 42)
        self.assertTrue((np.abs(residual) <= 3).all())
        self.assertTrue((slopes+residual > 0).all())
        np.testing.assert_array_equal(np.sort((quantile*len(slopes)).astype(int)), np.arange(len(slopes)))
        self.assertLess(residual[slopes == 5].min(), -2.9)
        self.assertGreater(residual.max(), 2.9)
        with self.assertRaisesRegex(ValueError, "no outward derivative"):
            stratified_residuals([-3.0], 3.0, 42)

    def test_seed_algorithm_matches_independent_integer_implementation(self):
        mask = (1 << 64)-1
        def mix(x):
            x = (x+0x9e3779b97f4a7c15) & mask
            x = ((x ^ (x >> 30))*0xbf58476d1ce4e5b9) & mask
            x = ((x ^ (x >> 27))*0x94d049bb133111eb) & mask
            return x ^ (x >> 31)
        root = 0xfedcba9876543210
        for resolution in (0, 1):
            expected = [mix(mix(root ^ 0x636f6c6c6973696f) ^ mix(0x10001) ^
                            mix(resolution+0x20001) ^ mix(i+0x30001)) for i in range(32)]
            self.assertEqual(collision_seeds(root, 0, resolution, 32).tolist(), expected)

    def test_same_scene_ray_splits_and_leakage_rejection(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            manifest_path = fixture(directory, SE_KERNEL)
            manifest = json.loads(manifest_path.read_text())
            manifest['split_unit'] = 'ray'
            for i, source in enumerate(manifest['sources']):
                source.update(scene_id='same_scene', geometry={'model': 'shader_ball'})
                path = Path(source['directory'])/'resolved_first_passage_config.json'
                config = json.loads(path.read_text())
                config['first_passage']['initial_condition']['rays'] = [
                    {'id': 'ray', 'origin': [i, 0, 0], 'direction': [1, 0, 0]}]
                path.write_text(json.dumps(config))
            manifest_path.write_text(json.dumps(manifest))
            report = prepare(manifest_path, directory/'data.pt')
            self.assertEqual(report['split_unit'], 'ray')
            self.assertTrue(all(x['scenes'] == 1 and x['samples'] == 2 for x in report['counts'].values()))
            data = load_data(directory/'data.pt')
            self.assertEqual(data['hit'].tolist(), [True, False]*3)
            path = directory/'validation'/'resolved_first_passage_config.json'
            config = json.loads(path.read_text())
            config['first_passage']['initial_condition']['rays'][0]['origin'] = [0, 0, 0]
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, 'physical ray'):
                prepare(manifest_path, directory/'invalid.pt')

    def test_declared_residual_range_is_checked(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            path = fixture(directory, SE_KERNEL)
            manifest = json.loads(path.read_text())
            manifest['surface_derivative_sampling'] = {'residual_limit': 0.5}
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'residual exceeds'):
                prepare(path, directory/'invalid.pt')


if __name__ == '__main__':
    unittest.main()
