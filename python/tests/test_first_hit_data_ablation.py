from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from first_hit_se.data_ablation import merge_profiles, profile_key
from first_hit_se.data_ablation_train import family_indices, sample_indices, train_fixed, prepare_contexts
from first_hit_se.gradient_head import cache_profiles, verify_frozen_predictions
from first_hit_se.model import FirstHitNetwork, ModelConfig
from renewal.collect import write_json, digest


class DataAblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def profile(self, source='unused/reference', split='train'):
        return {'id': source, 'split': split, 'source': source, 'horizon': 1.,
                'features': torch.tensor([[.4, -.4, -1.2, -.7, 0.]], dtype=torch.float64),
                'geometry': {'direction': [0, 0, 1], 'origin': [0,0,0], 'sigma': 1., 'ell': 1.},
                'distance': torch.tensor([.1,.3,.6,1.], dtype=torch.float64),
                'hit': torch.tensor([True, True, True, False]),
                'initial_sign': torch.tensor([1,-1,1,1]),
                'speed': torch.tensor([.4,.8,1.2,float('nan')], dtype=torch.float64),
                'samples_sha256': 'samples', 'resolved_sha256': 'geometry'}

    def test_nested_data_and_heldout_preservation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for folder in ('old', 'new'):
                (root/folder).mkdir()
                (root/folder/'ray.json').write_text(json.dumps({'pixel': [12,34]}))
            p = self.profile(str(root/'old'/'reference'))
            held = self.profile('held/reference', 'test')
            held['geometry']['origin'] = [1,0,0]
            base = {'profiles': [p, held]}
            added = self.profile(str(root/'new'/'reference'))
            added['speed'][:3] += .2
            merged = merge_profiles(base, {'profiles':[added]}, 'samples')
            self.assertEqual(len(base['profiles'][0]['hit']), 4)
            self.assertEqual(len(merged['profiles'][0]['hit']), 8)
            torch.testing.assert_close(merged['profiles'][0]['speed'][:4], p['speed'], equal_nan=True)
            torch.testing.assert_close(merged['profiles'][1]['speed'], held['speed'], equal_nan=True)
            changed = deepcopy(added)
            changed['geometry']['origin'] = [0,0,1]
            with self.assertRaisesRegex(ValueError, 'physical ray'):
                merge_profiles(base, {'profiles':[changed]}, 'samples')
            with self.assertRaisesRegex(ValueError, 'overlap'):
                merge_profiles(base, {'profiles':[added]}, 'coverage')
            extended = merge_profiles(base, {'profiles':[changed]}, 'coverage')
            self.assertEqual(len(extended['profiles']), 3)
            self.assertEqual(profile_key(extended['profiles'][1]), profile_key(held))

    def test_stratification_keeps_affine_draws_paired(self):
        a = {'camera': torch.arange(10), 'affine': torch.arange(10,30)}
        b = {'camera': torch.arange(100), 'affine': torch.arange(10,30)}
        for step in range(1,6):
            x = sample_indices(a, 3, 7, 123, step)
            y = sample_indices(b, 3, 7, 123, step)
            self.assertEqual(len(x), 7)
            self.assertTrue((x[:3]<10).all())
            torch.testing.assert_close(x[3:], y[3:], atol=0, rtol=0)
            torch.testing.assert_close(x, sample_indices(a,3,7,123,step), atol=0, rtol=0)
        self.assertFalse(torch.equal(sample_indices(a,3,7,123,2),sample_indices(a,3,7,124,1)),
                         'Adjacent seeds must not generate one-step-shifted identical batch streams')

    def test_expanded_context_cache_uses_serialized_paths_and_rejects_changed_data(self):
        source = FirstHitNetwork(ModelConfig(hidden=4,width=8,mixture_width=8,components=8,
                                            maximum_step=.125)).double()
        source.requires_grad_(False)
        profile = self.profile()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root/'source.pt'
            torch.save(source.state_dict(), checkpoint)
            dataset = root/'expanded.pt'
            torch.save({'profiles':[profile]}, dataset)
            record = {'dataset':str(dataset), 'sha256':digest(dataset)}
            write_json(root/'datasets.json', {'coverage':record,'samples':record})
            spec = {'initialize_from':str(checkpoint)}
            base = {'identity':{'dataset_sha256':'original'}, 'splits':{'train':{'speed':torch.ones(1)}}}
            with mock.patch('first_hit_se.data_ablation_train.load_data',return_value={'profiles':[profile]}):
                cached, identities = prepare_contexts(spec, root, source, base)
            with mock.patch('first_hit_se.data_ablation_train.cache_profiles',side_effect=AssertionError('cache missed')):
                restored, again = prepare_contexts(spec, root, source, base)
            self.assertEqual(identities, again)
            torch.testing.assert_close(cached['coverage']['inputs'],restored['coverage']['inputs'],atol=0,rtol=0)
            dataset.write_bytes(b'changed dataset')
            with self.assertRaisesRegex(ValueError,'expanded dataset changed'):
                prepare_contexts(spec, root, source, base)

    def test_fixed_update_freeze_and_resume(self):
        torch.manual_seed(18)
        source = FirstHitNetwork(ModelConfig(hidden=4,width=8,mixture_width=8,components=8,
                                            maximum_step=.125)).double()
        source.requires_grad_(False)
        p = self.profile()
        cached = cache_profiles(source, [p, {**p,'id':'affine'}])
        cached['profiles'][0]['pixel'] = [12,34]
        pools = family_indices(cached)
        self.assertEqual(len(pools['camera']), 3)
        settings = {'updates':5,'batch_size':4,'learning_rate':.001,'minimum_learning_rate':.0001,
                    'weight_decay':1e-5,'gradient_clip':1.,'validation_every':2}
        identity = {'test':'fixed'}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            whole, report = train_fixed(source,cached,cached,settings,identity,root/'whole',137,2)
            self.assertEqual(report['updates'],5)
            self.assertEqual(report['examples_seen'],20)
            self.assertTrue(report['backbone_unchanged'])
            self.assertEqual(max(verify_frozen_predictions(source,whole,[p]).values()),0)
            def interrupt(path,value):
                if Path(path).name=='history.json':
                    raise RuntimeError('interrupt checkpoint')
                write_json(path,value)
            with mock.patch('first_hit_se.data_ablation_train.write_json',side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError,'interrupt checkpoint'):
                    train_fixed(source,cached,cached,settings,identity,root/'resumed',137,2)
            resumed,_=train_fixed(source,cached,cached,settings,identity,root/'resumed',137,2)
            for name,value in whole.state_dict().items():
                torch.testing.assert_close(value,resumed.state_dict()[name],atol=0,rtol=0)


if __name__=='__main__':
    unittest.main()
