import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from first_hit_se.batch import query_many
from first_hit_se.model import FirstHitNetwork, ModelConfig
from first_hit_se.gradient_head import (initialize_head, parameters_at, cache_profiles,
    backbone_digest, train_one, verify_frozen_predictions)
from first_hit_se.train import load_model
from renewal.collect import write_json
from renewal.model import mixture_log_prob, mixture_cdf


class GradientHeadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setup_data(self):
        torch.manual_seed(42)
        model = FirstHitNetwork(ModelConfig(hidden=4, width=8, mixture_width=8,
                                           components=2, maximum_step=.125)).double()
        profile = {'id': 'toy', 'features': torch.tensor([[.4, -.4, -1.2, -.7, 0.]], dtype=torch.float64),
                   'horizon': 1., 'split': 'train',
                   'distance': torch.tensor([.1, .3, .6, 1.], dtype=torch.float64),
                   'hit': torch.tensor([True, True, True, False]),
                   'initial_sign': torch.tensor([1, -1, 1, -1]),
                   'speed': torch.tensor([.4, .8, 1.2, math.nan], dtype=torch.float64)}
        return model, profile

    def test_cache_matches_full_head_loss_and_gradients_for_both_signs(self):
        source, p = self.setup_data()
        cached = cache_profiles(source, [p, {**p, 'hit': torch.zeros(4, dtype=torch.bool)}])
        self.assertEqual(len(cached['speed']), 3)
        self.assertEqual(cached['profiles'][1]['hits'], 0)
        self.assertFalse(cached['inputs'].requires_grad)
        model = initialize_head(source, 2)
        direct = query_many(model, [p['features']], [p['distance']])[0]
        branch = (p['initial_sign'][p['hit']] < 0).long()
        rows = torch.arange(3)
        mix = tuple(direct[k][branch, rows] for k in ('log_weights', 'means', 'scales'))
        direct_loss = -mixture_log_prob(*mix, cached['speed']).mean()
        direct_loss.backward()
        expected = {n: v.grad.clone() for n, v in model.mixture_head.named_parameters()}
        model.zero_grad(set_to_none=True)
        cached_loss = -mixture_log_prob(*parameters_at(model, cached['inputs'], cached['derivative']),
                                       cached['speed']).mean()
        cached_loss.backward()
        torch.testing.assert_close(cached_loss, direct_loss, atol=1e-12, rtol=1e-12)
        for name, value in model.mixture_head.named_parameters():
            torch.testing.assert_close(value.grad, expected[name], atol=1e-12, rtol=1e-12)
        for name, value in model.named_parameters():
            if not name.startswith('mixture_head.'):
                self.assertIsNone(value.grad)

    def test_component_splitting_preserves_density_before_jitter(self):
        source, p = self.setup_data()
        cached = cache_profiles(source, [p])
        with torch.no_grad():
            expected = parameters_at(source, cached['inputs'], cached['derivative'])
            for count in (2, 4, 8):
                model = initialize_head(source, count, split_jitter=0)
                actual = parameters_at(model, cached['inputs'], cached['derivative'])
                for function in (mixture_log_prob, mixture_cdf):
                    torch.testing.assert_close(function(*actual, cached['speed']),
                                               function(*expected, cached['speed']), atol=1e-12, rtol=1e-12)
                self.assertEqual(backbone_digest(model), backbone_digest(source))
            separated = initialize_head(source, 8, seed=17, split_jitter=.002)
            means = parameters_at(separated, cached['inputs'], cached['derivative'])[1]
            self.assertTrue(torch.all(means[:, 0] != means[:, 1]))
        with self.assertRaises(ValueError):
            initialize_head(source, 3)

    def test_training_freeze_exportable_checkpoint_and_exact_resume(self):
        source, p = self.setup_data()
        source.requires_grad_(False)
        cached = cache_profiles(source, [p])
        cache = {'identity': {'test': 'fixed'}, 'splits': {'train': cached, 'validation': cached}}
        spec = {'initialize_from': 'toy.pt', 'split_jitter': .002,
                'training': {'epochs': 3, 'batch_size': 2, 'learning_rate': .001,
                             'minimum_learning_rate': .0001, 'weight_decay': 1e-5,
                             'gradient_clip': 1., 'validation_every': 1}}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            whole, result = train_one(source, cache, spec, 4, 137, root/'whole')
            self.assertTrue(result['backbone_unchanged'])
            self.assertEqual(result['optimizer_updates'], 6)
            frozen = verify_frozen_predictions(source, whole, [p])
            self.assertEqual(max(frozen.values()), 0.)
            initial = initialize_head(source, 4, 137, .002)
            self.assertTrue(any(not torch.equal(a, b) for a, b in
                                zip(whole.mixture_head.parameters(), initial.mixture_head.parameters())))
            exported, _ = load_model(root/'whole'/'last.pt')
            # load_model deliberately builds float32 inference parameters.
            self.assertEqual(exported.config.components, 4)
            def interrupt(path, value):
                if Path(path).name == 'history.json':
                    raise RuntimeError('test interruption after checkpoint')
                write_json(path, value)
            with mock.patch('first_hit_se.gradient_head.write_json', side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError, 'test interruption'):
                    train_one(source, cache, spec, 4, 137, root/'resumed')
            resumed, _ = train_one(source, cache, spec, 4, 137, root/'resumed')
            for key, value in whole.state_dict().items():
                torch.testing.assert_close(value, resumed.state_dict()[key], atol=0., rtol=0.)


if __name__ == '__main__':
    unittest.main()
