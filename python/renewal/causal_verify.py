"""Compare exported A/B inference and inverse samples against PyTorch."""
import argparse
import json
import math
from pathlib import Path
import subprocess

import torch

from first_hit_se.causal_data import queries
from first_hit_se.causal_residual import branch_values
from first_hit_se.gradient_cdf import sample
from .causal_data import geometry
from .causal_export import export_joint
from .causal_gradient import load_joint, query_points
from .collect import digest, write_json


@torch.no_grad()
def compare_profile(backbone, head, features, initial, executable, model, output, name, trials=64):
    features, initial = features.double(), initial.double()
    geo = geometry(features, initial, backbone.config)
    batch = {k: geo[k][None] for k in ('tokens', 'left', 'entry', 'mass')}
    delta = backbone(batch)

    def prediction(q):
        return tuple(v[0, 0] for v in branch_values(delta, batch, {k: v[None] for k, v in queries(geo, q).items()}))

    horizon = float(geo['horizon'])
    # Include original cubic boundaries and both sides of recurrent boundaries.
    q = torch.cat((torch.linspace(0, horizon, 33, dtype=torch.float64),
                   features[:, 4].exp().cumsum(0),
                   torch.tensor([0., .125-1e-8, .125, .125+1e-8, .25, .5, .75], dtype=torch.float64)))
    q = q[(q >= 0) & (q <= horizon)].unique(sorted=True)
    H, rate = prediction(q)
    expected_params = head(query_points(backbone, features, initial, q))
    generator = torch.Generator().manual_seed(3221)
    uniforms = torch.rand((trials, 3), dtype=torch.float64, generator=generator).clamp(1e-10, 1-1e-10)
    # Force hits across the profile and an escape when representable.
    depth = H[-1]*torch.tensor([.03, .2, .5, .9, 1.2], dtype=torch.float64)
    forced = torch.full((len(depth), 3), .37, dtype=torch.float64)
    forced[:, 0] = (-depth).exp().clamp(1e-12, 1-1e-12)
    uniforms = torch.cat((uniforms, forced))
    start = {'mode': 'positive_exterior'}
    if initial[0] == 1:
        start = {'mode': 'collision_state', 'outward_derivative': float(initial[3]+features[0, 2]/features[0, 4].exp())}
    request = {'features': features.tolist(), 'start': start, 'queries': q.tolist(),
               'uniforms': uniforms.tolist(), 'tolerance': 1e-10}
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    source, destination = output/f'{name}_request.json', output/f'{name}_cpp.json'
    write_json(source, request)
    subprocess.run([str(Path(executable).resolve()), str(Path(model).resolve()), str(source.resolve()),
                    str(destination.resolve())], check=True, capture_output=True, text=True)
    result = json.loads(destination.read_text(encoding='utf-8'))
    actual_H = torch.tensor([p['H'] for p in result['predictions']], dtype=torch.float64)
    actual_rate = torch.tensor([p['log_hazard'] for p in result['predictions']], dtype=torch.float64)
    torch.testing.assert_close(actual_H, H, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(actual_rate, rate, atol=2e-5, rtol=2e-5)
    mixture_error = 0.
    for key, expected in zip(('weights', 'means', 'scales'), (expected_params[0].exp(), *expected_params[1:])):
        actual = torch.tensor([p['mixture'][key] for p in result['predictions']], dtype=torch.float64)
        expected = expected.double()
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
        mixture_error = max(mixture_error, float((actual-expected).abs().max()))

    target = -uniforms[:, 0].log()
    hit = target <= H[-1]
    lo, hi = torch.zeros_like(target), torch.full_like(target, horizon)
    for _ in range(40):
        mid = (lo+hi)/2
        below = prediction(mid)[0] < target
        lo, hi = torch.where(below, mid, lo), torch.where(below, hi, mid)
    distance = torch.where(hit, (lo+hi)/2, horizon)
    actual_q = torch.tensor([s['q'] for s in result['samples']], dtype=torch.float64)
    actual_hit = torch.tensor([s['hit'] for s in result['samples']])
    torch.testing.assert_close(actual_hit, hit)
    torch.testing.assert_close(actual_q, distance, atol=3e-5, rtol=2e-5)
    speed_error = 0.
    if hit.any():
        expected_speed = sample('mixture', head(query_points(backbone, features, initial, actual_q[hit])), uniforms[hit, 1:])
        actual_speed = torch.tensor([s['speed'] for s in result['samples']], dtype=torch.float64)[hit]
        torch.testing.assert_close(actual_speed, expected_speed, atol=3e-5, rtol=2e-5)
        speed_error = float((actual_speed-expected_speed).abs().max())
    return {'name': name, 'mode': start['mode'], 'queries': len(q), 'samples': len(uniforms), 'hits': int(hit.sum()),
            'max_T_error': float(((-actual_H).exp()-(-H).exp()).abs().max()),
            'max_H_error': float((actual_H-H).abs().max()), 'max_log_hazard_error': float((actual_rate-rate).abs().max()),
            'max_mixture_parameter_error': mixture_error,
            'max_q_error': float((actual_q-distance).abs().max()), 'max_speed_error': speed_error}


def verify(checkpoint, executable, output, dataset=None):
    torch.set_num_threads(1)
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    bundle = output/'renewal_model.json'
    export_joint(checkpoint, bundle)
    backbone, head, _ = load_joint(checkpoint)
    profiles = [
        torch.tensor([[.6, -.4, -1., -.5, math.log(.7)], [-.4, .8, .3, 1., math.log(.6)]], dtype=torch.float64),
        torch.tensor([[0., 0., 0., 0., math.log(.03)]], dtype=torch.float64),
        torch.tensor([[4., 4., 0., 0., math.log(.61)]], dtype=torch.float64),
    ]
    rows = []
    for i, features in enumerate(profiles):
        b0, db0 = float(features[0, 0]), float(features[0, 2]/features[0, 4].exp())
        for mode in (0, 1):
            initial = torch.tensor([0, b0, 0, 0] if mode == 0 else [1, b0, -b0, .8-db0], dtype=torch.float64)
            rows.append(compare_profile(backbone, head, features, initial, executable, bundle, output, f'synthetic_{i}_{mode}'))
    if dataset:
        from .causal_data import ProfileDataset
        from .causal_gradient import raw_profiles
        from .data import load_data
        data = ProfileDataset(load_data(Path(dataset)))
        # Include one observed hit and one escape profile per mode when available.
        for raw in raw_profiles(data, 'test', limit_per_mode=2):
            rows.append(compare_profile(backbone, head, raw['features'], raw['initial'], executable, bundle, output,
                                        f"test_profile_{int(raw['profile'])}"))
    report = {'checkpoint_sha256': digest(Path(checkpoint)), 'passed': True, 'profiles': rows,
              'scope': 'Python/C++ deployment parity, not an estimate of model error against GP reference'}
    write_json(output/'verification.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--executable', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--dataset', type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.checkpoint, args.executable, args.output, args.dataset), indent=2))


if __name__ == '__main__':
    main()
