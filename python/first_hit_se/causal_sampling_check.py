"""Compare Python, Eigen scalar, LibTorch CPU/CUDA joint first-hit sampling."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import time

import torch

from renewal.collect import digest, write_json
from .causal_data import query_model
from .causal_gradient import query_points
from .causal_gradient_study import load_head
from .causal_study import load_run
from .causal_sampling import sample_profile
from .gradient_cdf import normalized_mixture


ROOT = Path(__file__).resolve().parents[2]


@torch.no_grad()
def run(args):
    torch.set_num_threads(1)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    bundle = json.loads(Path(args.model).read_text(encoding='utf-8'))
    provenance = bundle['provenance']
    for name in ('backbone', 'gradient'):
        if digest(Path(provenance[name])) != provenance[name+'_sha256']:
            raise ValueError('deployment checkpoint changed')
    backbone, _ = load_run(provenance['backbone'])
    head, _ = load_head(provenance['gradient'])
    data = torch.load(args.dataset, map_location='cpu', weights_only=True)
    test = [p for p in data['profiles'] if p['split'] == 'test']
    cases = [('flat', torch.tensor([[0., 0., 0., 0., math.log(2.03)]], dtype=torch.float64)),
             ('short', torch.tensor([[.3, .2, -.1, -.1, math.log(.03)]], dtype=torch.float64))]
    cases += [(f'test_{i}', test[i]['features']) for i in (0, 15, 84, 175, 266)]
    generator = torch.Generator().manual_seed(20261016)
    rows = []
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    start = time.perf_counter()
    for name, features in cases:
        Q = float(features[:, 4].exp().sum())
        q = torch.unique(torch.cat((torch.linspace(0, Q, 49, dtype=torch.float64),
            torch.tensor([0., .125-1e-10, .125, .125+1e-10, .25, .5], dtype=torch.float64))))
        q = q[(q >= 0) & (q <= Q)]
        uniforms = torch.rand((args.samples, 4), generator=generator, dtype=torch.float64)
        python_samples = sample_profile(backbone, head, features, uniforms)
        py = query_model(backbone, features, q)
        mixtures = [normalized_mixture(head(query_points(backbone, features, q,
            torch.full_like(q, sign)))) for sign in (1., -1.)]
        results = {}
        for backend in ('scalar', 'torch_cpu', 'torch_cuda'):
            request = {'features': features.tolist(), 'queries': q.tolist(), 'uniforms': uniforms.tolist(),
                       'backend': backend, 'batch_size': 256, 'tolerance': 1e-9}
            request_path = output/f'{name}_{backend}_request.json'
            result_path = output/f'{name}_{backend}.json'
            write_json(request_path, request)
            process = subprocess.run([str(Path(args.executable).resolve()), str(Path(args.model).resolve()),
                str(request_path.resolve()), str(result_path.resolve())], cwd=ROOT, env=env,
                capture_output=True, text=True)
            if process.returncode:
                raise RuntimeError(process.stdout+process.stderr)
            cpp = json.loads(result_path.read_text(encoding='utf-8'))
            predicted = cpp['predictions']; samples = cpp['samples']
            metrics = {'ray': name, 'backend': backend, 'query_backend': 'scalar', 'samples': len(samples), 'horizon': Q}
            for key in ('hit', 'initial_sign'):
                if [s[key] for s in samples] != python_samples[key].tolist():
                    raise AssertionError(f'{name}/{backend}: {key} mismatch')
            for key, tolerance in (('q', 3e-5), ('speed', 5e-4)):
                value = torch.tensor([s[key] for s in samples], dtype=torch.float64)
                metrics[key+'_max_abs_error'] = float((value-python_samples[key]).abs().max())
                if metrics[key+'_max_abs_error'] > tolerance:
                    raise AssertionError(metrics)
            T = torch.tensor([p['log_T'] for p in predicted], dtype=torch.float64).exp()
            metrics['T_max_abs_error'] = float((T-py['T']).abs().max())
            if metrics['T_max_abs_error'] > 2e-6:
                raise AssertionError(metrics)
            metrics['mixture_max_abs_error'] = 0.
            for branch in range(2):
                for index, key in enumerate(('weights', 'means', 'scales')):
                    expected = mixtures[branch][index]
                    if key == 'weights':
                        expected = expected.exp()
                    actual = torch.tensor([p['mixtures'][branch][key] for p in predicted], dtype=torch.float64)
                    error = float((actual-expected).abs().max())
                    metrics['mixture_max_abs_error'] = max(metrics['mixture_max_abs_error'], error)
            if metrics['mixture_max_abs_error'] > 3e-5:
                raise AssertionError(metrics)
            sample_q = torch.tensor([s['q'] for s in samples], dtype=torch.float64)
            sample_hit = torch.tensor([s['hit'] for s in samples])
            empirical = 1-((sample_q[:, None] <= q) & sample_hit[:, None]).double().mean(0)
            metrics['empirical_T_max_abs_error'] = float((empirical-py['T']).abs().max())
            rows.append(metrics); results[backend] = samples
            print(json.dumps(metrics), flush=True)
        if name == 'flat':
            # Different batch boundaries and changing active-slot membership.
            request['batch_size'] = 37
            request_path = output/'batch37_request.json'; result_path = output/'batch37.json'
            write_json(request_path, request)
            subprocess.run([str(Path(args.executable).resolve()), str(Path(args.model).resolve()),
                str(request_path.resolve()), str(result_path.resolve())], cwd=ROOT, env=env, check=True)
            small = json.loads(result_path.read_text())['samples']
            for a, b in zip(small, results['torch_cuda']):
                if a['hit'] != b['hit'] or a['initial_sign'] != b['initial_sign']:
                    raise AssertionError('batch-size-dependent event')
                if abs(a['q']-b['q']) > 3e-5 or abs(a['speed']-b['speed']) > 5e-4:
                    raise AssertionError('batch-size-dependent sample')
    # The wrapper must reject changes to the trained recurrent step.
    request['maximum_step'] = .0625
    bad = output/'wrong_step_request.json'; write_json(bad, request)
    rejected = subprocess.run([str(Path(args.executable).resolve()), str(Path(args.model).resolve()),
        str(bad.resolve()), str((output/'wrong_step.json').resolve())], cwd=ROOT, env=env, capture_output=True, text=True)
    if rejected.returncode == 0 or 'fixed by training' not in rejected.stderr:
        raise AssertionError('recurrent step override was not rejected')
    summary = {'passed': True, 'model_sha256': digest(Path(args.model)),
        'dataset_sha256': digest(Path(args.dataset)), 'device': torch.cuda.get_device_name(0),
        'sampling_contract': 'entry branch, exponential optical depth, positive mixture speed',
        'batch_sizes_checked': [37, 256], 'step_override_rejected': True,
        'elapsed_seconds': time.perf_counter()-start, 'rows': rows}
    write_json(output/'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='outputs/models/first_hit_causal_gru_sigma002/first_hit_model.json')
    parser.add_argument('--dataset', default='outputs/first_hit_multiview_sigma002_20261015/dataset.pt')
    parser.add_argument('--executable', default='build/Release/macrofacet_query_first_hit_se.exe')
    parser.add_argument('--output', default='outputs/first_hit_causal_render_sigma002_20261016/parity')
    parser.add_argument('--samples', type=int, default=512)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
