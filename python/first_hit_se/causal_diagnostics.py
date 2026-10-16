"""Prefix causality, history continuation, step sensitivity, and scoped timings."""
from dataclasses import replace
from pathlib import Path
import statistics
import time

import numpy as np
import torch

from renewal.collect import digest, resolve, write_json
from renewal.profiles import subdivide_features
from .causal_data import batch_profiles, geometry, queries, query_model, to_device
from .causal_residual import CausalResidual, branch_values
from .causal_study import load_run
from .data import read_json
from .distance_models import parameters, transmittance
from .distance_study import alter_suffix, restrict_profile, load_run as load_mlp
from .hazard_study import prepare as prepare_mlp, make_batch as mlp_batch
from .residual_supervision import ray_metrics


def models_for(spec, output, device='cpu'):
    seed = spec['diagnostics']['seed']
    models = {v['name']: load_run(output/'runs'/f"{v['name']}_{seed}"/'best.pt', device)[0] for v in spec['variants']}
    models['mlp_nll'] = load_mlp(resolve(spec['baseline_directory'])/'runs'/f'residual_{seed}'/'best.pt', device)[0]
    selected = read_json(resolve(spec['supervision_directory'])/'selection.json')['regularized_T']['name']
    models['mlp_T'] = load_mlp(resolve(spec['supervision_directory'])/'variants'/selected/'runs'/f'residual_{seed}'/'best.pt', device)[0]
    capacity = output/'capacity_control'/'runs'/f'local_matched_T_{seed}'/'best.pt'
    if capacity.exists():
        models['local_matched_T'] = load_run(capacity, device)[0]
    return models


def query(model, features, q, config=None):
    return query_model(model, features, q, config)['T'] if isinstance(model, CausalResidual) else model.query(features, q)['T']


def metric_row(T, reference):
    error = T-reference
    return {'max_abs_T_error': float(error.abs().max()), 'mse_T': float(error.square().mean()),
            'escape_error': float(error[-1])}


def summary_of(records):
    result = {}
    for kind in sorted({r['kind'] for r in records}):
        result[kind] = {}
        for method in sorted({r['method'] for r in records if r['kind'] == kind}):
            values = [r['max_T_difference'] for r in records if r['kind'] == kind and r['method'] == method]
            result[kind][method] = {'mean': float(np.mean(values)), 'p95': float(np.quantile(values, .95)),
                                     'max': float(max(values)), 'rays': len(values)}
    return result


def time_call(call, device='cpu', repeats=5, inner=1):
    def sync():
        if str(device).startswith('cuda'):
            torch.cuda.synchronize()
    for _ in range(3):
        call()
    times = []
    for _ in range(repeats):
        sync()
        start = time.perf_counter()
        for _ in range(inner):
            call()
        sync()
        times.append((time.perf_counter()-start)*1e6/inner)
    return {'median_us': statistics.median(times), 'runs_us': times}


@torch.no_grad()
def benchmark(spec, raw, output):
    ids = [int(i) for i in np.linspace(0, len(raw)-1, 16)]
    chosen = [raw[i] for i in ids]
    rows = torch.load(output/'prepared_test.pt', weights_only=True)['rows']
    models = models_for(spec, output)
    base = read_json(resolve(spec['baseline_directory'])/'training_config.json')
    old = prepare_mlp({'profiles': chosen}, base, 'cpu')['test']
    curve = torch.linspace(0, 1, spec['curve_points'], dtype=torch.float64)[None]
    timing = {}
    for name in (n for n in ('mlp_T', 'local_head_T', 'local_matched_T', 'gru_T') if n in models):
        model = models[name]
        causal = isinstance(model, CausalResidual)
        batches = [batch_profiles([rows[i]]) for i in ids] if causal else [mlp_batch([p]) for p in old]
        def infer(batch):
            if causal:
                H = branch_values(model(batch), batch, batch['curve'])[0]
                return torch.logsumexp(batch['log_prior'][:, :, None]-H, 1).exp()
            rates = parameters(model, batch)
            prior = torch.special.log_ndtr(batch['b0'][:, None]*batch['b0'].new_tensor([1., -1.]))
            prior -= torch.logsumexp(prior, -1, keepdim=True)
            return transmittance(model, rates, curve*batch['horizon'][:, None], batch['horizon'], prior)
        def prepared_call():
            return [infer(b) for b in batches]
        def full_call():
            return [query(model, p['features'], curve[0]*p['horizon']) for p in chosen]
        timing[name] = {kind: {**value, 'median_us_per_ray': value['median_us']/len(ids)}
                       for kind, value in (('cpu_prepared_curve', time_call(prepared_call)),
                                           ('cpu_from_cubics_curve', time_call(full_call)))}
        if causal:
            g = geometry(chosen[len(chosen)//2]['features'], model.config)
            state = model.initial_state(g['entry'])
            mid = len(g['widths'])//2
            def step_call():
                entering = model.advance(state, g['tokens'][:, mid]) if model.config.history else state
                return model.correction(entering, g['left'][:, min(mid+1, len(g['widths'])-1)])
            timing[name]['cpu_cached_state_one_step_two_signs'] = time_call(step_call, inner=64)
        if str(spec['device']).startswith('cuda'):
            model.to('cuda')
            curve = curve.to('cuda')
            # The same 16 rays repeated eight times; padding and both signs included.
            gpu_batch = batch_profiles([to_device(rows[i], 'cuda') for i in ids]*8) if causal else mlp_batch([to_device(p, 'cuda') for p in old]*8)
            measurement = time_call(lambda: infer(gpu_batch), 'cuda', inner=5)
            timing[name]['cuda_128_prepared_curve'] = {**measurement, 'median_us_per_ray': measurement['median_us']/128}
            model.cpu()
            curve = curve.cpu()
    return {'scope': '513-point transmittance curve for both entry signs; no VDB access, distance sampling, gradient sampling or rendering',
        'cpu_threads': torch.get_num_threads(), 'torch': str(torch.__version__), 'ray_indices': ids,
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'timings': timing}


@torch.no_grad()
def diagnose(spec, data, output):
    trained = read_json(output/'training.json')
    evaluation = read_json(output/'evaluation.json')
    if evaluation['training_sha256'] != digest(output/'training.json'):
        raise ValueError('training selection changed before diagnostics')
    models = models_for(spec, output)
    raw = [p for p in data['profiles'] if p['split'] == 'test']
    records, stream, step_rows = [], [], {}
    for i, p in enumerate(raw):
        f = p['features']
        Q = f[:, 4].exp().sum()
        q = torch.linspace(0, 1, spec['curve_points'], dtype=torch.float64)*Q
        reference = 1-torch.searchsorted(p['distance'][p['hit']].clamp_max(Q).sort().values, q, right=True).double()/len(p['hit'])
        partitioned = subdivide_features(f, torch.full((len(f),), spec['diagnostics']['subdivisions'], dtype=torch.long))
        future = alter_suffix(f, spec['diagnostics']['future_fraction'])
        for name, model in models.items():
            full = query(model, f, q)
            different = query(model, partitioned, q)
            records.append({'ray': i, 'method': name, 'kind': 'repartition', 'max_T_difference': float((full-different).abs().max())})
            for fraction in spec['diagnostics']['prefix_fractions']:
                subset = q <= Q*fraction+1e-12
                cut = query(model, restrict_profile(f, fraction), q[subset])
                records.append({'ray': i, 'method': name, 'kind': f'prefix_{fraction}',
                                'max_T_difference': float((full[subset]-cut).abs().max())})
            subset = q <= Q*spec['diagnostics']['future_fraction']+1e-12
            altered = query(model, future, q[subset])
            records.append({'ray': i, 'method': name, 'kind': 'future_change',
                            'max_T_difference': float((full[subset]-altered).abs().max())})
            if isinstance(model, CausalResidual):
                for step in spec['diagnostics']['steps']:
                    changed = full if step == model.config.step else query(model, f, q, replace(model.config, step=step))
                    records.append({'ray': i, 'method': name, 'kind': f'step_{step}',
                                    'max_T_difference': float((full-changed).abs().max())})
                    step_rows.setdefault(name, {}).setdefault(str(step), []).append(metric_row(changed, reference))
        model = models['gru_T']
        geo = geometry(f, model.config)
        state, deltas = model.initial_state(geo['entry']), []
        for j in range(len(geo['widths'])):
            deltas.append(model.correction(state, geo['left'][:, j]))
            state = model.advance(state, geo['tokens'][:, j])
        delta = torch.stack(deltas, -1)[None]
        batch = {k: geo[k][None] for k in ('tokens', 'left', 'entry', 'mass')}
        H = branch_values(delta, batch, {k: v[None] for k, v in queries(geo, q).items()})[0]
        T = torch.logsumexp(geo['log_prior'][None, :, None]-H, 1).exp()[0]
        direct = query(model, f, q)
        stream.append({'ray': i, 'max_T_difference': float((T-direct).abs().max()),
                       'max_delta_difference': float((delta-model(batch)).abs().max())})
        if (i+1) % 20 == 0 or i+1 == len(raw):
            print(f'causality/step diagnostics: {i+1}/{len(raw)} rays', flush=True)
    summary = summary_of(records)
    causal_names = [name for name, model in models.items() if isinstance(model, CausalResidual)]
    causal_max = max(value['max'] for kind, methods in summary.items() if not kind.startswith('step_')
                     for name, value in methods.items() if name in causal_names)
    stream_max = max(r['max_T_difference'] for r in stream)
    if max(causal_max, stream_max) > spec['diagnostics']['causal_tolerance']:
        raise ArithmeticError('causal or streaming consistency tolerance failed')
    quadrature = []
    validation = [p for p in data['profiles'] if p['split'] == 'validation']
    for i, p in enumerate(validation):
        q = torch.linspace(0, 1, spec['curve_points'], dtype=torch.float64)*p['features'][:, 4].exp().sum()
        for name in causal_names:
            model = models[name]
            fine = replace(model.config, integration_step=model.config.integration_step/2, quadrature_order=16)
            difference = (query(model, p['features'], q)-query(model, p['features'], q, fine)).abs().max()
            quadrature.append({'ray': i, 'method': name, 'max_T_difference': float(difference)})
    quad_max = max(r['max_T_difference'] for r in quadrature)
    if quad_max > spec['diagnostics']['quadrature_tolerance']:
        raise ArithmeticError('local quadrature tolerance failed')
    step_accuracy = {method: {step: {
        'mean_ray_max_abs_T_error': float(np.mean([r['max_abs_T_error'] for r in rows])),
        'max_abs_T_error': max(r['max_abs_T_error'] for r in rows),
        'rmse_T': float(np.sqrt(np.mean([r['mse_T'] for r in rows])))} for step, rows in values.items()}
        for method, values in step_rows.items()}
    old = evaluation['results']['local_mlp_grid'][0]['metrics']['per_ray']
    new = evaluation['results']['local_causal_grid'][0]['metrics']['per_ray']
    local_difference = [float(np.max(np.abs(np.array(a['T'])-b['T']))) for a, b in zip(old, new)]
    print(f'causal max={causal_max:.3g}, streaming={stream_max:.3g}, quadrature={quad_max:.3g}', flush=True)
    timing = benchmark(spec, raw, output)
    result = {'scope': 'first-seed checkpoints; fixed diagnostic rays; no further training or selection',
        'training_sha256': digest(output/'training.json'), 'evaluation_sha256': digest(output/'evaluation.json'),
        'capacity_evaluation_sha256': digest(output/'capacity_control'/'evaluation.json') if (output/'capacity_control'/'evaluation.json').exists() else None,
        'source_sha256': digest(Path(__file__)), 'summary': summary, 'records': records,
        'streaming': {'max_T_difference': stream_max, 'per_ray': stream},
        'quadrature': {'max_T_difference': quad_max, 'per_ray': quadrature},
        'step_accuracy': step_accuracy, 'step_per_ray': step_rows,
        'local_grid_difference': {'mean': float(np.mean(local_difference)), 'max': max(local_difference), 'per_ray': local_difference},
        'passed': True, 'benchmark': timing}
    write_json(output/'diagnostics.json', result)
    return result
