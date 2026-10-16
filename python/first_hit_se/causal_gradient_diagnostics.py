"""Conditional CDFs, reference-position sampling, and gradient-head causality."""
from pathlib import Path

import numpy as np
import torch

from renewal.collect import digest, resolve, write_json
from renewal.profiles import subdivide_features
from .causal_gradient import points, query_points
from .causal_gradient_study import backbone_path, load_head
from .causal_study import load_run as load_backbone
from .data import read_json
from .distance_models import latent
from .distance_probe import ks
from .distance_study import alter_suffix, restrict_profile, load_run as load_mlp
from .frozen_gradient_study import load_head as load_old_head
from .gradient_cdf import cdf, sample
from .spline import local_profile, resample_profile


def models_for(spec, output):
    seed = spec['diagnostics']['seed']
    backbone, _ = load_backbone(backbone_path(spec, seed), 'cpu')
    backbone.requires_grad_(False)
    old_root = resolve(spec['baseline_directory'])
    config = read_json(old_root/'training_config.json')
    old_spec = config['multiview_specification']
    old_source = old_root/'runs'/f'residual_{seed}'/'best.pt'
    old_model, _ = load_mlp(old_source, 'cpu')
    old_model.requires_grad_(False)
    heads = {v['name']: load_head(output/'runs'/f"{v['name']}_{seed}"/'best.pt')[0] for v in spec['variants']}
    kinds = {v['name']: v['kind'] for v in spec['variants']}
    for kind in ('mixture', 'cdf'):
        saved = torch.load(old_root/'gradient'/f'{kind}_{seed}'/'best.pt', weights_only=True, map_location='cpu')
        if saved['identity']['source_sha256'] != digest(old_source):
            raise ValueError('old gradient head backbone changed')
        heads['mlp_'+kind] = load_old_head(old_spec, config, old_root, kind, seed, 'cpu')
        kinds['mlp_'+kind] = kind
    return backbone, old_model, heads, kinds


@torch.no_grad()
def old_points(model, features, q, sign):
    fixed = resample_profile(features, model.config.profile_points)
    signs = fixed.new_tensor([1., -1.])
    z = latent(model, fixed[None].expand(2, -1), signs)[(sign < 0).long()]
    b, db = local_profile(features, q)
    return {'z': z, 'q': q, 'horizon': features[:, 4].exp().sum().expand_as(q), 'sign': sign, 'b': b, 'db': db}


def parameters(head, p, old=False):
    return head(*(p[k] for k in ('z', 'q', 'horizon', 'sign', 'b', 'db'))) if old else head(p)


def on_device(p, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in p.items()}


def cdf_grid(kind, params, grid):
    return cdf(kind, tuple(v[:, None] for v in params), grid[None])


def averaged_cdf(kind, params, mask, grid):
    selected = tuple(v[mask] for v in params)
    result = torch.zeros_like(grid)
    n = int(mask.sum())
    for begin in range(0, n, 256):
        result += cdf_grid(kind, tuple(v[begin:begin+256] for v in selected), grid).sum(0)
    return (result/n).cpu().tolist() if n else None


@torch.no_grad()
def diagnose(spec, data, output):
    backbone, old_model, heads, kinds = models_for(spec, output)
    raw = [p for p in data['profiles'] if p['split'] == 'test']
    training_w = torch.cat([p['speed'][p['hit']] for p in data['profiles'] if p['split'] == 'train'])
    cut_probabilities = torch.tensor([.1, .25, .5, .75, .9, .99], dtype=torch.float64)
    cuts = torch.quantile(training_w, cut_probabilities)
    grid = torch.linspace(0, float(torch.quantile(training_w, .995))*1.15,
                          spec['diagnostics']['speed_grid_points'], dtype=torch.float64)
    consistency = []
    def query_all(f, q, sign):
        new = query_points(backbone, f, q, sign)
        old = old_points(old_model, f, q, sign)
        return {name: cdf_grid(kinds[name], parameters(head, old if name.startswith('mlp_') else new,
                                                     name.startswith('mlp_')), cuts)
                for name, head in heads.items()}
    for i, p in enumerate(raw):
        f, Q = p['features'], p['features'][:, 4].exp().sum()
        q = torch.linspace(0, 1, 33, dtype=torch.float64)*Q
        q = q.repeat_interleave(2)
        sign = q.new_tensor([1., -1.]).repeat(33)
        full = query_all(f, q, sign)
        variations = [('repartition', subdivide_features(f, torch.full((len(f),), 4, dtype=torch.long)), torch.ones_like(q, dtype=torch.bool))]
        for fraction in spec['diagnostics']['prefix_fractions']:
            variations.append((f'prefix_{fraction}', restrict_profile(f, fraction), q <= Q*fraction+1e-12))
        variations.append(('future_change', alter_suffix(f, .5), q <= Q*.5+1e-12))
        for name, altered, mask in variations:
            changed = query_all(altered, q[mask], sign[mask])
            for method in heads:
                consistency.append({'ray': i, 'kind': name, 'method': method,
                    'max_CDF_difference': float((changed[method]-full[method][mask]).abs().max())})
        if (i+1) % 40 == 0:
            print(f'gradient prefix checks: {i+1}/{len(raw)} rays', flush=True)
    summary = {kind: {name: max(r['max_CDF_difference'] for r in consistency if r['kind'] == kind and r['method'] == name)
        for name in heads} for kind in sorted({r['kind'] for r in consistency})}
    causal_max = max(v for methods in summary.values() for name, v in methods.items() if not name.startswith('mlp_'))
    if causal_max > spec['diagnostics']['cdf_tolerance']:
        raise ArithmeticError(f'conditional CDF causality failed: {causal_max}')
    seed = spec['diagnostics']['seed']
    cached = torch.load(output/f'contexts_{seed}_test.pt', weights_only=True)['rows']
    evaluation = read_json(output/'evaluation.json')
    minimum = spec['diagnostics']['minimum_ray_hits']
    scores = []
    for i in range(len(raw)):
        rows = [r['metrics']['per_ray'][i] for r in evaluation['results']['mixture_gru']]
        if rows[0]['hits'] >= minimum:
            scores.append((float(np.mean([r['pit_ks'] for r in rows])), i))
    worst = [i for _, i in sorted(scores, reverse=True)[:3]]
    # Post-evaluation diagnostic: distinguish low-count PIT fluctuations from
    # errors on rays with enough observations. Never used to select a model.
    well_sampled_minimum = 1024
    well_sampled_worst = [i for _, i in sorted(scores, reverse=True)
                         if evaluation['results']['mixture_gru'][0]['metrics']['per_ray'][i]['hits'] >= well_sampled_minimum][:3]
    examples = sorted(set(spec['diagnostics']['fixed_rays']+worst+well_sampled_worst))
    device = spec['device']
    for head in heads.values():
        head.to(device)
    grid_device, cuts_device = grid.to(device), cuts.to(device)
    sample_pit = {name: [] for name in heads}
    probability_rows, example_rows = [], []
    generator = torch.Generator().manual_seed(seed+410000)
    for i, (raw_profile, row) in enumerate(zip(raw, cached)):
        p_cpu = points([row])
        n = len(p_cpu['w'])
        if not n:
            continue
        if n > spec['diagnostics']['conditional_samples_per_ray']:
            raise ValueError('reference hits exceed the configured conditional sampling budget')
        p = on_device(p_cpu, device)
        old = on_device(old_points(old_model, raw_profile['features'], p_cpu['q'], p_cpu['sign']), device)
        uniforms = torch.rand(n, 2, dtype=torch.float64, generator=generator).to(device)
        observed = (p['w'][:, None] <= cuts_device[None]).double().mean(0)
        record = {'ray': i, 'hits': n, 'reference_cdf': observed.cpu().tolist(), 'methods': {}}
        example = {'ray': i, 'conditions': []}
        if i in examples:
            for label, mask in (('all_hits', torch.ones(n, dtype=torch.bool, device=device)), ('late_half', p['fraction'] >= .5)):
                empirical = (torch.searchsorted(p['w'][mask].sort().values, grid_device, right=True).double()/int(mask.sum())).cpu().tolist() if mask.any() else None
                example['conditions'].append({'name': label, 'hits': int(mask.sum()), 'reference_cdf': empirical, 'predicted': {}})
        for name, head in heads.items():
            params = parameters(head, old if name.startswith('mlp_') else p, name.startswith('mlp_'))
            predicted = cdf_grid(kinds[name], params, cuts_device).mean(0)
            draws = sample(kinds[name], params, uniforms)
            if not torch.isfinite(draws).all() or (draws < 0).any():
                raise FloatingPointError('invalid conditional gradient samples')
            sampled = (draws[:, None] <= cuts_device[None]).double().mean(0)
            sample_pit[name].append(cdf(kinds[name], params, draws).cpu())
            record['methods'][name] = {'predicted_cdf': predicted.cpu().tolist(), 'sampled_cdf': sampled.cpu().tolist(),
                'max_reference_CDF_error': float((predicted-observed).abs().max()),
                'max_sampling_CDF_error': float((sampled-predicted).abs().max())}
            if i in examples:
                for condition in example['conditions']:
                    mask = torch.ones(n, dtype=torch.bool, device=device) if condition['name'] == 'all_hits' else p['fraction'] >= .5
                    condition['predicted'][name] = averaged_cdf(kinds[name], params, mask, grid_device)
        probability_rows.append(record)
        if i in examples:
            example_rows.append(example)
        if (i+1) % 40 == 0:
            print(f'gradient conditional CDF/sampling checks: {i+1}/{len(raw)} rays', flush=True)
    probability_summary = {name: {
        'mean_ray_max_reference_CDF_error': float(np.mean([r['methods'][name]['max_reference_CDF_error'] for r in probability_rows if r['hits'] >= minimum])),
        'sample_pit_ks': ks(torch.cat(sample_pit[name])),
        'sample_count': sum(len(v) for v in sample_pit[name]),
        'mean_ray_max_sampling_CDF_error': float(np.mean([r['methods'][name]['max_sampling_CDF_error'] for r in probability_rows if r['hits'] >= minimum]))}
        for name in heads}
    result = {'scope': 'first seed; observed reference hit positions and entry signs; no distance resampling or rendering',
        'evaluation_sha256': digest(output/'evaluation.json'), 'training_sha256': digest(output/'training.json'),
        'source_sha256': digest(Path(__file__)), 'consistency_max': summary, 'causal_max': causal_max,
        'consistency_rows': consistency, 'probability_summary': probability_summary, 'probability_rows': probability_rows,
        'threshold_training_quantiles': cut_probabilities.tolist(), 'speed_thresholds': cuts.tolist(),
        'speed_grid': grid.tolist(), 'fixed_rays': spec['diagnostics']['fixed_rays'], 'worst_rays': worst,
        'well_sampled_worst_rays': well_sampled_worst, 'well_sampled_minimum_hits': well_sampled_minimum,
        'examples': example_rows, 'passed': True}
    write_json(output/'diagnostics.json', result)
    return {'diagnostics': str(output/'diagnostics.json'), 'causal_max': causal_max, 'probability_summary': probability_summary}
