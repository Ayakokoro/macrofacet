"""Paired multiview distance training, then frozen conditional-gradient heads."""
from pathlib import Path

import torch

from renewal.collect import digest, resolve, write_json
from .camera_holdout import group_metrics
from .data import load_data, read_json
from .distance_study import evaluate, load_run, train_one
from .hazard import local_bin_log_rates, piecewise_distribution
from .hazard_study import prepare


def configuration(spec, output):
    base = read_json(resolve(spec['original_study']))
    return {**base, 'dataset': str(output/'dataset.pt'), 'output_directory': str(output),
            'collection_config': str(output/'collection_config.json'),
            'epochs': spec['epochs'], 'device': spec['device'], 'seeds': spec['seeds'],
            'multiview_specification': spec}


def test_groups(raw):
    profiles = [p for p in raw['profiles'] if p['split'] == 'test']
    return {'all_280': list(range(len(profiles))),
        'original_24': [i for i, p in enumerate(profiles) if p['scene_id'] == 'shader_ball_sigma002_camera_only'],
        **{group: [i for i, p in enumerate(profiles) if p['scene_id'].endswith('_'+group)]
           for group in ('spatial', 'diverse')}}


def train_distance(spec, config, data, output):
    splits = prepare(data, config, config['device'])
    data_hash = digest(output/'dataset.pt')
    reports = []
    for seed in spec['seeds']:
        for mode in ('direct', 'residual'):
            reports.append(train_one(config, data_hash, splits, mode, seed, output))
    write_json(output/'distance_training.json', reports)
    return {'runs': len(reports), 'dataset_sha256': data_hash}


def evaluate_distance(spec, config, data, output):
    base = read_json(resolve(spec['original_study']))
    profiles = prepare({'profiles': [p for p in data['profiles'] if p['split'] == 'test']}, config, config['device'])['test']
    results, hashes = {}, {}
    for name in ('local', 'old_direct', 'old_residual', 'direct', 'residual'):
        results[name] = []
        for seed in ([None] if name == 'local' else spec['seeds']):
            model = None
            if seed is not None:
                original = name.startswith('old_')
                directory = resolve(base['output_directory']) if original else output
                path = directory/'runs'/f'{name.removeprefix("old_")}_{seed}'/'best.pt'
                hashes[str(path)] = digest(path)
                model, _ = load_run(path, config['device'])
            result = evaluate(model, profiles, curves=True)
            results[name].append({'seed': seed, 'metrics': result})
            print(f"fixed test {name}/{seed}: mean max T={result['mean_ray_max_abs_T_error']:.6f}", flush=True)
    groups = test_groups(data)
    evaluation = {'dataset_sha256': digest(output/'dataset.pt'), 'checkpoint_sha256': hashes,
                  'results': results, 'groups': groups}
    evaluation['summary'] = {group: group_metrics(evaluation, ids) for group, ids in groups.items()}
    write_json(output/'distance_evaluation.json', evaluation)
    audit_local_grid(config, data, output)
    return {'groups': {k: len(v) for k, v in groups.items()}, 'summary': evaluation['summary']}


@torch.no_grad()
def audit_local_grid(config, data, output):
    """Record discretization sensitivity without changing the paired experiment."""
    rows = []
    for p in (p for p in data['profiles'] if p['split'] == 'validation'):
        q = torch.linspace(0, p['horizon'], 4099, dtype=torch.float64)
        prior = torch.special.log_ndtr(p['features'][0, 0]*torch.tensor([1., -1.], dtype=torch.float64))
        prior -= torch.logsumexp(prior, 0)
        def curve(bins, order):
            rates = local_bin_log_rates(p['features'], bins, order)
            H = piecewise_distribution(rates, q[None], p['horizon'])[0]
            return torch.logsumexp(prior[:, None]-H, 0).exp()
        baseline = curve(config['model']['bins'], config['model']['quadrature_order'])
        rows.append({'source': p['source'],
                     'grid_error': float((baseline-curve(config['model']['bins']*2, 16)).abs().max()),
                     'two_point_quadrature_error': float((baseline-curve(config['model']['bins'], 2)).abs().max())})
    result = {'rays': len(rows), 'scope': 'local-only sensitivity on expanded validation; grid retained for matched training',
        'max_grid_error': max(r['grid_error'] for r in rows),
        'max_two_point_quadrature_error': max(r['two_point_quadrature_error'] for r in rows),
        'previous_validation_tolerance': config['max_baseline_T_discretization_error'],
        'rays_above_previous_grid_tolerance': sum(r['grid_error'] > config['max_baseline_T_discretization_error'] for r in rows),
        'per_ray': rows}
    write_json(output/'local_grid_audit.json', result)
    print(f"expanded validation local-grid sensitivity: {result['max_grid_error']:.6g}", flush=True)
    return result


def run(config_path, phase='all'):
    spec = read_json(resolve(config_path))
    output = resolve(spec['output_directory'])
    config = configuration(spec, output)
    torch.set_num_threads(config['cpu_threads'])
    write_json(output/'training_config.json', config)
    data = load_data(output/'dataset.pt')
    audit = read_json(output/'data_audit.json')
    if digest(output/'dataset.pt') != audit['dataset_sha256']:
        raise ValueError('dataset changed after the audit')
    if phase in ('all', 'distance'):
        result = train_distance(spec, config, data, output)
        if phase == 'distance':
            return result
    if phase in ('all', 'evaluate'):
        result = evaluate_distance(spec, config, data, output)
        if phase == 'evaluate':
            return result
    if phase in ('all', 'gradient'):
        from .frozen_gradient_study import run as run_gradient
        result = run_gradient(spec, config, data, output)
        if phase == 'gradient':
            return result
    if phase in ('all', 'sampling'):
        from .multiview_sampling import sampling_checks
        result = sampling_checks(spec, config, data, output)
        if phase == 'sampling':
            return result
    if phase in ('all', 'report'):
        from .multiview_report import report
        return report(spec, config, data, output)
    raise ValueError(phase)
