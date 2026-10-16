"""Validation-selected residual-L2 and survival-curve loss ablation."""
import math
from pathlib import Path

import numpy as np
import torch

from renewal.collect import digest, resolve, write_json
from .data import load_data, read_json
from .distance_study import evaluate, load_run, train_one
from .hazard_study import prepare
from .multiview_study import test_groups


def variant_name(eta, beta):
    number = lambda v: format(v, '.5g').replace('.', 'p').replace('-', 'm')
    return f'l2_{number(eta)}_T_{number(beta)}'


def ray_metrics(rows, tail_fraction=.1):
    if not rows or not 0 < tail_fraction <= 1:
        raise ValueError('ray metrics require rays and a positive tail fraction')
    error = np.array([r['max_abs_T_error'] for r in rows])
    count = max(1, math.ceil(tail_fraction*len(error)))
    return {'rays': len(rows), 'distance_nll': float(np.mean([r['distance_nll'] for r in rows])),
        'mean_ray_max_abs_T_error': float(error.mean()),
        'p95_ray_max_abs_T_error': float(np.quantile(error, .95)),
        'worst_decile_mean': float(np.sort(error)[-count:].mean()),
        'max_abs_T_error': float(error.max()),
        'rmse_T': float(np.sqrt(np.mean([r['mse_T'] for r in rows]))),
        'mean_abs_escape_error': float(np.mean([abs(r['escape_error']) for r in rows]))}


def choose_candidate(candidates, baseline, selection):
    threshold = baseline['distance_nll']+selection['maximum_validation_nll_increase']
    eligible = [r for r in candidates if r['validation']['distance_nll'] <= threshold]
    if eligible:
        chosen = min(eligible, key=lambda r: (r['validation']['worst_decile_mean'],
            r['validation']['mean_ray_max_abs_T_error'], r['validation']['distance_nll'], r['name']))
    else:
        # Preserve the failed guard in the report; still evaluate the closest
        # candidate as an experiment, without presenting it as an accepted gain.
        chosen = min(candidates, key=lambda r: r['validation']['distance_nll'])
    return {'name': chosen['name'], 'eta': chosen['eta'], 'beta': chosen['beta'],
            'validation': chosen['validation'], 'nll_guard_passed': bool(eligible),
            'validation_nll_limit': threshold, 'candidate_names': [r['name'] for r in candidates],
            'eligible_names': [r['name'] for r in eligible]}


def protocol(spec, base, output):
    source = Path(__file__).parent
    files = [source/name for name in ('distance_objective.py', 'distance_study.py', 'distance_models.py',
             'hazard_study.py', 'hazard.py', 'spline.py', 'residual_supervision.py')]
    prior = resolve(spec['base_directory'])
    checkpoints = [prior/'runs'/f'residual_{seed}'/'best.pt' for seed in spec['seeds']]
    current = {'specification': spec, 'base_configuration': base, 'dataset_sha256': digest(resolve(base['dataset'])),
        'training_source_sha256': {str(p): digest(p) for p in files},
        'baseline_checkpoint_sha256': {str(p): digest(p) for p in checkpoints},
        'test_access': 'test metrics evaluated only after weights and selected checkpoints are frozen',
        'fixed_failure_examples': [266, 84, 175]}
    if current['dataset_sha256'] != spec['dataset_sha256']:
        raise ValueError('the fixed multiview dataset has changed')
    if spec['selection']['pilot_seed'] not in spec['seeds'] or spec['selection']['tail_fraction'] != .1:
        raise ValueError('invalid pilot seed or unsupported tail fraction')
    if spec['selection']['checkpoint_rule'] != 'minimum_validation_distance_nll':
        raise ValueError('unsupported checkpoint selection rule')
    if spec['selection']['weight_rule'] != 'minimum_validation_worst_decile_mean_subject_to_nll_guard':
        raise ValueError('unsupported objective-weight selection rule')
    path = output/'protocol.json'
    if path.exists() and read_json(path) != current:
        raise ValueError('experiment inputs or training implementation differ from the registered protocol')
    write_json(path, current)
    return current


def configuration(spec, base, output, eta, beta):
    return {**base, 'device': spec['device'], 'seeds': spec['seeds'], 'epochs': spec['epochs'],
            'output_directory': str(output/'variants'/variant_name(eta, beta)),
            'distance_objective': {'residual_l2_weight': eta, 'transmittance_weight': beta},
            'residual_supervision_specification': spec}


def validation_result(path, profiles, device, tail_fraction, destination):
    identity = {'checkpoint_sha256': digest(path), 'tail_fraction': tail_fraction}
    if destination.exists():
        saved = read_json(destination)
        if saved['identity'] != identity:
            raise ValueError('validation cache belongs to a different checkpoint')
        return saved['metrics']
    model, _ = load_run(path, device)
    score = evaluate(model, profiles, curves=True)
    result = {'identity': identity, 'metrics': ray_metrics(score['per_ray'], tail_fraction),
              'per_ray': [{k: v for k, v in p.items() if k != 'T'} for p in score['per_ray']]}
    write_json(destination, result)
    return result['metrics']


def train(spec, base, data, output, identity):
    # No test labels enter either optimization or checkpoint/weight selection.
    filtered = {'profiles': [p for p in data['profiles'] if p['split'] != 'test']}
    splits = {k: v for k, v in prepare(filtered, base, spec['device']).items() if v}
    tail = spec['selection']['tail_fraction']
    pilot = spec['selection']['pilot_seed']
    prior = resolve(spec['base_directory'])
    baseline = {}
    for seed in spec['seeds']:
        source = prior/'runs'/f'residual_{seed}'/'best.pt'
        baseline[str(seed)] = validation_result(source, splits['validation'], spec['device'], tail,
                                               output/'baseline_validation'/f'{seed}.json')
    def fit(eta, beta, seed):
        name = variant_name(eta, beta)
        config = configuration(spec, base, output, eta, beta)
        directory = output/'variants'/name
        print(f'training {name}, seed={seed}', flush=True)
        report = train_one(config, identity['dataset_sha256'], splits, 'residual', seed, directory)
        reference = read_json(prior/'runs'/f'residual_{seed}'/'run_report.json')
        if report['initial_parameter_sha256'] != reference['initial_parameter_sha256']:
            raise AssertionError('initial parameters differ from the paired baseline')
        if 'test' in report['metrics']:
            raise AssertionError('test labels were evaluated before selection was frozen')
        checkpoint = directory/'runs'/f'residual_{seed}'/'best.pt'
        val = validation_result(checkpoint, splits['validation'], spec['device'], tail,
                                checkpoint.parent/'validation_details.json')
        print(f"validation {name}/{seed}: NLL={val['distance_nll']:.6f}, tail10={val['worst_decile_mean']:.6f}", flush=True)
        return {'name': name, 'eta': eta, 'beta': beta, 'seed': seed, 'checkpoint': str(checkpoint),
                'checkpoint_sha256': digest(checkpoint), 'validation': val,
                'selected_epoch': report['selected_epoch'], 'parameters': report['parameters'],
                'training_seconds': report['training_seconds']}
    l2_candidates = [fit(float(eta), 0., pilot) for eta in spec['residual_l2_candidates']]
    l2 = choose_candidate(l2_candidates, baseline[str(pilot)], spec['selection'])
    curve_candidates = [fit(l2['eta'], float(beta), pilot) for beta in spec['transmittance_candidates']]
    curve = choose_candidate(curve_candidates, baseline[str(pilot)], spec['selection'])
    selection = {'protocol_sha256': digest(output/'protocol.json'), 'pilot_seed': pilot,
        'baseline_validation': baseline[str(pilot)], 'regularized': l2, 'regularized_T': curve,
        'candidates': l2_candidates+curve_candidates, 'test_metrics_used': False}
    selection_path = output/'selection.json'
    if selection_path.exists() and read_json(selection_path) != selection:
        raise ValueError('validation-only selection changed')
    write_json(selection_path, selection)
    print(f"selected eta={l2['eta']}, beta={curve['beta']} using validation only", flush=True)
    selected = [r for r in selection['candidates'] if r['name'] in (l2['name'], curve['name'])]
    for seed in spec['seeds']:
        if seed != pilot:
            selected += [fit(v['eta'], v['beta'], seed) for v in (l2, curve)]
    result = {'selection_sha256': digest(selection_path), 'selected_runs': selected,
              'baseline_validation': baseline, 'new_training_runs': len(selection['candidates'])+2*(len(spec['seeds'])-1)}
    write_json(output/'training.json', result)
    return result


def evaluate_selected(spec, base, data, output):
    selection, training = read_json(output/'selection.json'), read_json(output/'training.json')
    if training['selection_sha256'] != digest(output/'selection.json'):
        raise ValueError('selection changed after confirmation training')
    profiles = prepare({'profiles': [p for p in data['profiles'] if p['split'] == 'test']}, base, spec['device'])['test']
    results, hashes = {}, {}
    prior = resolve(spec['base_directory'])
    for method in ('local', 'baseline', 'regularized', 'regularized_T'):
        results[method] = []
        for seed in ([None] if method == 'local' else spec['seeds']):
            model = None
            if method == 'baseline':
                checkpoint = prior/'runs'/f'residual_{seed}'/'best.pt'
            elif seed is not None:
                name = selection[method]['name']
                record = next(r for r in training['selected_runs'] if r['name'] == name and r['seed'] == seed)
                checkpoint = Path(record['checkpoint'])
                if digest(checkpoint) != record['checkpoint_sha256']:
                    raise ValueError('selected model changed before test evaluation')
            if seed is not None:
                hashes[str(checkpoint)] = digest(checkpoint)
                model, _ = load_run(checkpoint, spec['device'])
            result = evaluate(model, profiles, curves=True)
            result.update(ray_metrics(result['per_ray'], spec['selection']['tail_fraction']))
            results[method].append({'seed': seed, 'metrics': result})
            print(f"fixed test {method}/{seed}: mean max T={result['mean_ray_max_abs_T_error']:.6f}, tail10={result['worst_decile_mean']:.6f}, worst={result['max_abs_T_error']:.6f}", flush=True)
    groups, summary = test_groups(data), {}
    for name, indices in groups.items():
        summary[name] = {}
        for method, runs in results.items():
            values = [ray_metrics([r['metrics']['per_ray'][i] for i in indices], spec['selection']['tail_fraction']) for r in runs]
            summary[name][method] = {metric: {'mean': float(np.mean([r[metric] for r in values])),
                'seed_std': float(np.std([r[metric] for r in values]))} for metric in values[0] if metric != 'rays'}
    result = {'dataset_sha256': digest(resolve(base['dataset'])), 'selection_sha256': digest(output/'selection.json'),
              'checkpoint_sha256': hashes, 'groups': groups, 'summary': summary, 'results': results}
    write_json(output/'evaluation.json', result)
    return result


def run(config_path, phase='all'):
    spec = read_json(resolve(config_path))
    output = resolve(spec['output_directory'])
    output.mkdir(parents=True, exist_ok=True)
    base = read_json(resolve(spec['base_directory'])/'training_config.json')
    torch.set_num_threads(base['cpu_threads'])
    identity = protocol(spec, base, output)
    data = load_data(resolve(base['dataset']))
    if phase in ('all', 'train'):
        result = train(spec, base, data, output, identity)
        if phase == 'train':
            return {'new_training_runs': result['new_training_runs'], 'selection': str(output/'selection.json')}
    if phase in ('all', 'evaluate'):
        result = evaluate_selected(spec, base, data, output)
        if phase == 'evaluate':
            return result['summary']
    if phase in ('all', 'report'):
        from .residual_supervision_report import report
        result = report(spec, base, data, output)
    for path, expected in identity['baseline_checkpoint_sha256'].items():
        if digest(Path(path)) != expected:
            raise AssertionError('baseline checkpoint changed')
    if digest(resolve(base['dataset'])) != identity['dataset_sha256']:
        raise AssertionError('fixed dataset changed')
    return result
