"""Frozen gradient-head accuracy under changes to the internal recurrent step.

Reference first-hit locations, signs, speeds and cubic profiles stay fixed.
The state/phase hybrids are diagnostic interventions, not usable integrators.
"""
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
import time

import numpy as np
import torch

from renewal.collect import digest, resolve, write_json
from .causal_data import to_device
from .causal_gradient import cache_profile, points
from .causal_gradient_diagnostics import averaged_cdf
from .causal_gradient_report import audit as audit_parent
from .causal_gradient_study import backbone_path, evaluate, load_head
from .causal_study import load_run as load_backbone
from .data import load_data, read_json
from .hazard_study import parameter_hash


@contextmanager
def recurrent_step(backbone, step):
    original = backbone.config
    try:
        backbone.config = replace(original, step=step)
        yield
    finally:
        backbone.config = original


def contexts(backbone, raw, step, device):
    with recurrent_step(backbone, step):
        rows = []
        for i, profile in enumerate(raw):
            rows.append(to_device(cache_profile(backbone, profile), device))
            if (i+1) % 70 == 0:
                print(f'contexts step={step:g}: {i+1}/{len(raw)} rays', flush=True)
    return rows


def assert_same_observations(base, changed):
    if len(base) != len(changed):
        raise AssertionError('different reference rays')
    for a, b in zip(base, changed):
        for key in ('entry', 'q', 'b', 'db', 'sign', 'hit', 'w', 'horizon'):
            torch.testing.assert_close(a[key], b[key], atol=0, rtol=0, equal_nan=True)


def hybrid_contexts(base, changed, mode):
    assert_same_observations(base, changed)
    if mode == 'state_only':
        # State and its matching index must move together; phase stays at training step.
        return [{**b, 'phase': a['phase'], 'joint_distance_sum': a['joint_distance_sum']}
                for a, b in zip(base, changed)]
    if mode == 'phase_only':
        return [{**a, 'phase': b['phase']} for a, b in zip(base, changed)]
    raise ValueError('unknown diagnostic intervention')


def conditional_metrics(metrics):
    # A distance likelihood cannot be attributed to the gradient head, especially
    # for deliberately inconsistent state/phase hybrid interventions.
    result = {k: v for k, v in metrics.items() if k != 'joint_nll'}
    result['per_ray'] = [{k: v for k, v in row.items() if k != 'joint_nll'}
                         for row in metrics['per_ray']]
    return result


def maximum_numeric_difference(a, b):
    if isinstance(a, dict):
        if a.keys() != b.keys():
            raise AssertionError('metric keys changed')
        return max((maximum_numeric_difference(a[k], b[k]) for k in a), default=0.)
    if isinstance(a, list):
        if len(a) != len(b):
            raise AssertionError('metric length changed')
        return max((maximum_numeric_difference(x, y) for x, y in zip(a, b)), default=0.)
    if a is None or isinstance(a, str):
        if a != b:
            raise AssertionError('metric labels changed')
        return 0.
    return abs(float(a)-float(b))


def register(spec, parent, output):
    parent_output = resolve(parent['output_directory'])
    checks, trained, previous, _ = audit_parent(parent, parent_output)
    original = read_json(parent_output/'protocol.json')
    if original['specification'] != parent:
        raise ValueError('parent configuration changed')
    steps = spec['steps']
    if (len(steps) != len(set(steps)) or spec['training_step'] not in steps
            or not set(spec['ablation_steps']+spec['example_steps']).issubset(steps)
            or spec['ablation_seed'] not in parent['seeds'] or spec['example_seed'] not in parent['seeds']
            or not set(spec['example_methods']).issubset(v['name'] for v in parent['variants'])
            or spec['minimum_ray_hits'] < 1 or spec['speed_grid_points'] < 2):
        raise ValueError('invalid step experiment configuration')
    sources = dict(original['source_sha256'])
    for name in ('causal_gradient_step.py', 'causal_gradient_diagnostics.py'):
        path = Path(__file__).with_name(name)
        sources[str(path)] = digest(path)
    checkpoints = dict(original['checkpoint_sha256'])
    checkpoints.update(previous['checkpoint_sha256'])
    identity = {'specification': spec, 'source_sha256': sources, 'checkpoint_sha256': checkpoints,
        'parent_sha256': {name: digest(parent_output/name) for name in
                          ('protocol.json', 'training.json', 'evaluation.json', 'diagnostics.json')},
        'dataset_sha256': original['dataset_sha256'], 'parent_audit': checks,
        'scope': 'Fixed reference hits; frozen weights; changed internal q-step; no retraining, distance resampling or rendering.',
        'selection': 'No model/step selection using these test results.',
        'torch_version': str(torch.__version__), 'cuda_version': torch.version.cuda,
        'device': parent['device'], 'cpu_threads': parent['cpu_threads'], 'tf32': False}
    path = output/'protocol.json'
    if path.exists() and read_json(path) != identity:
        raise ValueError('step experiment identity changed; use a new output directory')
    if not path.exists():
        write_json(path, identity)
    return previous


@torch.no_grad()
def example_cdfs(head, profiles, grid, ids):
    examples = []
    for i in ids:
        p = points([profiles[i]])
        params = head(p)
        reference = [(p['w'] <= w).double().mean().item() for w in grid] if len(p['w']) else None
        predicted = averaged_cdf(head.config.kind, params, torch.ones_like(p['w'], dtype=torch.bool), grid)
        examples.append({'ray': i, 'hits': len(p['w']), 'reference_cdf': reference, 'predicted_cdf': predicted,
            'max_grid_CDF_error': float(np.max(np.abs(np.array(predicted)-reference))) if reference else None})
    return examples


def evaluate_record(head, profiles, spec, output, seed, method, step, mode, grid, previous):
    identity = {'protocol_sha256': digest(output/'protocol.json'), 'seed': seed,
                'method': method, 'step': step, 'mode': mode}
    path = output/'runs'/f'{method}_{seed}_step{step:g}_{mode}.json'
    if path.exists():
        record = read_json(path)
        if record['identity'] != identity:
            raise ValueError('cached evaluation identity changed')
        return record
    began = time.perf_counter()
    score = conditional_metrics(evaluate(head, profiles))
    before = next(r['metrics'] for r in previous['results'][method] if r['seed'] == seed)
    if score['hits'] != before['hits'] or score['samples'] != before['samples']:
        raise AssertionError('reference labels changed')
    difference = None
    if mode == 'full' and step == spec['training_step']:
        difference = maximum_numeric_difference(score, conditional_metrics(before))
        if difference > spec['baseline_tolerance']:
            raise AssertionError(f'training-step reproduction failed: {difference}')
    curves = []
    if (mode == 'full' and seed == spec['example_seed'] and method in spec['example_methods']
            and step in spec['example_steps']):
        curves = example_cdfs(head, profiles, grid, spec['example_rays'])
    record = {'identity': identity, 'metrics': score, 'examples': curves,
              'baseline_max_metric_difference': difference, 'elapsed_seconds': time.perf_counter()-began}
    write_json(path, record)
    eligible = [r['pit_ks'] for r in score['per_ray'] if r['hits'] >= spec['minimum_ray_hits']]
    print(f'{method}/{seed} step={step:g} {mode}: NLL={score["speed_nll"]:.6f}, '
          f'mean ray PIT={np.mean(eligible):.6f}, late PIT={score["distance_bins"][-1]["pit_ks"]:.6f}', flush=True)
    return record


def evaluate_steps(spec, parent, previous, data, output):
    raw = [p for p in data['profiles'] if p['split'] == 'test']
    training_w = torch.cat([p['speed'][p['hit']] for p in data['profiles'] if p['split'] == 'train'])
    grid = torch.linspace(0, float(torch.quantile(training_w, .995))*1.15,
                          spec['speed_grid_points'], dtype=torch.float64, device=parent['device'])
    records, configs, weight_checks = [], {}, []
    order = [spec['training_step']]+[s for s in spec['steps'] if s != spec['training_step']]
    for seed in parent['seeds']:
        backbone, _ = load_backbone(backbone_path(parent, seed), 'cpu')
        backbone.requires_grad_(False)
        if backbone.config.step != spec['training_step']:
            raise ValueError('unexpected trained recurrent step')
        for step in spec['steps']:
            replace(backbone.config, step=step)  # Validate without changing integration accuracy.
        configs[str(seed)] = backbone.configuration()
        before = parameter_hash(backbone)
        heads = {v['name']: load_head(resolve(parent['output_directory'])/'runs'/f'{v["name"]}_{seed}'/'best.pt',
                                      parent['device'])[0].requires_grad_(False) for v in parent['variants']}
        head_hash = {name: parameter_hash(head) for name, head in heads.items()}
        base = contexts(backbone, raw, spec['training_step'], parent['device'])
        for step in order:
            changed = base if step == spec['training_step'] else contexts(backbone, raw, step, parent['device'])
            assert_same_observations(base, changed)
            for method, head in heads.items():
                records.append(evaluate_record(head, changed, spec, output, seed, method, step, 'full', grid, previous))
            if seed == spec['ablation_seed'] and step in spec['ablation_steps']:
                for mode in ('state_only', 'phase_only'):
                    hybrid = hybrid_contexts(base, changed, mode)
                    for method, head in heads.items():
                        record = evaluate_record(head, hybrid, spec, output, seed, method, step, mode, grid, previous)
                        if mode == 'state_only' and not head.config.history:
                            baseline = next(r for r in records if r['identity']['seed'] == seed
                                and r['identity']['method'] == method and r['identity']['step'] == spec['training_step']
                                and r['identity']['mode'] == 'full')
                            if maximum_numeric_difference(record['metrics'], baseline['metrics']) > spec['baseline_tolerance']:
                                raise AssertionError('no-history head unexpectedly uses recurrent state')
                        records.append(record)
                    del hybrid
            del changed
        if before != parameter_hash(backbone) or backbone.configuration() != configs[str(seed)]:
            raise AssertionError('backbone changed')
        for name, head in heads.items():
            if parameter_hash(head) != head_hash[name]:
                raise AssertionError('gradient head changed')
        weight_checks.append({'seed': seed, 'backbone': before, 'heads': head_hash, 'unchanged': True})
        del base, heads, backbone
    register(spec, parent, output)  # Recheck immutable input files after evaluation.
    result = {'protocol_sha256': digest(output/'protocol.json'), 'records': records,
        'backbone_configs': configs, 'weight_checks': weight_checks, 'groups': previous['groups'],
        'speed_grid': grid.cpu().tolist(), 'reference_rays': len(raw),
        'reference_hits': sum(int(p['hit'].sum()) for p in raw),
        'reference_sampler': data['reference_sampler']}
    write_json(output/'evaluation.json', result)
    return result


def run(config_path, phase='all'):
    spec = read_json(resolve(config_path))
    parent = read_json(resolve(spec['gradient_config']))
    output = resolve(spec['output_directory'])
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(parent['cpu_threads'])
    torch.backends.cuda.matmul.allow_tf32 = False
    previous = register(spec, parent, output)
    if phase in ('all', 'evaluate'):
        result = evaluate_steps(spec, parent, previous, load_data(resolve(parent['dataset'])), output)
    if phase in ('all', 'report'):
        from .causal_gradient_step_report import report
        result = report(spec, parent, output)
    else:
        result = {'evaluations': len(result['records']), 'output': str(output/'evaluation.json')}
    return result
