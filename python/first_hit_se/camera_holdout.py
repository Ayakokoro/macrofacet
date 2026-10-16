"""Geometry-selected camera holdout; existing model checkpoints stay fixed.

Spatial coverage and deliberately diverse profiles are reported separately.
Selection never uses reference samples, model predictions, or model errors.
"""
from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
import csv
import html
import math
from pathlib import Path
import subprocess

import numpy as np
import torch

from renewal.collect import digest, resolve, write_json
from renewal.reference_sampler import require_same_sampler
from .camera_study import plotting, save_figure, write_csv
from .data import collect, load_data, prepare as prepare_manifest, read_json
from .distance_models import MODES
from .distance_study import evaluate, load_run
from .hazard_study import prepare
from .spline import local_profile


def camera_position(position, target, azimuth_offset_degrees, elevation_degrees=None):
    """Orbit the scene's z axis while preserving camera radius and target."""
    offset = np.asarray(position, dtype=float)-np.asarray(target, dtype=float)
    radius = np.linalg.norm(offset)
    if not radius > 0:
        raise ValueError('camera position equals target')
    if azimuth_offset_degrees == 0 and elevation_degrees is None:
        return list(position)
    azimuth = math.atan2(offset[1], offset[0])+math.radians(azimuth_offset_degrees)
    elevation = (math.asin(offset[2]/radius) if elevation_degrees is None
                 else math.radians(elevation_degrees))
    return (np.asarray(target)+radius*np.array([
        math.cos(elevation)*math.cos(azimuth), math.cos(elevation)*math.sin(azimuth),
        math.sin(elevation)])).tolist()


def ray_key(ray):
    return tuple(round(float(v), 10) for key in ('entry_origin', 'direction') for v in ray[key])


def describe(features, horizon, points=129, weights=(1., .5, 1.)):
    """Dimensionless shape distance: RMS(asinh mean), slope, log horizon."""
    q = torch.linspace(0, horizon, points, dtype=torch.float64)
    b, db = (v.numpy() for v in local_profile(features, q))
    descriptor = np.concatenate((weights[0]*np.arcsinh(b)/math.sqrt(points),
                                 weights[1]*np.arcsinh(db)/math.sqrt(points),
                                 [weights[2]*math.log1p(horizon)]))
    # Diagnostics use a denser grid than selection; crossing counts are sampled.
    dense = torch.linspace(0, horizon, 2049, dtype=torch.float64)
    mean, slope = (v.numpy() for v in local_profile(features, dense))
    stats = {'horizon': horizon, 'entry_mean_sigma': float(mean[0]),
             'minimum_mean_sigma': float(mean.min()), 'maximum_mean_sigma': float(mean.max()),
             'sampled_zero_crossings': int(np.count_nonzero(np.signbit(mean[1:]) != np.signbit(mean[:-1]))),
             'near_zero_fraction': float(np.mean(np.abs(mean) <= 2)),
             'max_abs_slope': float(np.max(np.abs(slope)))}
    return descriptor, stats


def nearest_distance(values, anchors):
    if len(anchors) == 0:
        raise ValueError('profile selection requires existing anchors')
    # Chunk both dimensions to bound memory for larger candidate collections.
    result = np.full(len(values), np.inf)
    for begin in range(0, len(values), 128):
        block = values[begin:begin+128]
        for start in range(0, len(anchors), 256):
            delta = block[:, None]-anchors[None, start:start+256]
            result[begin:begin+len(block)] = np.minimum(
                result[begin:begin+len(block)], np.square(delta).sum(-1).min(-1))
    return np.sqrt(result)


def diverse_indices(values, anchors, count):
    """Greedy max-min coverage relative to existing profiles and new selections."""
    if not 0 <= count <= len(values):
        raise ValueError('insufficient distinct candidates for diversity selection')
    score = nearest_distance(values, anchors)
    selected = []
    for _ in range(count):
        index = int(np.argmax(score))
        selected.append(index)
        score = np.minimum(score, np.linalg.norm(values-values[index], axis=1))
        score[selected] = -np.inf
    return selected


def candidates(width, height, grid, seed, excluded=()):
    rng = np.random.default_rng(seed)
    blocked = set(excluded)
    pixels = []
    for y in range(grid):
        for x in range(grid):
            choices = [(px, py) for py in range(y*height//grid, (y+1)*height//grid)
                       for px in range(x*width//grid, (x+1)*width//grid)
                       if (px, py) not in blocked]
            if not choices:
                raise ValueError('candidate cell has no unused pixel')
            pixel = choices[int(rng.integers(len(choices)))]
            pixels.append(pixel)
            blocked.add(pixel)
    return pixels


def source_identity(config, base):
    scene = read_json(resolve(base['render_config']))
    return {'config': config, 'base_config': base,
            'old_dataset_sha256': digest(resolve(base['dataset'])),
            'scene_sha256': digest(resolve(base['render_config'])),
            'field_sha256': digest(resolve(scene['field']['grid_file'])),
            'inspector_sha256': digest(resolve('build/Release/macrofacet_inspect_first_hit_se.exe')),
            'selection_source_sha256': digest(Path(__file__))}


def select(config, base, output):
    identity = source_identity(config, base)
    path = output/'selection.json'
    if path.exists():
        selected = read_json(path)
        if selected['identity'] != identity:
            raise ValueError('holdout identity changed; choose a new output directory')
        return selected
    old = load_data(resolve(base['dataset']))
    old_rays = [read_json(Path(p['source']).parent/'ray.json') for p in old['profiles']]
    occupied = {ray_key(r) for r in old_rays}
    descriptor = lambda p: describe(p['features'], p['horizon'],
                                    config['descriptor_points'], config['descriptor_weights'])
    old_descriptors = np.stack([descriptor(p)[0] for p in old['profiles']])
    training = old_descriptors[[p['split'] == 'train' for p in old['profiles']]]
    anchors = old_descriptors.copy()
    old_requests = read_json(resolve(base['collection_config']))
    excluded_pixels = {tuple(p) for source in old_requests['sources'] for p in source['pixels']}
    scene = read_json(resolve(base['render_config']))
    width, height = scene['render']['width'], scene['render']['height']
    all_rows, selected_rows, camera_records = [], [], []
    inspector = resolve('build/Release/macrofacet_inspect_first_hit_se.exe')
    for camera_index, spec in enumerate(config['cameras']):
        camera = copy.deepcopy(scene)
        camera['render']['camera_position'] = camera_position(
            scene['render']['camera_position'], scene['render']['camera_target'],
            spec['azimuth_offset_degrees'], spec.get('elevation_degrees'))
        folder = output/'candidates'/spec['id']
        camera_path = output/'cameras'/f"{spec['id']}.json"
        camera['output_directory'] = str(output/'unused_render'/spec['id'])
        write_json(camera_path, camera)
        same_camera = camera['render']['camera_position'] == scene['render']['camera_position']
        pixels = candidates(width, height, config['candidate_grid'], config['seed']+camera_index,
                            excluded_pixels if same_camera else ())

        def inspect(item):
            index, pixel = item
            dest = folder/f'{index:04d}'
            stamp = {'camera': digest(camera_path), 'pixel': list(pixel),
                     'field': identity['field_sha256'], 'inspector': identity['inspector_sha256']}
            cached = dest/'inspected.json'
            if not cached.exists() or read_json(cached) != stamp:
                result = subprocess.run([str(inspector), str(camera_path), str(dest),
                                         str(pixel[0]), str(pixel[1])], capture_output=True, text=True)
                if result.returncode:
                    if result.stderr.strip() == 'ray misses medium':
                        return {'camera': spec['id'], 'pixel': list(pixel), 'status': 'domain_miss'}
                    raise RuntimeError(f'camera extraction failed: {pixel}: {result.stderr}')
                write_json(cached, stamp)
            ray = read_json(dest/'ray.json')
            if ray_key(ray) in occupied:
                return {'camera': spec['id'], 'pixel': list(pixel), 'status': 'duplicate_ray'}
            raw = read_json(dest/'mean_profile.json')
            raw['features'] = torch.tensor(raw['features'], dtype=torch.float64)
            vector, stats = descriptor(raw)
            return {'camera': spec['id'], 'pixel': list(pixel), 'status': 'valid',
                    'descriptor': vector.tolist(), **stats, 'geometry_directory': str(dest),
                    'render_config': str(camera_path), 'ray_key': list(ray_key(ray))}

        with ThreadPoolExecutor(max_workers=config['collection_workers']) as pool:
            inspected = list(pool.map(inspect, enumerate(pixels)))
        all_rows.extend(inspected)
        valid = [r for r in inspected if r['status'] == 'valid']
        rng = np.random.default_rng(config['seed']+1000+camera_index)
        coverage, grid = [], config['coverage_grid']
        for cy in range(grid):
            for cx in range(grid):
                options = [i for i, r in enumerate(valid)
                           if r['pixel'][0]*grid//width == cx and r['pixel'][1]*grid//height == cy]
                if options:
                    coverage.append(int(rng.choice(options)))
        # If a view has an entirely empty coarse cell, fill from other valid rays.
        remaining = [i for i in range(len(valid)) if i not in coverage]
        if len(coverage) < grid*grid:
            coverage.extend(rng.choice(remaining, size=grid*grid-len(coverage), replace=False).tolist())
        selected_here = []
        for i in coverage:
            selected_here.append({**valid[i], 'group': 'spatial'})
        values = np.asarray([r['descriptor'] for r in selected_here])
        anchors = np.concatenate((anchors, values))
        remaining = [r for i, r in enumerate(valid) if i not in coverage]
        values = np.asarray([r['descriptor'] for r in remaining])
        for i in diverse_indices(values, anchors, config['diverse_per_camera']):
            selected_here.append({**remaining[i], 'group': 'diverse'})
        anchors = np.concatenate((anchors, [r['descriptor'] for r in selected_here if r['group'] == 'diverse']))
        for row in selected_here:
            key = tuple(row['ray_key'])
            if key in occupied:
                raise ValueError('selected camera rays overlap')
            occupied.add(key)
            row['nearest_training_profile_distance'] = float(nearest_distance(
                np.asarray(row['descriptor'])[None], training)[0])
        selected_rows.extend(selected_here)
        camera_records.append({'id': spec['id'], 'position': camera['render']['camera_position'],
            'target': camera['render']['camera_target'], 'render_config': str(camera_path),
            'valid_candidates': len(valid), 'selected': len(selected_here)})
        print(f"selected {spec['id']}: {len(valid)}/{len(pixels)} valid, {len(selected_here)} held-out rays", flush=True)
    result = {'identity': identity, 'selection_uses': 'geometry only; no labels or model predictions',
        'cameras': camera_records, 'candidate_count': len(all_rows), 'candidates': all_rows,
        'selected': selected_rows}
    collection = copy.deepcopy(old_requests)
    collection.update(seed=config['seed']+1000000, output_directory=str(output/'data'),
        realizations_per_profile=config['realizations_per_profile'], threads=config['threads'],
        collection_workers=config['collection_workers'], skip_domain_misses=False)
    collection['sources'] = []
    for camera in config['cameras']:
        for group in ('spatial', 'diverse'):
            selected = [r for r in selected_rows if r['camera'] == camera['id'] and r['group'] == group]
            collection['sources'].append({'scene_id': f"sigma002_{camera['id']}_{group}",
                'split': 'test', 'render_config': selected[0]['render_config'],
                'realizations_per_profile': config['realizations_per_profile'],
                'pixels': [r['pixel'] for r in selected]})
    write_json(output/'collection_config.json', collection)
    write_json(path, result)
    return result


def audit_and_combine(config, base, output):
    old_path, new_path = resolve(base['dataset']), output/'data'/'dataset.pt'
    old, new = load_data(old_path), load_data(new_path)
    require_same_sampler(old['reference_sampler'], new['reference_sampler'], 'additional camera holdout')
    selection = read_json(output/'selection.json')
    if digest(old_path) != selection['identity']['old_dataset_sha256']:
        raise ValueError('original training dataset changed')
    expected = {(r['camera'], tuple(r['pixel'])): r for r in selection['selected']}
    keys, seeds, rows = set(), set(), []
    for origin, data in (('original', old), ('new', new)):
        for profile in data['profiles']:
            ray = read_json(Path(profile['source']).parent/'ray.json')
            key = ray_key(ray)
            if key in keys:
                raise ValueError('physical camera ray leakage')
            keys.add(key)
            if origin == 'new':
                if profile['split'] != 'test':
                    raise ValueError('new holdout was assigned to training or validation')
                camera = Path(ray['source_config']).stem
                selected = expected.pop((camera, tuple(ray['pixel'])))
                source_camera = read_json(ray['source_config'])['render']
                expected_origin = np.asarray(source_camera['camera_position'])+ray['entry_distance']*np.asarray(ray['direction'])
                if not np.allclose(expected_origin, ray['entry_origin'], rtol=0, atol=1e-9):
                    raise ValueError('reference ray is not the requested camera ray')
                raw = read_json(Path(selected['geometry_directory'])/'mean_profile.json')
                torch.testing.assert_close(profile['features'], torch.tensor(raw['features'], dtype=torch.float64), rtol=1e-10, atol=1e-10)
                if not math.isclose(profile['geometry']['sigma'], .02, abs_tol=1e-9):
                    raise ValueError('holdout field is not sigma=0.02')
                rows.append({k: v for k, v in selected.items() if k not in ('descriptor', 'ray_key')}
                            | {'profile': profile['id'], 'samples': len(profile['hit']),
                               'hits': int(profile['hit'].sum()),
                               'negative_starts': int((profile['initial_sign'] < 0).sum())})
            with (Path(profile['source'])/'first_passage_samples.csv').open(encoding='utf-8', newline='') as stream:
                for row in csv.DictReader(stream):
                    seed = int(row['seed'])
                    if seed in seeds:
                        raise ValueError('reference RNG seed reused across old/new rays')
                    seeds.add(seed)
    if expected:
        raise ValueError('some selected rays have no labels')
    write_csv(output/'test_rays.csv', rows)
    manifest = read_json(old_path.parent/'manifest.json')
    manifest['sources'] += read_json(new_path.parent/'manifest.json')['sources']
    manifest['parent_datasets'] = [{'path': str(p), 'sha256': digest(p)} for p in (old_path, new_path)]
    manifest['collection_config_sha256'] = digest(output/'collection_config.json')
    manifest['skipped_domain_misses'] = []
    write_json(output/'combined_manifest.json', manifest)
    combined = prepare_manifest(output/'combined_manifest.json', output/'combined_dataset.pt')
    audit = {'new_test_rays': len(rows), 'new_samples': sum(r['samples'] for r in rows),
        'new_hits': sum(r['hits'] for r in rows), 'new_negative_starts': sum(r['negative_starts'] for r in rows),
        'unique_physical_rays': len(keys), 'unique_realization_seeds': len(seeds),
        'original_dataset_unchanged': digest(old_path) == selection['identity']['old_dataset_sha256'],
        'original_dataset_sha256': digest(old_path), 'new_dataset_sha256': digest(new_path),
        'reference_sampler': new['reference_sampler'], 'combined': combined}
    write_json(output/'audit.json', audit)
    return audit


def evaluate_holdout(config, base, output):
    raw = load_data(output/'data'/'dataset.pt')
    rows = read_json(output/'selection.json')['selected']
    mapping = {(r['camera'], tuple(r['pixel'])): r for r in rows}
    metadata = []
    for p in raw['profiles']:
        ray = read_json(Path(p['source']).parent/'ray.json')
        metadata.append(mapping[Path(ray['source_config']).stem, tuple(ray['pixel'])])
    prepared = prepare(raw, base, config['device'])['test']
    results, hashes = {}, {}
    for mode in ('local', *MODES):
        seeds = [None] if mode == 'local' else base['seeds']
        runs = []
        for seed in seeds:
            model = None
            if seed is not None:
                checkpoint = resolve(base['output_directory'])/'runs'/f'{mode}_{seed}'/'best.pt'
                hashes[str(checkpoint)] = digest(checkpoint)
                model, bundle = load_run(checkpoint, config['device'])
                if bundle['dataset_sha256'] != digest(resolve(base['dataset'])):
                    raise ValueError('checkpoint does not match the original training dataset')
            scores = evaluate(model, prepared, curves=True)
            runs.append({'seed': seed, 'metrics': scores})
            print(f"holdout {mode}/{seed}: mean max T={scores['mean_ray_max_abs_T_error']:.5f}, NLL={scores['distance_nll']:.5f}", flush=True)
        results[mode] = runs
    for path, expected in hashes.items():
        if digest(Path(path)) != expected:
            raise ValueError('existing checkpoint changed during evaluation')
    result = {'dataset_sha256': digest(output/'data'/'dataset.pt'), 'checkpoint_sha256': hashes,
              'selection': 'original validation-selected checkpoints; no retraining or reselection',
              'profiles': [{k: v for k, v in m.items() if k != 'descriptor'} for m in metadata],
              'results': results}
    write_json(output/'evaluation.json', result)
    return result


def group_metrics(evaluation, indices):
    result = {}
    for mode, runs in evaluation['results'].items():
        values = []
        for run in runs:
            rows = [run['metrics']['per_ray'][i] for i in indices]
            values.append({'distance_nll': float(np.mean([r['distance_nll'] for r in rows])),
                'mean_ray_max_abs_T_error': float(np.mean([r['max_abs_T_error'] for r in rows])),
                'worst_T_error': max(r['max_abs_T_error'] for r in rows),
                'rmse_T': float(np.sqrt(np.mean([r['mse_T'] for r in rows]))),
                'escape_error': float(np.mean([abs(r['escape_error']) for r in rows]))})
        result[mode] = {key: {'mean': float(np.mean([r[key] for r in values])),
                             'seed_std': float(np.std([r[key] for r in values]))} for key in values[0]}
    return result


def report(config, base, output):
    selection, audit, evaluation = (read_json(output/name) for name in
                                    ('selection.json', 'audit.json', 'evaluation.json'))
    metadata = evaluation['profiles']
    old, new = load_data(resolve(base['dataset'])), load_data(output/'data'/'dataset.pt')
    describe_one = lambda p: describe(p['features'], p['horizon'], config['descriptor_points'], config['descriptor_weights'])
    training = np.stack([describe_one(p)[0] for p in old['profiles'] if p['split'] == 'train'])
    old_test = [p for p in old['profiles'] if p['split'] == 'test']
    old_description = [describe_one(p) for p in old_test]
    old_distance = nearest_distance(np.stack([v[0] for v in old_description]), training)
    groups = {'all_new': list(range(len(metadata))),
              **{g: [i for i, m in enumerate(metadata) if m['group'] == g] for g in ('spatial', 'diverse')},
              **{c['id']: [i for i, m in enumerate(metadata) if m['camera'] == c['id']] for c in config['cameras']}}
    metrics = {name: group_metrics(evaluation, indices) for name, indices in groups.items()}
    def distribution(values):
        return {'min': float(np.min(values)), 'median': float(np.median(values)),
                'p95': float(np.quantile(values, .95)), 'max': float(np.max(values))}
    diversity = {'old_test_nearest_training': distribution(old_distance),
        'new_nearest_training': {name: distribution([metadata[i]['nearest_training_profile_distance'] for i in indices])
                                 for name, indices in groups.items()},
        'profile_ranges': {key: {'old_test': distribution([v[1][key] for v in old_description]),
                                'new_test': distribution([r[key] for r in metadata])}
                           for key in ('horizon', 'entry_mean_sigma', 'minimum_mean_sigma',
                                       'sampled_zero_crossings', 'near_zero_fraction', 'max_abs_slope')}}
    write_json(output/'summary.json', {'audit': audit, 'group_sizes': {g: len(i) for g, i in groups.items()},
                                      'metrics': metrics, 'diversity': diversity})
    plt = plotting()
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=True)
    for ax, spec in zip(axes.flat, config['cameras']):
        for i in groups[spec['id']]:
            p = new['profiles'][i]
            u = torch.linspace(0, 1, 513, dtype=torch.float64)
            mean, _ = local_profile(p['features'], u*p['horizon'])
            ax.plot(u, mean, color='#2563eb' if metadata[i]['group'] == 'spatial' else '#e87918', alpha=.45, lw=.8)
        ax.axhline(0, color='black', lw=.7)
        ax.set_title(spec['id'])
        ax.set_ylim(-25, 40)
        ax.set_xlabel('q / horizon')
        ax.set_ylabel('mean / sigma')
    fig.suptitle('New camera profiles: spatial (blue), diverse (orange); vertical display clipped')
    fig.tight_layout()
    save_figure(plt, fig, output, 'mean_profiles')
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    labels = ['old test', 'new spatial', 'new diverse']
    values = [old_distance]+[np.array([metadata[i]['nearest_training_profile_distance'] for i in groups[g]]) for g in ('spatial', 'diverse')]
    axes[0].boxplot(values, tick_labels=labels)
    axes[0].set_ylabel('Distance to nearest training profile')
    axes[1].boxplot([[p['horizon'] for p in old_test]]+[[new['profiles'][i]['horizon'] for i in groups[g]] for g in ('spatial', 'diverse')], tick_labels=labels)
    axes[1].set_ylabel('Dimensionless horizon')
    for k, mode in enumerate(evaluation['results']):
        x = np.arange(2)+k*.19
        axes[2].bar(x, [metrics[g][mode]['mean_ray_max_abs_T_error']['mean'] for g in ('spatial', 'diverse')], width=.18, label=mode)
    axes[2].set_xticks(np.arange(2)+.285, ['spatial', 'diverse'])
    axes[2].set_ylabel('Mean per-ray max |T error|')
    axes[2].legend()
    fig.tight_layout()
    save_figure(plt, fig, output, 'coverage_and_errors')
    # Fixed geometry-selected examples: each view's farthest profile from training.
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharey=True)
    for ax, spec in zip(axes.flat, config['cameras']):
        i = max(groups[spec['id']], key=lambda j: metadata[j]['nearest_training_profile_distance'])
        p = new['profiles'][i]
        q = torch.linspace(0, p['horizon'], base['curve_points'], dtype=torch.float64)
        reference = 1-torch.searchsorted(p['distance'][p['hit']].sort().values, q, right=True).double()/len(p['hit'])
        ax.plot(q*p['geometry']['ell'], reference, 'k', lw=2, label='reference')
        for mode, runs in evaluation['results'].items():
            curves = np.array([r['metrics']['per_ray'][i]['T'] for r in runs])
            ax.plot(q*p['geometry']['ell'], curves.mean(0), label=mode)
            ax.fill_between(q*p['geometry']['ell'], curves.min(0), curves.max(0), alpha=.10)
        ax.set_title(f"{spec['id']} {metadata[i]['pixel']}")
        ax.set_xlabel('Distance from medium entry')
        ax.set_ylabel('Transmittance')
    axes.flat[0].legend(fontsize=8)
    fig.suptitle('Farthest profile per view selected by geometry; bands span training seeds')
    fig.tight_layout()
    save_figure(plt, fig, output, 'holdout_curves')
    table = ['<tr><th>Group</th><th>Rays</th><th>Local</th><th>Direct</th><th>Residual</th><th>CDF</th></tr>']
    for name, indices in groups.items():
        cells = ''.join(f"<td>{metrics[name][mode]['mean_ray_max_abs_T_error']['mean']:.6f}</td>" for mode in ('local', *MODES))
        table.append(f'<tr><td>{html.escape(name)}</td><td>{len(indices)}</td>{cells}</tr>')
    document = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Expanded camera holdout, sigma=0.02</title>
<style>body{{font:16px system-ui;max-width:1300px;margin:32px auto;padding:0 20px}}img{{width:100%}}td,th{{padding:8px 12px;border-bottom:1px solid #ddd;text-align:right}}table{{border-collapse:collapse}}</style>
<h1>Expanded camera holdout, sigma=0.02</h1>
<p>{len(metadata)} new rays, {audit['new_samples']:,} samples, 8 views. Existing training/validation data and checkpoints are unchanged.</p>
<p>Spatial coverage uses one valid candidate per coarse image cell. Diverse profiles use geometry-only max-min selection. These are diagnostic cohorts, not an unbiased distribution of render pixels. All cameras view the same scene.</p>
<h2>Mean per-ray maximum transmittance error</h2><p>Neural entries average the three original validation-selected seeds.</p><table>{''.join(table)}</table>
<h2>Profile diversity</h2><img src="coverage_and_errors.png"><img src="mean_profiles.png">
<h2>Geometry-selected examples</h2><img src="holdout_curves.png">
<p>Reference: fixed H=1/B=8 Tungsten-linear sampler, step 1/64, 8192 realizations per ray. No new gradient-CDF models, joint sampling, or rendering are tested here.</p>
<p><a href="summary.json">Summary</a> · <a href="test_rays.csv">Ray metadata</a> · <a href="collection_config.json">Collection config</a> · <a href="audit.json">Audit</a></p></html>'''
    (output/'summary.html').write_text(document, encoding='utf-8')
    return {'report': str(output/'summary.html'), 'dataset': str(output/'combined_dataset.pt'),
            'new_rays': len(metadata), 'metrics': metrics['all_new'], 'diversity': diversity}


def run(config_path, phase='all'):
    config = read_json(resolve(config_path))
    base = read_json(resolve(config['base_study']))
    output = resolve(config['output_directory'])
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(base['cpu_threads'])
    if phase in ('all', 'select'):
        selected = select(config, base, output)
        if phase == 'select':
            return {'candidates': selected['candidate_count'], 'selected': len(selected['selected'])}
    if phase in ('all', 'collect'):
        collect(output/'collection_config.json')
        audit = audit_and_combine(config, base, output)
        if phase == 'collect':
            return audit
    if phase in ('all', 'evaluate'):
        result = evaluate_holdout(config, base, output)
        if phase == 'evaluate':
            return {'models': list(result['results']), 'rays': len(result['profiles'])}
    if phase in ('all', 'report'):
        return report(config, base, output)
    raise ValueError(f'unknown phase: {phase}')
