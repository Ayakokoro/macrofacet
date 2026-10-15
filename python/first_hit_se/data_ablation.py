"""Paired tests of additional camera-ray histories versus GP draws per ray."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import csv
import hashlib
import math
from pathlib import Path
import random
import shutil
import subprocess
import time

import torch

from renewal.collect import digest, resolve, write_json
from . import FORMAT, VERSION, EVENT, SE_KERNEL
from .data import load_data, prepare, read_json


def immutable_json(path, value):
    if path.exists() and read_json(path) != value:
        raise ValueError(f'existing experiment definition changed: {path}')
    write_json(path, value)


def profile_key(profile):
    geometry = {k: v for k, v in profile['geometry'].items() if k not in ('kernel_id', 'state_id')}
    return hashlib.sha256(str((profile['features'].tolist(), sorted(geometry.items()))).encode()).hexdigest()


def pixel_of(profile):
    path = Path(profile['source']).parent/'ray.json'
    return tuple(read_json(path)['pixel']) if path.exists() else None


def collect(spec_path):
    spec = read_json(spec_path)
    root = resolve(spec['output_directory'])
    root.mkdir(parents=True, exist_ok=True)
    manifest = read_json(resolve(spec['manifest']))
    dataset = load_data(resolve(spec['dataset']))
    original = [s for s in manifest['sources'] if s['split'] == 'train' and s.get('camera_metadata')]
    all_pixels = {tuple(read_json(s['camera_metadata'])['pixel'])
                  for s in manifest['sources'] if s.get('camera_metadata')}
    scene = read_json(resolve(spec['render_config']))
    identity = {'specification': spec, 'specification_sha256': digest(spec_path),
                'dataset_sha256': digest(resolve(spec['dataset'])),
                'source_checkpoint_sha256': digest(resolve(spec['initialize_from'])),
                'render_config_sha256': digest(resolve(spec['render_config'])),
                'vdb_sha256': digest(resolve(scene['field']['grid_file'])),
                'executable_sha256': digest(resolve(spec['executable'])),
                'inspector_sha256': digest(resolve(spec['inspector'])),
                'original_camera_train_rays': len(original),
                'excluded_pixels': [list(p) for p in sorted(all_pixels)]}
    immutable_json(root/'experiment.json', identity)
    selection_path = root/'ray_selection.json'
    began = time.perf_counter()
    inspector = str(resolve(spec['inspector']))
    candidates = list(range(scene['render']['width']*scene['render']['height']))
    random.Random(spec['selection_seed']).shuffle(candidates)
    candidates = [(i % scene['render']['width'], i//scene['render']['width']) for i in candidates]
    candidates = [p for p in candidates if p not in all_pixels]

    def inspect(item):
        index, pixel = item
        folder = root/'collection'/'coverage'/f'candidate_{index:05d}'
        folder.mkdir(parents=True, exist_ok=True)
        ray = folder/'ray.json'
        if ray.exists():
            if tuple(read_json(ray)['pixel']) != pixel:
                raise ValueError('cached camera pixel changed')
            return {'pixel': list(pixel), 'directory': str(folder)}
        completed = subprocess.run([inspector, str(resolve(spec['render_config'])), str(folder),
                                    str(pixel[0]), str(pixel[1])], capture_output=True, text=True)
        if completed.returncode:
            if completed.stderr.strip() == 'ray misses medium':
                return None
            raise RuntimeError(f'camera extraction failed: {completed.stderr}')
        return {'pixel': list(pixel), 'directory': str(folder)}

    if selection_path.exists():
        selection = read_json(selection_path)
    else:
        selected, rejected, offset = [], [], 0
        with ThreadPoolExecutor(max_workers=spec['collection_workers']) as executor:
            while len(selected) < spec['new_camera_rays']:
                # Geometry selection uses no GP labels or model errors.
                count = min(32, spec['new_camera_rays']-len(selected))
                items = list(enumerate(candidates[offset:offset+count], offset))
                if not items:
                    raise ValueError('not enough distinct camera rays')
                for (_, pixel), result in zip(items, executor.map(inspect, items)):
                    if result is None:
                        rejected.append(list(pixel))
                    else:
                        selected.append(result)
                offset += count
                print(f'GEOMETRY {len(selected)}/{spec["new_camera_rays"]}; domain misses={len(rejected)}', flush=True)
        selection = {'method': 'Seeded uniform pixel permutation; exclude all original split pixels; accept active-domain intersections.',
                     'selected': selected, 'domain_misses': rejected}
        immutable_json(selection_path, selection)
    if len(selection['selected']) != spec['new_camera_rays']:
        raise ValueError('incorrect count of additional ray histories')
    jobs = []
    for index, item in enumerate(selection['selected']):
        folder = Path(item['directory'])
        jobs.append(('coverage', index, folder, read_json(folder/'reference.json'),
                     spec['new_ray_realizations'], spec['coverage_seed']+104729*index))
    for index, source in enumerate(original):
        folder = root/'collection'/'samples'/f'source_{index:05d}'
        folder.mkdir(parents=True, exist_ok=True)
        old = Path(source['directory']).parent
        immutable_json(folder/'ray.json', read_json(old/'ray.json'))
        request = read_json(old/'collection_reference.json')
        jobs.append(('samples', index, folder, request,
                     spec['extra_realizations_per_old_ray'], spec['samples_seed']+104729*index))

    def generate(job):
        group, index, folder, request, realizations, seed = job
        config = deepcopy(request)
        fp = config['first_passage']
        if fp['initial_condition']['type'] != EVENT:
            raise ValueError('expected unconditional first zero')
        horizon = fp['grid']['max_time']
        fp['grid']['step_sizes'] = [horizon/math.ceil(horizon/s) for s in spec['reference_steps']]
        fp['monte_carlo'] = {'trajectories': realizations, 'thread_count': spec['threads_per_reference'],
                             'write_raw_samples': True}
        config['seed'] = seed
        config['output_directory'] = str(folder/'reference')
        completed_path = folder/'completed.json'
        fingerprint = {'request': config, 'executable': identity['executable_sha256']}
        if completed_path.exists():
            if read_json(completed_path) != fingerprint:
                raise ValueError('reference generation settings changed')
        else:
            write_json(folder/'collection_reference.json', config)
            with (folder/'generation.log').open('w', encoding='utf-8') as stream:
                subprocess.run([str(resolve(spec['executable'])), 'first-passage', '--config',
                                str(folder/'collection_reference.json')], stdout=stream,
                               stderr=subprocess.STDOUT, check=True)
            write_json(completed_path, fingerprint)
        return group, index, {'scene_id': 'shader_ball_sigma005', 'split': 'train',
                             'directory': str(folder/'reference'), 'camera_metadata': str(folder/'ray.json')}

    generated = {'coverage': {}, 'samples': {}}
    with ThreadPoolExecutor(max_workers=spec['collection_workers']) as executor:
        futures = [executor.submit(generate, job) for job in jobs]
        for completed, future in enumerate(as_completed(futures), 1):
            group, index, source = future.result()
            generated[group][index] = source
            progress = {'completed': completed, 'total': len(jobs),
                        'coverage': len(generated['coverage']), 'samples': len(generated['samples']),
                        'elapsed_seconds': time.perf_counter()-began}
            write_json(root/'collection_progress.json', progress)
            if completed % 8 == 0 or completed == len(jobs):
                print(f'REFERENCE {completed}/{len(jobs)}: coverage={progress["coverage"]} '
                      f'samples={progress["samples"]} seconds={progress["elapsed_seconds"]:.1f}', flush=True)
    additions = {}
    for group, sources in generated.items():
        folder = root/'datasets'
        folder.mkdir(exist_ok=True)
        path = folder/f'{group}_additions_manifest.json'
        merged = {'format': FORMAT, 'version': VERSION, 'event': EVENT, 'kernel': SE_KERNEL,
                  'split_unit': 'ray', 'sources': [sources[i] for i in sorted(sources)]}
        immutable_json(path, merged)
        output = folder/f'{group}_additions.pt'
        if not output.exists():
            prepare(path, output)
        additions[group] = load_data(output)
    return merge_datasets(spec, root, dataset, additions)


def merge_profiles(base, added, mode):
    """Retain original labels and untouched non-training splits."""
    result = deepcopy(base)
    heldout = {profile_key(p) for p in base['profiles'] if p['split'] != 'train'}
    if mode == 'coverage':
        known = {profile_key(p) for p in base['profiles']}
        for index, original in enumerate(added['profiles']):
            p = deepcopy(original)
            key = profile_key(p)
            if p['split'] != 'train' or key in known or key in heldout:
                raise ValueError('new coverage profiles overlap an existing split')
            known.add(key)
            p['id'] = f'coverage_new:{index}:{p["id"]}'
            result['profiles'].append(p)
    elif mode == 'samples':
        by_pixel = {pixel_of(p): p for p in result['profiles'] if p['split'] == 'train' and pixel_of(p)}
        seen = set()
        for p in added['profiles']:
            pixel = pixel_of(p)
            if pixel not in by_pixel or pixel in seen or p['split'] != 'train':
                raise ValueError('additional samples must match unique original training rays')
            seen.add(pixel)
            target = by_pixel[pixel]
            if profile_key(target) != profile_key(p):
                raise ValueError(f'GP resampling changed the physical ray: {pixel}')
            for key in ('distance', 'hit', 'initial_sign', 'speed'):
                target[key] = torch.cat((target[key], p[key]))
            target['additional_sources'] = [{'source': p['source'], 'samples_sha256': p['samples_sha256'],
                                             'resolved_sha256': p['resolved_sha256']}]
    else:
        raise ValueError('unknown data expansion mode')
    return result


def merge_datasets(spec, root, base, additions):
    records = {}
    original_holdout = [p for p in base['profiles'] if p['split'] != 'train']
    for group, added in additions.items():
        merged = merge_profiles(base, added, group)
        path = root/'datasets'/f'{group}.pt'
        if not path.exists():
            torch.save(merged, path)
        actual = load_data(path)
        after = [p for p in actual['profiles'] if p['split'] != 'train']
        for a,b in zip(original_holdout, after):
            if profile_key(a) != profile_key(b):
                raise AssertionError('held-out geometry changed')
            for key in ('features', 'distance', 'hit', 'initial_sign', 'speed'):
                torch.testing.assert_close(a[key], b[key], atol=0, rtol=0, equal_nan=True)
        train = [p for p in actual['profiles'] if p['split'] == 'train']
        camera = [p for p in train if pixel_of(p)]
        affine = [p for p in train if not pixel_of(p)]
        records[group] = {'dataset': str(path), 'sha256': digest(path), 'train_profiles': len(train),
                          'camera_rays': len(camera), 'camera_realizations': sorted({len(p['hit']) for p in camera}),
                          'camera_samples': sum(len(p['hit']) for p in camera),
                          'camera_hits': sum(int(p['hit'].sum()) for p in camera),
                          'affine_profiles': len(affine), 'affine_samples': sum(len(p['hit']) for p in affine),
                          'heldout_profiles': len(after), 'heldout_unchanged': True}
    if records['coverage']['camera_rays'] != 480 or records['samples']['camera_rays'] != 120:
        raise AssertionError('experiment camera-ray counts differ from the design')
    if records['coverage']['camera_samples'] != records['samples']['camera_samples']:
        raise AssertionError('additional data variants have unequal camera realization budgets')
    write_json(root/'datasets.json', records)
    print('DATASETS READY', records, flush=True)
    return records


def run(config_path, phase='all'):
    if phase in ('collect', 'all'):
        result = collect(config_path)
    if phase in ('train', 'all'):
        from .data_ablation_train import train_experiment
        result = train_experiment(config_path)
    if phase in ('report', 'all'):
        from .data_ablation_report import report
        result = report(config_path)
    return result
