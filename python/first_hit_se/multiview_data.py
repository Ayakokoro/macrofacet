"""Independent multiview training/validation rays, retaining a fixed diagnostic set."""
import copy
import csv
from pathlib import Path

import numpy as np
import torch

from renewal.collect import digest, resolve, write_json
from renewal.reference_sampler import require_same_sampler
from .camera_holdout import ray_key, select
from .data import collect, load_data, prepare, read_json


def select_data(spec, base, output):
    original = load_data(resolve(base['dataset']))
    anchor = output/'selection_training_anchors.pt'
    if not anchor.exists():
        torch.save({**original, 'profiles': [p for p in original['profiles'] if p['split'] == 'train']}, anchor)
    actual = load_data(anchor)['profiles']
    expected = [p for p in original['profiles'] if p['split'] == 'train']
    if len(actual) != len(expected):
        raise ValueError('selection anchors differ from the original training set')
    for a, b in zip(actual, expected):
        if a['source'] != b['source']:
            raise ValueError('selection anchor changed')
        torch.testing.assert_close(a['features'], b['features'], atol=0, rtol=0)
    settings = {**spec['selection'], 'seed': spec['seed'], 'device': spec['device']}
    selected = select(settings, {**base, 'dataset': str(anchor)}, output/'selection')
    split_by_camera = {c['id']: c['split'] for c in settings['cameras']}
    request = read_json(output/'selection'/'collection_config.json')
    request['output_directory'] = str(output/'new_data')
    # Separate stream family from all earlier collections; audit actual trajectory seeds too.
    request['seed'] = spec['seed']+20000000
    for source in request['sources']:
        source['split'] = split_by_camera[Path(source['render_config']).stem]
    write_json(output/'collection_config.json', request)
    return {'new_rays': len(selected['selected']), 'candidate_count': selected['candidate_count']}


def collect_data(spec, base, output):
    collect(output/'collection_config.json')
    fixed_path = resolve(spec['fixed_dataset'])
    new_path = output/'new_data'/'dataset.pt'
    fixed, new = load_data(fixed_path), load_data(new_path)
    require_same_sampler(fixed['reference_sampler'], new['reference_sampler'], 'multiview augmentation')
    fixed_hash = digest(fixed_path)
    keys, seeds, cameras = set(), set(), {'train': set(), 'validation': set(), 'test': set()}
    for source, data in (('fixed', fixed), ('new', new)):
        for p in data['profiles']:
            ray = read_json(Path(p['source']).parent/'ray.json')
            key = ray_key(ray)
            if key in keys:
                raise ValueError('physical ray repeated across fixed and new data')
            keys.add(key)
            render = read_json(ray['source_config'])
            if render['field']['grid_file'] != read_json(resolve(base['render_config']))['field']['grid_file']:
                raise ValueError('wrong field')
            camera_key = tuple(render['render']['camera_position']+render['render']['camera_target'])
            if source == 'new':
                if p['split'] == 'test' or len(p['hit']) != spec['selection']['realizations_per_profile']:
                    raise ValueError('new labels have incorrect split or sample count')
                cameras[p['split']].add(camera_key)
            elif p['split'] == 'test':
                cameras['test'].add(camera_key)
            for name, other in (('origin', 'entry_origin'), ('direction', 'direction')):
                if not np.allclose(p['geometry'][name], ray[other], atol=1e-10, rtol=0):
                    raise ValueError('reference geometry differs from camera')
            with (Path(p['source'])/'first_passage_samples.csv').open(encoding='utf-8', newline='') as stream:
                for row in csv.DictReader(stream):
                    seed = int(row['seed'])
                    if seed in seeds:
                        raise ValueError('reference realization seed reused')
                    seeds.add(seed)
    if (cameras['train'] & cameras['validation'] or cameras['train'] & cameras['test']
            or cameras['validation'] & cameras['test']):
        raise ValueError('new training/validation cameras overlap each other or the fixed test cameras')
    manifest = copy.deepcopy(read_json(fixed_path.parent/'combined_manifest.json'))
    manifest['sources'] += read_json(new_path.parent/'manifest.json')['sources']
    manifest['parent_datasets'] = [{'path': str(p), 'sha256': digest(p)} for p in (fixed_path, new_path)]
    manifest['collection_config_sha256'] = digest(output/'collection_config.json')
    write_json(output/'manifest.json', manifest)
    summary = prepare(output/'manifest.json', output/'dataset.pt')
    merged = load_data(output/'dataset.pt')
    for old, actual in zip(fixed['profiles'], merged['profiles']):
        for name in ('features', 'distance', 'initial_sign', 'hit', 'speed'):
            torch.testing.assert_close(old[name], actual[name], atol=0, rtol=0, equal_nan=True)
        if old['split'] != actual['split'] or old['geometry'] != actual['geometry']:
            raise ValueError('fixed profiles changed')
    if digest(fixed_path) != fixed_hash:
        raise ValueError('fixed dataset changed')
    result = {'fixed_dataset_sha256': fixed_hash, 'dataset_sha256': summary['sha256'],
              'fixed_profiles_exactly_preserved': True, 'new_cameras_disjoint': True,
              'distinct_physical_rays': len(keys), 'unique_reference_seeds': len(seeds),
              'camera_counts': {s: len(c) for s, c in cameras.items()}, 'dataset': summary}
    write_json(output/'data_audit.json', result)
    return result


def run(config_path, phase='all'):
    spec = read_json(resolve(config_path))
    base = read_json(resolve(spec['original_study']))
    torch.set_num_threads(base['cpu_threads'])
    output = resolve(spec['output_directory'])
    output.mkdir(parents=True, exist_ok=True)
    if phase in ('all', 'select'):
        result = select_data(spec, base, output)
        if phase == 'select':
            return result
    if phase in ('all', 'collect'):
        return collect_data(spec, base, output)
    raise ValueError(phase)
