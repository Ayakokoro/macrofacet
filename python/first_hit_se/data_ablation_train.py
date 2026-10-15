"""Fixed-update, family-stratified head-only training for data ablations."""
import math
import hashlib
from pathlib import Path
import time

import torch

from renewal.collect import digest, resolve, write_json
from renewal.model import mixture_log_prob
from .data import load_data, read_json
from .data_ablation import immutable_json, pixel_of
from .gradient_head import (backbone_digest, cache_profiles, to_device, initialize_head,
                            parameters_at, evaluate_head, verify_frozen_predictions, _bundle)
from .train import load_model


def family_indices(cached):
    groups = {'camera': [], 'affine': []}
    for p in cached['profiles']:
        key = 'camera' if p['pixel'] is not None else 'affine'
        groups[key].append(torch.arange(p['begin'], p['end'], dtype=torch.long))
    return {k: torch.cat(v) if v else torch.empty(0, dtype=torch.long) for k,v in groups.items()}


def sample_indices(pools, camera_count, batch_size, seed, update):
    """Separate RNG streams keep affine labels paired across data variants."""
    def generator(family):
        # Hash the tuple, rather than add its entries: adjacent seeds must not
        # share the same batch sequence shifted by one optimizer update.
        key = f'first_hit_se_data_ablation:{seed}:{update}:{family}'.encode('ascii')
        value = int.from_bytes(hashlib.blake2b(key,digest_size=8).digest(),'little')
        return torch.Generator().manual_seed(value)
    camera_rng, affine_rng = generator('camera'), generator('affine')
    camera = torch.randint(len(pools['camera']), (camera_count,), generator=camera_rng)
    affine = torch.randint(len(pools['affine']), (batch_size-camera_count,), generator=affine_rng)
    return torch.cat((pools['camera'][camera], pools['affine'][affine]))


def train_fixed(source, train, validation, settings, identity, directory, seed, camera_count):
    directory.mkdir(parents=True, exist_ok=True)
    config = {**settings, 'seed': seed, 'camera_per_batch': camera_count,
              'affine_per_batch': settings['batch_size']-camera_count,
              'sampler': 'with_replacement_per_family_blake2b_seed_update_stream_v2'}
    immutable_json(directory/'training_config.json', config)
    model = initialize_head(source, 8, seed, split_jitter=0)
    model.eval()
    expected = backbone_digest(source)
    pools = family_indices(train)
    if not len(pools['camera']) or not len(pools['affine']):
        raise ValueError('both fixed training families need hit labels')
    optimizer = torch.optim.AdamW(model.mixture_head.parameters(), lr=settings['learning_rate'],
                                  weight_decay=settings['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=settings['updates'], eta_min=settings['minimum_learning_rate'])
    history, start, best = [], 1, math.inf
    if (directory/'last.pt').exists():
        saved = torch.load(directory/'last.pt', map_location=next(model.parameters()).device, weights_only=True)
        if saved['training_config'] != config or saved['cache_identity'] != identity:
            raise ValueError('fixed-update resume config or data changed')
        model.load_state_dict(saved['model_state'])
        optimizer.load_state_dict(saved['optimizer_state'])
        scheduler.load_state_dict(saved['scheduler_state'])
        history, start, best = saved['history'], saved['optimizer_updates']+1, saved['best']
    running = train['speed'].new_zeros(())
    interval = 0
    began = time.perf_counter()
    for update in range(start, settings['updates']+1):
        index = sample_indices(pools, camera_count, settings['batch_size'], seed, update).to(train['speed'].device)
        model.mixture_head.train()
        optimizer.zero_grad(set_to_none=True)
        mixture = parameters_at(model, train['inputs'][index], train['derivative'][index])
        loss = -mixture_log_prob(*mixture, train['speed'][index]).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError('nonfinite data-ablation loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.mixture_head.parameters(), settings['gradient_clip'],
                                       error_if_nonfinite=True)
        optimizer.step()
        rate = optimizer.param_groups[0]['lr']
        scheduler.step()
        running += loss.detach()
        interval += 1
        if update == 1 or update % settings['validation_every'] == 0 or update == settings['updates']:
            model.eval()
            validation_metrics = evaluate_head(model, validation, detailed=False)
            score = validation_metrics['speed_nll_on_hits']
            improved = score < best
            if improved:
                best = score
            if backbone_digest(model) != expected:
                raise AssertionError('frozen backbone changed')
            history.append({'update': update, 'training_speed_nll': float(running/interval),
                            'validation': validation_metrics, 'learning_rate': rate,
                            'examples_seen': update*settings['batch_size'],
                            'seconds': time.perf_counter()-began})
            bundle = {**_bundle(model, config, identity, update, validation_metrics),
                      'training_mode': 'gradient_head_only_fixed_updates', 'optimizer_updates': update}
            if improved:
                torch.save(bundle, directory/'best.pt')
            torch.save({**bundle, 'optimizer_state': optimizer.state_dict(),
                        'scheduler_state': scheduler.state_dict(), 'history': history,
                        'best': best}, directory/'last.pt')
            write_json(directory/'history.json', history)
            if update == 1 or update % (4*settings['validation_every']) == 0 or update == settings['updates']:
                print(f'{directory.name} step={update}: train={history[-1]["training_speed_nll"]:.6f} '
                      f'val={score:.6f}', flush=True)
            running.zero_()
            interval = 0
            began = time.perf_counter()
    report = {'updates': settings['updates'], 'batch_size': settings['batch_size'],
              'examples_seen': settings['updates']*settings['batch_size'],
              'camera_per_batch': camera_count, 'affine_per_batch': settings['batch_size']-camera_count,
              'training_seconds': sum(h['seconds'] for h in history),
              'backbone_sha256': backbone_digest(model), 'backbone_unchanged': backbone_digest(model)==expected,
              'training_camera_hits': len(pools['camera']), 'training_affine_hits': len(pools['affine'])}
    write_json(directory/'run_report.json', report)
    return model, report


def prepare_contexts(spec, root, source, base):
    datasets = read_json(root/'datasets.json')
    contexts = {'baseline': base['splits']['train']}
    identities = {'baseline': {'dataset_sha256': base['identity']['dataset_sha256'],
                               'checkpoint_sha256': digest(resolve(spec['initialize_from']))}}
    for variant in ('coverage', 'samples'):
        record = datasets[variant]
        identity = {'dataset_sha256': digest(Path(record['dataset'])),
                    'checkpoint_sha256': digest(resolve(spec['initialize_from']))}
        if identity['dataset_sha256'] != record['sha256']:
            raise ValueError('expanded dataset changed')
        identities[variant] = identity
        path = root/f'contexts_{variant}.pt'
        if path.exists():
            saved = torch.load(path, map_location='cpu', weights_only=True)
            if saved['identity'] != identity:
                raise ValueError('cached expanded training contexts changed')
        else:
            profiles = [p for p in load_data(record['dataset'])['profiles'] if p['split'] == 'train']
            print(f'CACHE {variant}: {len(profiles)} training profiles', flush=True)
            mapping = {pixel_of(p): 'train' for p in profiles if pixel_of(p)}
            cached = cache_profiles(source, profiles, mapping)
            saved = {'identity': identity, 'train': cached}
            torch.save(saved, path)
        contexts[variant] = saved['train']
        print(f'CACHED {variant}: {len(contexts[variant]["speed"])} hits', flush=True)
    return contexts, identities


def train_experiment(config_path):
    spec = read_json(config_path)
    root = resolve(spec['output_directory'])
    experiment = read_json(root/'experiment.json')
    if digest(config_path) != experiment['specification_sha256']:
        raise ValueError('data experiment specification changed')
    torch.set_num_threads(spec.get('cpu_threads', 1))
    source, _ = load_model(resolve(spec['initialize_from']), spec['device'])
    source.requires_grad_(False)
    base = torch.load(resolve(spec['base_contexts']), map_location='cpu', weights_only=True)
    if (base['identity']['checkpoint_sha256'] != digest(resolve(spec['initialize_from'])) or
            base['identity']['dataset_sha256'] != digest(resolve(spec['dataset'])) or
            base['identity']['scene_dataset_sha256'] != digest(resolve(spec['scene_dataset']))):
        raise ValueError('baseline cache does not match the common model/data')
    contexts, identities = prepare_contexts(spec, root, source, base)
    original_pools = family_indices(contexts['baseline'])
    camera_count = round(spec['training']['batch_size']*len(original_pools['camera'])/len(contexts['baseline']['speed']))
    provenance = {'source_checkpoint_sha256': digest(resolve(spec['initialize_from'])),
                  'sampler': 'with_replacement_per_family_blake2b_seed_update_stream_v2',
                  'base_contexts_sha256': digest(resolve(spec['base_contexts'])),
                  'backbone_sha256': backbone_digest(source), 'camera_per_batch': camera_count,
                  'affine_per_batch': spec['training']['batch_size']-camera_count,
                  'original_camera_hit_fraction': len(original_pools['camera'])/len(contexts['baseline']['speed']),
                  'data_identities': identities,
                  'runs': [{'name': f'{v}_s{s}', 'variant': v, 'seed': s}
                           for s in spec['seeds'] for v in ('baseline', 'coverage', 'samples')]}
    immutable_json(root/'training_experiment.json', provenance)
    common = {k: to_device(v, spec['device']) for k,v in base['splits'].items()}
    contexts = {k: to_device(v, spec['device']) for k,v in contexts.items()}
    evaluations = root/'evaluations'
    evaluations.mkdir(exist_ok=True)
    if not (evaluations/'source.json').exists():
        write_json(evaluations/'source.json', {'name': 'source',
            'splits': {k: evaluate_head(source, v) for k,v in common.items()}})
    scene_profiles = load_data(resolve(spec['scene_dataset']))['profiles']
    for i, r in enumerate(provenance['runs'], 1):
        directory = resolve(spec['model_directory'])/r['name']
        output = evaluations/f'{r["name"]}.json'
        if output.exists():
            old = read_json(output)
            if old['checkpoint_sha256'] != digest(directory/'last.pt'):
                raise ValueError('completed evaluation changed')
            print(f'SKIP {i}/{len(provenance["runs"])} {r["name"]}', flush=True)
            continue
        print(f'START {i}/{len(provenance["runs"])} {r["name"]}', flush=True)
        model, training = train_fixed(source, contexts[r['variant']], common['validation'], spec['training'],
                                      identities[r['variant']], directory, r['seed'], camera_count)
        primary = {k: evaluate_head(model, v) for k,v in common.items()}
        own = evaluate_head(model, contexts[r['variant']], detailed=False)
        fixed = verify_frozen_predictions(source, model, scene_profiles)
        best, selected = load_model(directory/'best.pt', spec['device'])
        # Detailed best-checkpoint metrics also isolate the three scene rays
        # that were originally held out; the other five remain explicitly marked.
        secondary = {k: evaluate_head(best, v) for k,v in common.items()}
        write_json(output, {**r, 'training': training,
                           'checkpoint': str(directory/'last.pt'), 'checkpoint_sha256': digest(directory/'last.pt'),
                           'splits': primary, 'own_training_distribution': own,
                           'evaluation_note': 'All train/validation/test rows use the same ORIGINAL reference profiles and labels.',
                           'frozen_prediction_max_abs_difference': fixed,
                           'validation_selected': {'update': selected['optimizer_updates'], 'splits': secondary}})
        print(f'DONE {i}/{len(provenance["runs"])} {r["name"]}: '
              f'test NLL={primary["test"]["speed_nll_on_hits"]:.6f}, '
              f'PIT={primary["test"]["speed_pit_ks"]:.6f}', flush=True)
        del model, best
    return {'completed_runs': len(provenance['runs']), 'output_directory': str(root)}
