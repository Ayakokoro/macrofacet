"""Fit only the conditional first-hit speed head on a frozen RNO backbone.

The cache contains inputs at reference hit locations, not model-generated hits.
Censored samples carry no gradient-head label. No validation/test labels enter
training, and all runs use the same fixed backbone and cached training examples.
"""
from copy import deepcopy
import hashlib
import math
from pathlib import Path
import statistics
import time

import torch
from torch.nn import functional as F

from renewal.collect import digest, resolve, write_json
from renewal.model import mixture_cdf, mixture_log_prob
from . import FORMAT, VERSION, EVENT, SE_KERNEL
from .batch import query_many
from .data import load_data, read_json
from .model import FirstHitNetwork, ModelConfig
from .train import load_model


def backbone_digest(model):
    result = hashlib.sha256()
    for name, value in model.state_dict().items():
        if not name.startswith('mixture_head.'):
            result.update(name.encode())
            result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def initialize_head(source, components, seed=0, split_jitter=0.002):
    """Duplicate mixture components, then slightly separate their locations.

    Without jitter, splitting preserves the density up to floating-point
    roundoff. Paired, zero-mean location offsets break duplicate symmetry;
    initialization diagnostics report their actual density/CDF changes.
    """
    if source.config.architecture != 'rno':
        raise ValueError('head-only fitting requires an RNO')
    old = source.config.components
    if components < old or components % old:
        raise ValueError('component count must be a positive multiple of the source count')
    if not math.isfinite(split_jitter) or split_jitter < 0:
        raise ValueError('invalid split jitter')
    config = {**source.configuration(), 'components': components}
    model = FirstHitNetwork(ModelConfig(**config)).to(next(source.parameters()))
    state = model.state_dict()
    repeat = components//old
    for name, value in source.state_dict().items():
        if name in ('mixture_head.4.weight', 'mixture_head.4.bias'):
            state[name].copy_(torch.cat([part.repeat_interleave(repeat, 0)
                                        for part in value.chunk(3, 0)]))
        else:
            state[name].copy_(value)
    state['mixture_head.4.bias'][:components] -= math.log(repeat)
    if repeat > 1 and split_jitter:
        generator = torch.Generator().manual_seed(seed)
        # Each parent's offsets sum to zero. Random radii avoid identical
        # perturbations at every original component while retaining symmetry.
        offsets = torch.linspace(-1., 1., repeat)[None].repeat(old, 1)
        offsets *= (0.5+0.5*torch.rand((old, 1), generator=generator))*split_jitter
        state['mixture_head.4.bias'][components:2*components] += offsets.flatten().to(
            state['mixture_head.4.bias'])
    model.load_state_dict(state)
    model.requires_grad_(False)
    model.mixture_head.requires_grad_(True)
    if backbone_digest(model) != backbone_digest(source):
        raise AssertionError('initialization changed the frozen backbone')
    return model


def parameters_at(model, inputs, derivative):
    logits, delta, raw_scale = model.mixture_head(inputs).chunk(3, -1)
    return (F.log_softmax(logits, -1), -derivative[..., None]+delta,
            F.softplus(raw_scale)+model.config.sigma_floor)


def _pixel(profile):
    source = profile.get('source')
    ray = Path(source).parent/'ray.json' if source else None
    return read_json(ray)['pixel'] if ray and ray.exists() else None


@torch.no_grad()
def cache_profiles(model, profiles, original_splits=None, batch_size=8):
    fields = {k: [] for k in ('inputs', 'derivative', 'speed', 'distance', 'fraction', 'sign', 'profile')}
    records, offset, maximum = [], 0, 0.
    original_splits = original_splits or {}
    for begin in range(0, len(profiles), batch_size):
        batch = profiles[begin:begin+batch_size]
        # query_many requires a nonempty query vector, even for a zero-hit ray.
        queries = [p['distance'][p['hit']] if p['hit'].any() else p['distance'].new_zeros(1)
                   for p in batch]
        results = query_many(model, [p['features'] for p in batch], queries,
                             with_mixture_context=True)
        for p, r in zip(batch, results):
            n = int(p['hit'].sum())
            pixel = _pixel(p)
            records.append({'id': p.get('id', str(len(records))), 'split': p['split'],
                            'original_split': original_splits.get(tuple(pixel)) if pixel else p['split'],
                            'pixel': pixel, 'horizon': float(p['horizon']), 'samples': len(p['hit']),
                            'hits': n, 'begin': offset, 'end': offset+n})
            if not n:
                continue
            sign = p['initial_sign'][p['hit']]
            device = r['mixture_inputs'].device
            rows = torch.arange(n, device=device)
            branch = (sign < 0).long().to(device)
            x = r['mixture_inputs'][branch, rows]
            db = r['mixture_derivative'][branch, rows]
            actual = parameters_at(model, x, db)
            for key, value in zip(('log_weights', 'means', 'scales'), actual):
                maximum = max(maximum, float((value-r[key][branch, rows]).abs().max()))
            fields['inputs'].append(x.cpu())
            fields['derivative'].append(db.cpu())
            for key, value in (('speed', p['speed'][p['hit']]), ('distance', p['distance'][p['hit']]),
                               ('fraction', p['distance'][p['hit']]/p['horizon']), ('sign', sign),
                               ('profile', torch.full((n,), len(records)-1, dtype=torch.long))):
                fields[key].append(value.cpu())
            offset += n
    if not offset:
        raise ValueError('head-only fitting requires at least one hit in each split')
    if maximum > 2e-5:
        raise AssertionError(f'cached head differs from direct query: {maximum}')
    return {**{k: torch.cat(v) for k, v in fields.items()}, 'profiles': records,
            'cache_max_parameter_difference': maximum}


def prepare_cache(spec, root, source):
    identity = {'format': 'macrofacet.first_hit_se.gradient_cache', 'version': 1,
                'checkpoint_sha256': digest(resolve(spec['initialize_from'])),
                'dataset_sha256': digest(resolve(spec['dataset'])),
                'scene_dataset_sha256': digest(resolve(spec['scene_dataset'])),
                'backbone_sha256': backbone_digest(source)}
    path = root/'contexts.pt'
    if path.exists():
        cache = torch.load(path, map_location='cpu', weights_only=True)
        if cache['identity'] != identity:
            raise ValueError('cached contexts refer to a different checkpoint or dataset')
        return cache
    data = load_data(resolve(spec['dataset']))['profiles']
    original = {tuple(pixel): p['split'] for p in data if (pixel := _pixel(p))}
    groups = {s: [p for p in data if p['split'] == s] for s in ('train', 'validation', 'test')}
    groups['scene'] = load_data(resolve(spec['scene_dataset']))['profiles']
    cache = {'identity': identity, 'splits': {}}
    for key, profiles in groups.items():
        print(f'CACHE {key}: {len(profiles)} profiles', flush=True)
        cache['splits'][key] = cache_profiles(source, profiles, original)
        print(f'CACHED {key}: {len(cache["splits"][key]["speed"])} hits', flush=True)
    torch.save(cache, path)
    write_json(root/'cache_manifest.json', {**identity, 'splits': {
        k: {'profiles': len(v['profiles']), 'hits': len(v['speed']),
            'maximum_parameter_difference': v['cache_max_parameter_difference']}
        for k, v in cache['splits'].items()}})
    return cache


def to_device(cached, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in cached.items()}


def _ks(pit):
    if not len(pit):
        return None
    ordered = pit.double().sort().values
    n = len(ordered)
    return max(float((torch.arange(1, n+1, device=pit.device)/n-ordered).max()),
               float((ordered-torch.arange(n, device=pit.device)/n).max()))


@torch.no_grad()
def evaluate_head(model, cached, detailed=True, batch_size=4096):
    nlls, pits, probabilities = [], [], []
    thresholds = cached['speed'].new_tensor([0.1, 0.25, 0.5, 2., 3., 4.])
    for begin in range(0, len(cached['speed']), batch_size):
        sl = slice(begin, begin+batch_size)
        mix = parameters_at(model, cached['inputs'][sl], cached['derivative'][sl])
        speed = cached['speed'][sl]
        nlls.append(-mixture_log_prob(*mix, speed).cpu())
        pits.append(mixture_cdf(*mix, speed).cpu())
        if detailed:
            probabilities.append(torch.stack([mixture_cdf(*mix, x.expand_as(speed))
                                              for x in thresholds], -1).cpu())
    nll, pit = torch.cat(nlls), torch.cat(pits)
    if not torch.isfinite(nll).all() or not torch.isfinite(pit).all():
        raise FloatingPointError('nonfinite gradient-head evaluation')
    result = {'hits': len(nll), 'speed_nll_on_hits': float(nll.mean()), 'speed_pit_ks': _ks(pit)}
    if not detailed:
        return result
    probs = torch.cat(probabilities)
    speed = cached['speed'].cpu()
    fraction, sign = cached['fraction'].cpu(), cached['sign'].cpu()
    grid = torch.linspace(0, 1, 101, dtype=torch.float64)
    def subset(mask):
        a, b = nll[mask], pit[mask]
        if not len(a):
            return {'hits': 0}
        ordered = b.sort().values
        levels = b.new_tensor([.01, .05, .1, .5, .9, .95, .99])
        return {'hits': len(a), 'speed_nll_on_hits': float(a.mean()), 'speed_pit_ks': _ks(b),
                'pit_ecdf': (torch.searchsorted(ordered, grid, right=True)/len(b)).tolist(),
                'quantile_calibration': [{'nominal': float(t), 'observed': float((b <= t).double().mean())}
                                         for t in levels],
                'threshold_calibration': [
                    {'W': float(t), 'observed_cdf': float((speed[mask] <= t).double().mean()),
                     'predicted_cdf': float(probs[mask, i].mean()),
                     'absolute_error': float(((speed[mask] <= t).double().mean()-probs[mask, i].mean()).abs())}
                    for i, t in enumerate(thresholds.cpu())]}
    all_mask = torch.ones(len(nll), dtype=torch.bool)
    result.update(subset(all_mask))
    result['pit_grid'] = grid.tolist()
    result['by_sign'] = {str(s): subset(sign == s) for s in (1, -1)}
    result['by_distance_fraction'] = {}
    for i in range(4):
        mask = (fraction >= i/4) & (fraction <= 1 if i == 3 else fraction < (i+1)/4)
        result['by_distance_fraction'][f'{i/4:g}-{(i+1)/4:g}'] = subset(mask)
    result['profiles'] = []
    for record in cached['profiles']:
        mask = torch.zeros(len(nll), dtype=torch.bool)
        mask[record['begin']:record['end']] = True
        row = {**record, **subset(mask)}
        row['by_distance_fraction'] = {}
        for i in range(4):
            sub = mask & (fraction >= i/4) & (fraction <= 1 if i == 3 else fraction < (i+1)/4)
            # Local groups retain metrics and counts; omit bulky repeated CDFs.
            detail = subset(sub)
            row['by_distance_fraction'][f'{i/4:g}-{(i+1)/4:g}'] = {
                k: detail[k] for k in ('hits', 'speed_nll_on_hits', 'speed_pit_ks') if k in detail}
        result['profiles'].append(row)
    groups = [p['speed_pit_ks'] for p in result['profiles'] if p['hits'] >= 64]
    result['mean_profile_pit_ks_min64'] = statistics.mean(groups) if groups else None
    result['by_original_split'] = {}
    for key in ('train', 'validation', 'test'):
        mask = torch.zeros(len(nll), dtype=torch.bool)
        for record in cached['profiles']:
            if record['original_split'] == key:
                mask[record['begin']:record['end']] = True
        result['by_original_split'][key] = subset(mask)
    return result


@torch.no_grad()
def initial_difference(source, model, cached, batch_size=4096):
    nll_delta, max_cdf = 0., 0.
    for begin in range(0, len(cached['speed']), batch_size):
        sl = slice(begin, begin+batch_size)
        x, db, w = (cached[k][sl] for k in ('inputs', 'derivative', 'speed'))
        a, b = parameters_at(source, x, db), parameters_at(model, x, db)
        nll_delta += float((mixture_log_prob(*a, w)-mixture_log_prob(*b, w)).sum())
        max_cdf = max(max_cdf, float((mixture_cdf(*a, w)-mixture_cdf(*b, w)).abs().max()))
    return {'training_nll_change': nll_delta/len(cached['speed']),
            'training_max_abs_cdf_change_at_observed_speed': max_cdf}


def _bundle(model, config, identity, epoch, validation):
    return {'format': FORMAT, 'version': VERSION, 'event': EVENT, 'kernel': SE_KERNEL,
            'model_config': model.configuration(),
            'model_state': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            'training_config': config, 'cache_identity': identity, 'epoch': epoch,
            'validation': validation, 'training_mode': 'gradient_head_only'}


def train_one(source, cache, spec, components, seed, directory):
    directory.mkdir(parents=True, exist_ok=True)
    config = {**spec['training'], 'components': components, 'seed': seed,
              'initialize_from': spec['initialize_from'], 'split_jitter': spec['split_jitter']}
    if (directory/'training_config.json').exists() and read_json(directory/'training_config.json') != config:
        raise ValueError('existing gradient-head run has a different configuration')
    write_json(directory/'training_config.json', config)
    model = initialize_head(source, components, seed, spec['split_jitter'])
    expected_digest = backbone_digest(source)
    train, validation = cache['splits']['train'], cache['splits']['validation']
    optimizer = torch.optim.AdamW(model.mixture_head.parameters(), lr=config['learning_rate'],
                                  weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['epochs'], eta_min=config['minimum_learning_rate'])
    history, best, start, updates = [], math.inf, 1, 0
    initial = initial_difference(source, model, train)
    if (directory/'last.pt').exists():
        saved = torch.load(directory/'last.pt', map_location=next(model.parameters()).device, weights_only=True)
        if saved['training_config'] != config or saved['cache_identity'] != cache['identity']:
            raise ValueError('resume configuration/cache mismatch')
        model.load_state_dict(saved['model_state'])
        optimizer.load_state_dict(saved['optimizer_state'])
        scheduler.load_state_dict(saved['scheduler_state'])
        history, best, start, updates = saved['history'], saved['best'], saved['epoch']+1, saved['optimizer_updates']
    count = len(train['speed'])
    batch_size = config['batch_size']
    for epoch in range(start, config['epochs']+1):
        began = time.perf_counter()
        generator = torch.Generator().manual_seed(seed+epoch)
        order = torch.randperm(count, generator=generator).to(train['speed'].device)
        total = train['speed'].new_zeros(())
        model.mixture_head.train()
        for begin in range(0, count, batch_size):
            indices = order[begin:begin+batch_size]
            optimizer.zero_grad(set_to_none=True)
            mixture = parameters_at(model, train['inputs'][indices], train['derivative'][indices])
            loss = -mixture_log_prob(*mixture, train['speed'][indices]).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite head-only training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.mixture_head.parameters(), config['gradient_clip'],
                                           error_if_nonfinite=True)
            optimizer.step()
            total += loss.detach()*len(indices)
            updates += 1
        rate = optimizer.param_groups[0]['lr']
        scheduler.step()
        model.eval()
        checked = epoch == 1 or epoch % config['validation_every'] == 0 or epoch == config['epochs']
        evaluated = evaluate_head(model, validation, detailed=False) if checked else None
        improved = evaluated is not None and evaluated['speed_nll_on_hits'] < best
        if improved:
            best = evaluated['speed_nll_on_hits']
        row = {'epoch': epoch, 'training_speed_nll': float(total/count),
               'validation': evaluated, 'learning_rate': rate, 'optimizer_updates': updates,
               'hits_this_epoch': count, 'seconds': time.perf_counter()-began}
        history.append(row)
        if checked:
            if backbone_digest(model) != expected_digest:
                raise AssertionError('a frozen backbone parameter changed')
            bundle = _bundle(model, config, cache['identity'], epoch, evaluated)
            if improved:
                torch.save(bundle, directory/'best.pt')
            torch.save({**bundle, 'optimizer_state': optimizer.state_dict(),
                        'scheduler_state': scheduler.state_dict(), 'history': history,
                        'best': best, 'optimizer_updates': updates}, directory/'last.pt')
            write_json(directory/'history.json', history)
            print(f'K={components} seed={seed} epoch={epoch}: train={row["training_speed_nll"]:.6f} '
                  f'val={evaluated["speed_nll_on_hits"]:.6f} seconds={row["seconds"]:.2f}', flush=True)
    result = {'components': components, 'seed': seed, 'initialization': initial,
              'epochs': history[-1]['epoch'], 'optimizer_updates': updates,
              'training_seconds': sum(h['seconds'] for h in history),
              'parameters': sum(p.numel() for p in model.parameters()),
              'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
              'frozen_backbone_sha256': backbone_digest(model),
              'backbone_unchanged': backbone_digest(model) == expected_digest,
              'primary_checkpoint': 'last.pt', 'best_checkpoint_criterion': 'validation speed NLL'}
    write_json(directory/'run_report.json', result)
    return model, result


@torch.no_grad()
def verify_frozen_predictions(source, model, profiles):
    maximum = {'T': 0., 'H': 0., 'log_hazard': 0., 'log_density': 0.}
    # Exclude the head from both forward calls. Its output size then cannot
    # influence floating-point operation shapes in the backbone.
    for begin in range(0, len(profiles), 8):
        batch = profiles[begin:begin+8]
        queries = [torch.linspace(0, p['horizon'], 129, dtype=torch.float64) for p in batch]
        a = query_many(source, [p['features'] for p in batch], queries, with_mixture=False)
        b = query_many(model, [p['features'] for p in batch], queries, with_mixture=False)
        for left, right in zip(a, b):
            for key in maximum:
                maximum[key] = max(maximum[key], float((left[key]-right[key]).abs().max()))
    if any(maximum.values()):
        raise AssertionError(f'head-only fitting changed the distance distribution: {maximum}')
    return maximum


def run(config_path):
    spec = read_json(config_path)
    root = resolve(spec['output_directory'])
    root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(spec.get('cpu_threads', 1))
    source, _ = load_model(resolve(spec['initialize_from']), spec['device'])
    source.requires_grad_(False)
    experiment = {'specification': spec, 'specification_sha256': digest(config_path),
                  'source_checkpoint_sha256': digest(resolve(spec['initialize_from'])),
                  'backbone_sha256': backbone_digest(source),
                  'protocol': 'Fixed backbone; full hit-label epochs; paired shuffles; last epoch primary.',
                  'evaluation_scope': 'Conditional speed at reference hit positions and entry signs.',
                  'runs': [{'name': f'k{k}_s{s}', 'components': k, 'seed': s}
                           for s in spec['seeds'] for k in spec['components']]}
    manifest = root/'experiment.json'
    if manifest.exists() and read_json(manifest) != experiment:
        raise ValueError('gradient-head experiment changed; use a new output directory')
    write_json(manifest, experiment)
    cache = prepare_cache(spec, root, source)
    cache['splits'] = {k: to_device(v, spec['device']) for k, v in cache['splits'].items()}
    evaluations = root/'evaluations'
    evaluations.mkdir(exist_ok=True)
    if not (evaluations/'source.json').exists():
        write_json(evaluations/'source.json', {'name': 'source', 'checkpoint_sha256': experiment['source_checkpoint_sha256'],
            'splits': {k: evaluate_head(source, v) for k, v in cache['splits'].items()}})
    scene = load_data(resolve(spec['scene_dataset']))['profiles']
    for i, record in enumerate(experiment['runs'], 1):
        name = record['name']
        directory = resolve(spec['model_directory'])/name
        output = evaluations/f'{name}.json'
        if output.exists():
            completed = read_json(output)
            if completed['checkpoint_sha256'] != digest(directory/'last.pt'):
                raise ValueError('completed head evaluation refers to another checkpoint')
            print(f'SKIP {i}/{len(experiment["runs"])} {name}', flush=True)
            continue
        print(f'START {i}/{len(experiment["runs"])} {name}', flush=True)
        model, training = train_one(source, cache, spec, record['components'], record['seed'], directory)
        fixed = verify_frozen_predictions(source, model, scene)
        primary = {k: evaluate_head(model, v) for k, v in cache['splits'].items()}
        best, selected = load_model(directory/'best.pt', spec['device'])
        secondary = {k: evaluate_head(best, v, detailed=False) for k, v in cache['splits'].items()}
        # Scene held-out statistics must exclude the four training and one
        # validation ray, even though these labels came from fresh GP samples.
        write_json(output, {**record, 'training': training, 'checkpoint': str(directory/'last.pt'),
                   'checkpoint_sha256': digest(directory/'last.pt'),
                   'frozen_prediction_max_abs_difference': fixed, 'splits': primary,
                   'validation_selected': {'epoch': selected['epoch'], 'splits': secondary}})
        del model, best
        print(f'DONE {i}/{len(experiment["runs"])} {name}: '
              f'test NLL={primary["test"]["speed_nll_on_hits"]:.6f} '
              f'PIT={primary["test"]["speed_pit_ks"]:.6f}', flush=True)
    from .gradient_head_report import report
    return report(config_path)
