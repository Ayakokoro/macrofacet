"""Paired distance-only comparison, numerical audits, and a conditional gradient probe."""
import copy
import html
import math
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from renewal.collect import digest, resolve, write_json
from renewal.model import mean_at
from renewal.profiles import subdivide_features
from .camera_study import pixel_of, plotting, save_figure
from .data import collect, load_data, read_json
from .distance_models import (MODES, make_model, predictions, sample_distance,
                              transmittance)
from .distance_objective import distance_objective, objective_weights
from .hazard import local_bin_log_rates, local_log_hazard, piecewise_distribution
from .hazard_study import (METRICS, SPLITS, audit_data, make_batch,
                           parameter_hash, prepare)
from .spline import local_profile, resample_profile


def synchronize(device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()


def load_run(path, device='cpu'):
    bundle = torch.load(path, map_location=device, weights_only=True)
    if bundle['format'] != 'first_hit_distance_design':
        raise ValueError('not a distance-design checkpoint')
    model = make_model(bundle['config'], bundle['mode'], bundle['seed']).to(device)
    model.load_state_dict(bundle['model_state'])
    model.eval()
    return model, bundle


@torch.no_grad()
def evaluate(model, profiles, curves=False, batch_size=4):
    if model is not None:
        model.eval()
    totals = {k: 0. for k in ('training_nll', 'distance_nll', 'complete_distance_nll')}
    count, rows = 0, []
    for begin in range(0, len(profiles), batch_size):
        batch = make_batch(profiles[begin:begin+batch_size])
        p = predictions(model, batch)
        for key in totals:
            totals[key] += float(p[key].sum())
        count += batch['hit'].numel()
        q = torch.linspace(0, 1, batch['reference_T'].shape[-1], dtype=torch.float64,
                           device=batch['fixed'].device)[None]*batch['horizon'][:, None]
        T = transmittance(model, p['encoded'], q, batch['horizon'], p['log_prior'])
        if (not torch.isfinite(T).all() or not torch.isfinite(p['distance_nll']).all()
                or (T[:, 1:]-T[:, :-1] > 1e-12).any() or (T[:, 0]-1).abs().max() > 1e-10):
            raise FloatingPointError('invalid probability distribution')
        for j in range(len(T)):
            error = T[j]-batch['reference_T'][j]
            row = {'max_abs_T_error': float(error.abs().max()),
                   'mse_T': float(error.square().mean()), 'escape_error': float(error[-1]),
                   'distance_nll': float(p['distance_nll'][j].mean())}
            if curves:
                row['T'] = T[j].cpu().tolist()
            rows.append(row)
    result = {k: v/count for k, v in totals.items()}
    result.update(rays=len(profiles), samples=count,
        max_abs_T_error=max(r['max_abs_T_error'] for r in rows),
        mean_ray_max_abs_T_error=float(np.mean([r['max_abs_T_error'] for r in rows])),
        p95_ray_max_abs_T_error=float(np.quantile([r['max_abs_T_error'] for r in rows], .95)),
        rmse_T=float(np.sqrt(np.mean([r['mse_T'] for r in rows]))),
        mean_abs_escape_error=float(np.mean([abs(r['escape_error']) for r in rows])),
        max_abs_escape_error=max(abs(r['escape_error']) for r in rows))
    if curves:
        result['per_ray'] = rows
    return result


def train_one(config, data_hash, splits, mode, seed, output):
    weights = objective_weights(config.get('distance_objective'), mode)
    destination = output/'runs'/f'{mode}_{seed}'
    destination.mkdir(parents=True, exist_ok=True)
    report_path = destination/'run_report.json'
    if report_path.exists():
        report = read_json(report_path)
        if report['config'] != config or report['dataset_sha256'] != data_hash:
            raise ValueError('completed run has a different experiment identity')
        return report
    model = make_model(config, mode, seed).to(config['device'])
    initial_hash, encoder_hash = parameter_hash(model), parameter_hash(model.encoder)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'],
                                 weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, config['epochs'], eta_min=config['minimum_learning_rate'])
    history, best, start_epoch, elapsed = [], math.inf, 1, 0.
    latest = destination/'latest.pt'
    if latest.exists():
        saved = torch.load(latest, weights_only=True, map_location=config['device'])
        if saved['config'] != config or saved['dataset_sha256'] != data_hash:
            raise ValueError('resume identity differs')
        model.load_state_dict(saved['model_state'])
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        history, best = saved['history'], saved['best_score']
        start_epoch, elapsed = saved['epoch']+1, saved['elapsed']
    synchronize(config['device'])
    began = time.perf_counter()
    for epoch in range(start_epoch, config['epochs']+1):
        model.train()
        generator = torch.Generator().manual_seed(seed+epoch)
        order = torch.randperm(len(splits['train']), generator=generator).tolist()
        total, count = 0., 0
        components = {key: 0. for key in ('loss', 'residual_l2', 'transmittance_mse')}
        for begin in range(0, len(order), config['profiles_per_batch']):
            ps = [splits['train'][i] for i in order[begin:begin+config['profiles_per_batch']]]
            indices = [torch.randperm(len(p['hit']), generator=generator)[
                :config['samples_per_profile_per_epoch']].to(config['device']) for p in ps]
            batch = make_batch(ps, indices)
            optimizer.zero_grad(set_to_none=True)
            result = predictions(model, batch)
            terms = distance_objective(model, batch, result, weights)
            loss = terms['loss']
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite distance loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'],
                                          error_if_nonfinite=True)
            optimizer.step()
            total += float(terms['nll'].detach())*batch['hit'].numel()
            for key in components:
                components[key] += float(terms[key].detach())*batch['hit'].numel()
            count += batch['hit'].numel()
        scheduler.step()
        if epoch % config['validation_every'] and epoch != config['epochs']:
            continue
        validation = evaluate(model, splits['validation'])
        score = validation['distance_nll']
        bundle = {'format': 'first_hit_distance_design', 'config': config, 'mode': mode,
                  'seed': seed, 'dataset_sha256': data_hash, 'epoch': epoch,
                  'model_state': model.state_dict(), 'validation': validation}
        if score < best:
            best = score
            torch.save(bundle, destination/'best.pt')
        synchronize(config['device'])
        seconds = elapsed+time.perf_counter()-began
        history.append({'epoch': epoch, 'train_nll': total/count,
                        'validation': validation, 'elapsed_seconds': seconds,
                        **({'train_'+k: v/count for k, v in components.items()}
                           if config.get('distance_objective') else {})})
        torch.save({**bundle, 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                    'history': history, 'best_score': best, 'elapsed': seconds}, latest)
        write_json(destination/'history.json', history)
        if epoch % 50 == 0 or epoch == config['epochs']:
            print(f'{mode} seed={seed} epoch={epoch}: val NLL={score:.6f}, '
                  f'mean max T={validation["mean_ray_max_abs_T_error"]:.5f}, {seconds:.1f}s', flush=True)
    model, chosen = load_run(destination/'best.pt', config['device'])
    report = {'config': config, 'mode': mode, 'seed': seed, 'dataset_sha256': data_hash,
              'parameters': sum(p.numel() for p in model.parameters()),
              'initial_parameter_sha256': initial_hash, 'initial_encoder_sha256': encoder_hash,
              'selected_epoch': chosen['epoch'], 'training_seconds': history[-1]['elapsed_seconds'],
              'selection': 'minimum validation marginal distance NLL',
              'metrics': {s: evaluate(model, ps) for s, ps in splits.items()}}
    write_json(report_path, report)
    return report


@torch.no_grad()
def numerical_audit(config, data, output):
    c, rows = config['model'], []
    for p in (p for p in data['profiles'] if p['split'] == 'validation'):
        q = torch.linspace(0, p['horizon'], 4099, dtype=torch.float64)
        prior = torch.special.log_ndtr(p['features'][0, 0]*torch.tensor([1., -1.]))
        prior -= torch.logsumexp(prior, 0)
        def curve(bins, order):
            rates = local_bin_log_rates(p['features'], bins, order)
            H = piecewise_distribution(rates, q[None], p['horizon'])[0]
            return torch.logsumexp(prior[:, None]-H, 0).exp()
        base = curve(c['bins'], c['quadrature_order'])
        rows.append({'pixel': list(pixel_of(p)),
                     'quadrature_error': float((base-curve(c['bins'], 32)).abs().max()),
                     'grid_error': float((base-curve(c['bins']*2, 16)).abs().max())})
    report = {'rows': rows, 'max_grid_error': max(r['grid_error'] for r in rows),
              'max_quadrature_error': max(r['quadrature_error'] for r in rows)}
    report['passed'] = report['max_grid_error'] <= config['max_baseline_T_discretization_error']
    write_json(output/'quadrature_audit.json', report)
    if not report['passed']:
        raise ArithmeticError(f'baseline grid needs refinement before training: {report}')
    return report


def reference_audit(config, data, output):
    """Sensitivity to the reference step and finite memory, using validation rays only."""
    path = output/'reference_audit.json'
    if path.exists():
        return read_json(path)
    spec = config['reference_audit']
    candidates = [p for p in data['profiles'] if p['split'] == 'validation']
    # Geometry-only selection: largest dimensionless mean derivative.
    chosen = sorted(candidates, key=lambda p: float(
        (p['features'][:, 2:4].abs()/p['features'][:, 4:5].exp()).max()), reverse=True)[:spec['rays']]
    base = read_json(resolve(config['collection_config']))
    rows, collected = [], {}
    for variant in ('base', 'fine', 'history'):
        settings = copy.deepcopy(base)
        settings['output_directory'] = str(output/'reference_audit'/variant)
        settings['sources'] = [{'scene_id': 'reference_audit_sigma002', 'split': 'validation',
            'render_config': config['render_config'], 'realizations_per_profile': spec['trajectories'],
            'pixels': [list(pixel_of(p)) for p in chosen]}]
        settings['seed'] = 20261025
        if variant == 'fine':
            settings['reference_sampler']['step_size'] = spec['fine_step']
        if variant == 'history':
            settings['reference_sampler']['history_points'] = spec['history_points']
        request = output/'reference_audit'/f'{variant}.json'
        write_json(request, settings)
        collect(request)
        collected[variant] = load_data(resolve(settings['output_directory'])/'dataset.pt')['profiles']
    for i, p in enumerate(chosen):
        curves, speeds = {}, {}
        q = torch.linspace(0, p['horizon'], 2049, dtype=torch.float64)
        for name, profiles in collected.items():
            item = profiles[i]
            curves[name] = 1-torch.searchsorted(item['distance'][item['hit']].sort().values,
                q, right=True).double()/len(item['hit'])
            speeds[name] = item['speed'][item['hit']]
        for variant in ('fine', 'history'):
            rows.append({'pixel': list(pixel_of(p)), 'comparison': f'{variant}-base',
                'max_T_difference': float((curves[variant]-curves['base']).abs().max()),
                'mean_speed_base': float(speeds['base'].mean()),
                'mean_speed_variant': float(speeds[variant].mean())})
    # Sum of two marginal DKW bands, union over every empirical CDF used.
    bound = 2*math.sqrt(math.log(2*3*len(chosen)/.01)/(2*spec['trajectories']))
    report = {'selection': 'validation geometry only; largest absolute mean derivatives',
              'scope': 'teacher sensitivity, not certification of the continuous GP',
              'simultaneous_99_percent_two_sample_bound': bound, 'records': rows,
              'base_sampler': data['reference_sampler'], 'config': spec}
    write_json(path, report)
    return report


def restrict_profile(features, fraction):
    """Exact restriction of the existing mean; no resampling of its cubics."""
    widths = features[:, 4].exp()
    ends = widths.cumsum(0)
    limit = ends[-1]*fraction
    i = min(int(torch.searchsorted(ends, limit)), len(features)-1)
    start = ends[i]-widths[i]
    width = limit-start
    if width <= 0:
        return features[:i].clone()
    u = width/widths[i]
    b, db = mean_at(features[i], u)
    last = torch.stack((features[i, 0], b, features[i, 2]*u, db*width, width.log()))
    return torch.cat((features[:i], last[None]))


def alter_suffix(features, fraction, amplitude=2.):
    """Change only the suffix by a quadratic with zero value/slope at the cut."""
    widths = features[:, 4].exp()
    ends = widths.cumsum(0)
    limit, horizon = ends[-1]*fraction, ends[-1]
    i = min(int(torch.searchsorted(ends, limit)), len(features)-1)
    remaining = ends[i]-limit
    if remaining > 0:
        u = 1-remaining/widths[i]
        b, db = mean_at(features[i], u)
        first = torch.stack((b, features[i, 1], db*remaining,
                             features[i, 3]*remaining/widths[i], remaining.log()))
        suffix = torch.cat((first[None], features[i+1:]))
    else:
        suffix = features[i+1:].clone()
    prefix = restrict_profile(features, fraction)
    widths = suffix[:, 4].exp()
    right = widths.cumsum(0)/(horizon-limit)
    left = right-widths/(horizon-limit)
    suffix[:, 0] += amplitude*left.square()
    suffix[:, 1] += amplitude*right.square()
    suffix[:, 2] += 2*amplitude*left*widths/(horizon-limit)
    suffix[:, 3] += 2*amplitude*right*widths/(horizon-limit)
    return torch.cat((prefix, suffix))


@torch.no_grad()
def consistency(config, data, output):
    seed = config['seeds'][0]
    models = {m: load_run(output/'runs'/f'{m}_{seed}'/'best.pt')[0] for m in MODES}
    rows = []
    for p in (p for p in data['profiles'] if p['split'] == 'test'):
        f = p['features']
        split = subdivide_features(f, torch.full((len(f),), config['consistency']['subdivisions'],
                                                dtype=torch.long))
        q = torch.linspace(0, p['horizon'], 513, dtype=torch.float64)
        for mode, model in models.items():
            full = model.query(f, q)['T']
            rows.append({'pixel': list(pixel_of(p)), 'mode': mode, 'kind': 'repartition',
                         'max_T_difference': float((full-model.query(split, q)['T']).abs().max())})
            for fraction in config['consistency']['prefix_fractions']:
                cut = restrict_profile(f, fraction)
                prefix_q = q[q <= p['horizon']*fraction]
                rows.append({'pixel': list(pixel_of(p)), 'mode': mode,
                    'kind': f'prefix_{fraction}',
                    'max_T_difference': float((full[:len(prefix_q)]-model.query(cut, prefix_q)['T']).abs().max())})
            fraction = config['consistency']['suffix_test_fraction']
            altered = alter_suffix(f, fraction)
            prefix_q = q[q <= p['horizon']*fraction]
            rows.append({'pixel': list(pixel_of(p)), 'mode': mode, 'kind': 'future_change',
                'max_T_difference': float((full[:len(prefix_q)]-model.query(altered, prefix_q)['T']).abs().max())})
    result = {'scope': 'first-seed checkpoints, no fitting on these diagnostics', 'records': rows}
    write_json(output/'consistency.json', result)
    return result


@torch.no_grad()
def sampling_audit(config, data, output):
    seed, n = config['seeds'][0], 8192
    models = {'local': None, **{m: load_run(output/'runs'/f'{m}_{seed}'/'best.pt')[0] for m in MODES}}
    generator = torch.Generator().manual_seed(20261026)
    rows = []
    for p in (p for p in data['profiles'] if p['split'] == 'test'):
        fixed = resample_profile(p['features'], config['model']['profile_points'])
        baseline = local_bin_log_rates(p['features'], config['model']['bins'],
                                       config['model']['quadrature_order'])
        positive = torch.rand(n, generator=generator, dtype=torch.float64) < torch.special.ndtr(p['features'][0, 0])
        u = torch.rand(n, generator=generator, dtype=torch.float64)
        q = torch.linspace(0, p['horizon'], 1025, dtype=torch.float64)
        prepared = prepare({'profiles': [p]}, config, 'cpu')['test']
        batch = make_batch(prepared)
        for mode, model in models.items():
            draws, hit = torch.empty(n, dtype=torch.float64), torch.zeros(n, dtype=torch.bool)
            for k, sign in enumerate((1., -1.)):
                chosen = positive if k == 0 else ~positive
                if chosen.any():
                    d, h = sample_distance(model, fixed[None], torch.tensor([sign]), baseline[k:k+1],
                        torch.tensor([p['horizon']], dtype=torch.float64), u[chosen][None])
                    draws[chosen], hit[chosen] = d[0], h[0]
            pred = predictions(model, batch)
            truth = transmittance(model, pred['encoded'], q[None], batch['horizon'], pred['log_prior'])[0]
            empirical = 1-torch.searchsorted(draws[hit].sort().values, q, right=True).double()/n
            if not torch.isfinite(draws).all() or (draws < 0).any() or (draws > p['horizon']+1e-10).any():
                raise ArithmeticError('invalid inverse samples')
            rows.append({'pixel': list(pixel_of(p)), 'mode': mode,
                         'max_CDF_difference': float((truth-empirical).abs().max())})
    bound = math.sqrt(math.log(2*len(rows)/.01)/(2*n))
    result = {'records': rows, 'simultaneous_99_percent_DKW_bound': bound,
              'passed': max(r['max_CDF_difference'] for r in rows) <= bound}
    write_json(output/'sampling_audit.json', result)
    if not result['passed']:
        raise ArithmeticError('sampling audit failed')
    return result


@torch.no_grad()
def benchmark(config, data, output):
    profiles = [p for p in data['profiles'] if p['split'] == 'test']
    seed = config['seeds'][0]
    models = {'local': None, **{m: load_run(output/'runs'/f'{m}_{seed}'/'best.pt')[0] for m in MODES}}
    cache = prepare({'profiles': profiles}, config, 'cpu')['test']
    generator = torch.Generator().manual_seed(20261027)
    randoms = torch.rand(len(profiles), 2, generator=generator, dtype=torch.float64)
    def call(mode, i, prepared):
        p, row = profiles[i], cache[i]
        k = int(randoms[i, 0] >= torch.special.ndtr(row['b0']))
        sign = torch.tensor([1. if k == 0 else -1.])
        fixed = row['fixed'] if prepared or mode == 'local' else resample_profile(
            p['features'], config['model']['profile_points'])
        baseline = row['baseline']
        if not prepared and mode in ('local', 'residual'):
            baseline = local_bin_log_rates(p['features'], config['model']['bins'],
                                           config['model']['quadrature_order'])
        d, hit = sample_distance(models[mode], fixed[None], sign, baseline[k:k+1],
                                 row['horizon'][None], randoms[i:i+1, 1:])
        return float(d[0, 0])+int(hit[0, 0])
    cases = [(m, ready) for ready in (True, False) for m in models]
    timing, checksums = {}, {}
    for m, ready in cases:
        for i in range(len(profiles)):
            call(m, i, ready)
        timing[f'{m}_{"prepared" if ready else "from_cubics"}'] = []
    for repeat in range(config['benchmark_repeats']):
        for mode, ready in cases[repeat % len(cases):]+cases[:repeat % len(cases)]:
            name = f'{mode}_{"prepared" if ready else "from_cubics"}'
            checksum = 0.
            start = time.perf_counter()
            for _ in range(config['benchmark_inner_repeats']):
                for i in range(len(profiles)):
                    checksum += call(mode, i, ready)
            timing[name].append((time.perf_counter()-start)*1e6/(
                len(profiles)*config['benchmark_inner_repeats']))
            if name in checksums and abs(checksums[name]-checksum) > 1e-8:
                raise ArithmeticError('benchmark samples changed')
            checksums[name] = checksum
    for mode in models:
        if abs(checksums[f'{mode}_prepared']-checksums[f'{mode}_from_cubics']) > 1e-8:
            raise ArithmeticError('prepared and constructed distributions differ')
    result = {'scope': 'CPU single ray, one distance sample; no VDB access, gradient, or rendering',
              'threads': torch.get_num_threads(), 'torch': str(torch.__version__),
              'platform': platform.platform(),
              'timings': {k: {'median_us': statistics.median(v), 'runs_us': v} for k, v in timing.items()}}
    if torch.cuda.is_available():
        gpu = {}
        for size in (len(profiles), 256):
            indices = torch.arange(size) % len(profiles)
            batch = make_batch([cache[i] for i in indices.tolist()])
            fixed, horizon = batch['fixed'].cuda(), batch['horizon'].cuda()
            k = (randoms[indices, 0] >= torch.special.ndtr(batch['b0'])).long()
            sign = (1.-2*k).cuda()
            base = batch['baseline'][torch.arange(size), k].cuda()
            uniforms = randoms[indices, 1:].cuda()
            for mode, model in models.items():
                model = None if model is None else model.cuda()
                for _ in range(5):
                    sample_distance(model, fixed, sign, base, horizon, uniforms)
                runs = []
                for _ in range(config['benchmark_repeats']):
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    for _ in range(20):
                        sample_distance(model, fixed, sign, base, horizon, uniforms)
                    torch.cuda.synchronize()
                    runs.append((time.perf_counter()-start)*1e6/(20*size))
                gpu[f'{mode}_batch_{size}'] = {'median_us_per_ray': statistics.median(runs),
                                              'runs_us_per_ray': runs}
        result['cuda_prepared'] = gpu
        result['cuda_device'] = torch.cuda.get_device_name()
        result['cuda_scope'] = 'prepared inputs on device; amortized batch throughput, excluding VDB/baseline preparation'
    write_json(output/'benchmark.json', result)
    return result


@torch.no_grad()
def baseline_tradeoff(config, data, output):
    """Post-training baseline approximations; selection uses validation only."""
    seed = config['seeds'][0]
    model, _ = load_run(output/'runs'/f'residual_{seed}'/'best.pt')
    c = config['model']
    def baseline(p, method):
        if method == 'midpoint':
            q = (torch.arange(c['bins'], dtype=torch.float64)+.5)*(p['horizon']/c['bins'])
            b, db = local_profile(p['features'], q)
            signs = torch.tensor([1., -1.])[:, None]
            return local_log_hazard(signs*b, signs*db)
        return local_bin_log_rates(p['features'], c['bins'], 2 if method == 'gauss2' else c['quadrature_order'])
    def measure(ps, method):
        values = []
        for repeat in range(config['benchmark_repeats']+1):
            start = time.perf_counter()
            for p in ps:
                fixed = resample_profile(p['features'], c['profile_points'])
                rates = baseline(p, method)
                sample_distance(model, fixed[None], torch.ones(1), rates[:1],
                                torch.tensor([p['horizon']], dtype=torch.float64),
                                torch.tensor([[.5]], dtype=torch.float64))
            if repeat:
                values.append((time.perf_counter()-start)*1e6/len(ps))
        return {'median_us': statistics.median(values), 'runs_us': values}
    validation = [p for p in data['profiles'] if p['split'] == 'validation']
    prepared = prepare({'profiles': validation}, config, 'cpu')['validation']
    exact = evaluate(model, prepared, curves=True)
    rows = []
    for method in ('gauss8', 'gauss2', 'midpoint'):
        changed = [{**row, 'baseline': baseline(p, method)} for p, row in zip(validation, prepared)]
        metrics = evaluate(model, changed, curves=True)
        difference = max(float(np.max(np.abs(np.array(a['T'])-np.array(b['T']))))
                         for a, b in zip(exact['per_ray'], metrics['per_ray']))
        rows.append({'method': method, 'max_T_change': difference,
                     'validation_nll': metrics['distance_nll'], **measure(validation, method)})
    tolerance = config['max_baseline_T_discretization_error']
    selected = min((r for r in rows if r['max_T_change'] <= tolerance), key=lambda r: r['median_us'])['method']
    test = [p for p in data['profiles'] if p['split'] == 'test']
    prepared = prepare({'profiles': test}, config, 'cpu')['test']
    exact = evaluate(model, prepared, curves=True)
    changed = [{**row, 'baseline': baseline(p, selected)} for p, row in zip(test, prepared)]
    metrics = evaluate(model, changed, curves=True)
    difference = max(float(np.max(np.abs(np.array(a['T'])-np.array(b['T']))))
                     for a, b in zip(exact['per_ray'], metrics['per_ray']))
    metrics.pop('per_ray')
    result = {'seed': seed, 'scope': 'unchanged residual checkpoint, both baseline signs constructed; '
              'CPU single-ray sampling from existing cubics, no VDB or gradients',
              'selection': 'fastest validation timing with max T change at most the tolerance',
              'validation_tolerance': tolerance, 'validation': rows, 'selected': selected,
              'test_max_T_change': difference, 'test_metrics': metrics,
              'test_timing': measure(test, selected)}
    write_json(output/'baseline_tradeoff.json', result)
    return result


def analyze(config, data, splits, output):
    runs = [read_json(output/'runs'/f'{m}_{s}'/'run_report.json') for m in MODES for s in config['seeds']]
    for seed in config['seeds']:
        pair = [r for r in runs if r['seed'] == seed]
        if len({r['initial_encoder_sha256'] for r in pair}) != 1:
            raise AssertionError('encoder initialization differs across representations')
        if len({r['initial_parameter_sha256'] for r in pair if r['mode'] != 'cdf'}) != 1:
            raise AssertionError('direct/residual initial parameters differ')
    baseline = {s: evaluate(None, p) for s, p in splits.items()}
    metrics = (*METRICS, 'p95_ray_max_abs_T_error')
    summary = {'local': {k: {'mean': baseline['test'][k], 'std': 0.} for k in metrics}}
    for mode in MODES:
        chosen = [r for r in runs if r['mode'] == mode]
        summary[mode] = {k: {'mean': float(np.mean([r['metrics']['test'][k] for r in chosen])),
                            'std': float(np.std([r['metrics']['test'][k] for r in chosen], ddof=1)) if len(chosen) > 1 else 0.}
                         for k in metrics}
    curves = {'local': evaluate(None, splits['test'], curves=True)['per_ray']}
    for mode in MODES:
        for seed in config['seeds']:
            model, _ = load_run(output/'runs'/f'{mode}_{seed}'/'best.pt', config['device'])
            curves[f'{mode}_{seed}'] = evaluate(model, splits['test'], curves=True)['per_ray']
    report = {'config': config, 'runs': runs, 'baseline': baseline, 'summary': summary, 'curves': curves,
              'data_audit': read_json(output/'data_audit.json')}
    write_json(output/'analysis.json', report)
    return report


def report(config, data, output):
    analysis = read_json(output/'analysis.json')
    plt = plotting()
    profiles = [p for p in data['profiles'] if p['split'] == 'test']
    ncols, nrows = 4, math.ceil(len(profiles)/4)
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 2.6*nrows), squeeze=False)
    colors = {'local': '#2563eb', 'direct': '#ea580c', 'residual': '#059669', 'cdf': '#9333ea'}
    seed = config['seeds'][0]
    for i, (axis, p) in enumerate(zip(axes.flat, profiles)):
        q = np.linspace(0, p['horizon'], config['curve_points'])
        hits = p['distance'][p['hit']].sort().values
        reference = 1-torch.searchsorted(hits, torch.tensor(q), right=True).numpy()/len(p['hit'])
        band = math.sqrt(math.log(2*len(profiles)/.05)/(2*len(p['hit'])))
        axis.fill_between(q, np.maximum(0, reference-band), np.minimum(1, reference+band),
                          color='#64748b', alpha=.15)
        axis.plot(q, reference, 'k', label='Reference', lw=1.6)
        for mode in ('local', *MODES):
            key = mode if mode == 'local' else f'{mode}_{seed}'
            axis.plot(q, analysis['curves'][key][i]['T'], color=colors[mode], label=mode)
        axis.set(title=str(pixel_of(p)), xlabel='q = s / ell', ylim=(-.02, 1.02))
    for axis in list(axes.flat)[len(profiles):]:
        axis.set_visible(False)
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    save_figure(plt, fig, output, 'distance_curves')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for mode in MODES:
        for seed_i, s in enumerate(config['seeds']):
            history = read_json(output/'runs'/f'{mode}_{s}'/'history.json')
            axes[0].plot([r['epoch'] for r in history],
                [r['validation']['distance_nll'] for r in history], color=colors[mode],
                alpha=.55, label=mode if seed_i == 0 else None)
        means = [analysis['curves'][f'{mode}_{s}'] for s in config['seeds']]
        error = np.mean([[r['max_abs_T_error'] for r in rows] for rows in means], axis=0)
        axes[1].plot(np.sort(error), label=mode, color=colors[mode])
    axes[0].set(xlabel='Epoch', ylabel='Validation marginal distance NLL')
    axes[1].plot(np.sort([r['max_abs_T_error'] for r in analysis['curves']['local']]),
                 label='local', color=colors['local'])
    axes[1].set(xlabel='Sorted held-out ray', ylabel='Max transmittance error (mean over seeds)')
    for axis in axes:
        axis.legend()
    fig.tight_layout()
    save_figure(plt, fig, output, 'training_and_errors')
    keys = ('distance_nll', 'mean_ray_max_abs_T_error', 'max_abs_T_error', 'rmse_T', 'mean_abs_escape_error')
    table = '<tr><th>Method</th>'+''.join(f'<th>{k}</th>' for k in keys)+'</tr>'
    for mode, values in analysis['summary'].items():
        table += f'<tr><td>{mode}</td>'+''.join(
            f'<td>{values[k]["mean"]:.6f} ± {values[k]["std"]:.6f}</td>' for k in keys)+'</tr>'
    timing = read_json(output/'benchmark.json') if (output/'benchmark.json').exists() else None
    extras = ''
    if timing:
        extras += '<h2>CPU sampling (microseconds / ray)</h2><table>'
        for name, value in timing['timings'].items():
            extras += f'<tr><td>{name}</td><td>{value["median_us"]:.2f}</td></tr>'
        extras += '</table><p>'+html.escape(timing['scope'])+'</p>'
        if timing.get('cuda_prepared'):
            extras += '<h2>CUDA prepared batch throughput (microseconds / ray)</h2><table>'
            for name, value in timing['cuda_prepared'].items():
                extras += f'<tr><td>{name}</td><td>{value["median_us_per_ray"]:.3f}</td></tr>'
            extras += '</table><p>'+html.escape(timing['cuda_scope'])+'</p>'
    if (output/'consistency.json').exists():
        audit = read_json(output/'consistency.json')
        extras += ('<h2>Consistency diagnostics</h2><p>Prefix truncation changes the fixed input grid and '
                   'the output coordinate scale. Future-change keeps the horizon and prefix mean fixed.</p>'
                   '<table><tr><th>Mode</th><th>Change</th><th>Mean max T difference</th><th>Worst</th></tr>')
        for mode in MODES:
            for kind in sorted({r['kind'] for r in audit['records']}):
                values = [r['max_T_difference'] for r in audit['records']
                          if r['mode'] == mode and r['kind'] == kind]
                extras += f'<tr><td>{mode}</td><td>{kind}</td><td>{np.mean(values):.6g}</td><td>{max(values):.6g}</td></tr>'
        extras += '</table>'
    if (output/'gradient_probe.json').exists():
        probe = read_json(output/'gradient_probe.json')
        extras += ('<h2>Conditional gradient probe</h2><p>'+html.escape(probe['scope'])+'</p>'
                   '<table><tr><th>Mode</th><th>Regime</th><th>Speed NLL</th><th>PIT KS</th>'
                   '<th>Joint NLL</th><th>Mean max T error</th></tr>')
        for row in probe['runs']:
            m = row['metrics']['test']
            error = row['distance_metrics']['test']['mean_ray_max_abs_T_error']
            extras += (f'<tr><td>{row["mode"]}</td><td>{row["regime"]}</td><td>{m["speed_nll"]:.6f}</td>'
                       f'<td>{m["pit_ks"]:.6f}</td><td>{m["joint_nll"]:.6f}</td><td>{error:.6f}</td></tr>')
        extras += '</table>'
    if (output/'baseline_tradeoff.json').exists():
        tradeoff = read_json(output/'baseline_tradeoff.json')
        extras += ('<h2>Cheaper local baseline, unchanged residual network</h2>'
                   '<table><tr><th>Method</th><th>Validation max T change</th><th>CPU microseconds / ray</th></tr>')
        for row in tradeoff['validation']:
            extras += f'<tr><td>{row["method"]}</td><td>{row["max_T_change"]:.6g}</td><td>{row["median_us"]:.2f}</td></tr>'
        extras += (f'</table><p>Selected using validation: {tradeoff["selected"]}. '
                   f'Test max T change: {tradeoff["test_max_T_change"]:.6g}.</p>')
    for name in ('reference_audit', 'consistency', 'sampling_audit', 'gradient_probe', 'baseline_tradeoff'):
        if (output/f'{name}.json').exists():
            extras += f'<p><a href="{name}.json">{name}</a></p>'
    page = ('<!doctype html><meta charset="utf-8"><title>First-hit distance design: sigma=0.02</title>'
        '<style>body{font:15px system-ui;max-width:1500px;margin:32px auto;padding:0 20px;line-height:1.6}'
        'table{border-collapse:collapse}td,th{padding:8px;border:1px solid #ddd}img{max-width:100%}</style>'
        '<h1>First-hit distance representation study — sigma=0.02</h1>'
        '<p>Same fixed-profile encoder, paired seeds and label batches, distance-only censored likelihood. '
        'CDF has a different output-head size. Checkpoints selected using validation distance NLL.</p>'
        '<p>The residual method also evaluates a high-resolution analytic local baseline from the original '
        'cubic profile. This comparison measures the complete hybrid design; it does not isolate '
        'residual parameterization from the extra analytic information.</p>'
        '<p>Reference: finite-memory H1/B8 Tungsten-linear, step 1/64. '
        'This measures fitting to that teacher, not exact continuous-GP accuracy.</p>'
        f'<table>{table}</table><h2>Training and errors</h2><img src="training_and_errors.png">'
        '<h2>Held-out rays (first seed)</h2><p>Shading: simultaneous 95% DKW bands across the 24 reference CDFs.</p>'
        '<img src="distance_curves.png">'
        +extras+'<p><a href="analysis.json">Full analysis</a> · <a href="study_config.json">Configuration</a>'
        ' · <a href="data_audit.json">Data provenance</a></p>')
    (output/'summary.html').write_text(page, encoding='utf-8')
    return {'summary': str(output/'summary.html'), 'metrics': analysis['summary']}


def run(config_path, phase='all'):
    config = read_json(config_path)
    output = resolve(config['output_directory'])
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(config['cpu_threads'])
    torch.backends.cuda.matmul.allow_tf32 = False
    if (output/'study_config.json').exists() and read_json(output/'study_config.json') != config:
        raise ValueError('experiment configuration changed; use another output directory')
    if phase in ('all', 'collect') and not resolve(config['dataset']).exists():
        collect(resolve(config['collection_config']))
    if phase == 'collect':
        return {'dataset': config['dataset']}
    data = load_data(resolve(config['dataset']))
    if phase in ('all', 'audit'):
        audit = audit_data(data, config)
        audit['physical_sigma'] = sorted({p['geometry']['sigma'] for p in data['profiles']})
        audit['physical_ell'] = sorted({p['geometry']['ell'] for p in data['profiles']})
        audit['field_sha256'] = digest(resolve(read_json(resolve(config['render_config']))['field']['grid_file']))
        write_json(output/'data_audit.json', audit)
        numerical_audit(config, data, output)
        reference_audit(config, data, output)
    if phase == 'audit':
        return {'audit': str(output/'data_audit.json')}
    if phase in ('all', 'train', 'evaluate'):
        if not (output/'quadrature_audit.json').exists() or not read_json(output/'quadrature_audit.json')['passed']:
            raise ValueError('run the numerical audit before training')
        write_json(output/'study_config.json', config)
        splits = prepare(data, config, config['device'])
        if phase in ('all', 'train'):
            for seed in config['seeds']:
                for mode in MODES:
                    train_one(config, digest(resolve(config['dataset'])), splits, mode, seed, output)
        analyze(config, data, splits, output)
    if phase in ('all', 'evaluate'):
        consistency(config, data, output)
        sampling_audit(config, data, output)
    if phase in ('all', 'probe'):
        from .distance_probe import run_probe
        run_probe(config, data, output)
    if phase in ('all', 'benchmark'):
        benchmark(config, data, output)
        baseline_tradeoff(config, data, output)
    if phase in ('all', 'report'):
        return report(config, data, output)
    return {'output': str(output), 'phase': phase}
