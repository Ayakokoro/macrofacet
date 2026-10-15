"""Reproducible, paired-seed distance-only comparison on existing camera rays."""
import hashlib
import html
import math
import os
from pathlib import Path
import platform
import statistics
import time

import numpy as np
import torch

from renewal.collect import digest, resolve, write_json
from .camera_study import pixel_of, plotting, save_figure, wilson, write_csv
from .data import load_data, read_json
from .hazard import (HazardConfig, HazardNetwork, batch_predictions,
                     inverse_piecewise, local_bin_log_rates, piecewise_distribution)
from .spline import resample_profile, rational_quadratic, spline_parameters
from .train import load_model


MODES = ('direct', 'residual')
SPLITS = ('train', 'validation', 'test')
FIELDS = ('fixed', 'baseline', 'distance', 'hit', 'initial_sign', 'horizon', 'b0', 'reference_T')
METRICS = ('distance_nll', 'max_abs_T_error', 'mean_ray_max_abs_T_error', 'rmse_T',
           'mean_abs_escape_error', 'max_abs_escape_error')


def make_batch(profiles, indices=None):
    labels = ('distance', 'hit', 'initial_sign')
    return {key: torch.stack([p[key] if indices is None or key not in labels else p[key][indices[i]]
                              for i, p in enumerate(profiles)]) for key in FIELDS}


def prepare(data, config, device):
    result = {split: [] for split in SPLITS}
    for p in data['profiles']:
        q = torch.linspace(0, p['horizon'], config['curve_points'], dtype=torch.float64)
        sorted_hits = p['distance'][p['hit']].sort().values
        observed = 1-torch.searchsorted(sorted_hits, q, right=True).double()/len(p['hit'])
        row = {k: p[k] for k in ('distance', 'hit', 'initial_sign')}
        row.update(fixed=resample_profile(p['features'], config['model']['profile_points']),
                   baseline=local_bin_log_rates(p['features'], config['model']['bins'],
                                                config['model']['quadrature_order']),
                   horizon=torch.tensor(p['horizon'], dtype=torch.float64),
                   b0=p['features'][0, 0], reference_T=observed)
        result[p['split']].append({k: v.to(device) for k, v in row.items()})
    return result


@torch.no_grad()
def evaluate(model, profiles, batch_size=8, curves=False):
    if model is not None:
        model.eval()
    totals = {key: 0. for key in ('distance_nll', 'training_nll', 'complete_distance_nll')}
    count = hits = 0
    rows = []
    for begin in range(0, len(profiles), batch_size):
        batch = make_batch(profiles[begin:begin+batch_size])
        prediction = batch_predictions(model, batch)
        for key in totals:
            totals[key] += float(prediction[key].sum())
        count += batch['hit'].numel()
        hits += int(batch['hit'].sum())
        unit = torch.linspace(0, 1, batch['reference_T'].shape[-1], dtype=torch.float64,
                              device=batch['fixed'].device)
        q = batch['horizon'][:, None]*unit
        H, _ = piecewise_distribution(prediction['log_rates'], q[:, None], batch['horizon'][:, None])
        signs = q.new_tensor([1., -1.])
        prior = torch.special.log_ndtr(batch['b0'][:, None]*signs)
        prior -= torch.logsumexp(prior, -1, keepdim=True)
        T = torch.logsumexp(prior[:, :, None]-H, 1).exp()
        if (not torch.isfinite(T).all() or not torch.isfinite(prediction['distance_nll']).all()
                or (T[:, 1:]-T[:, :-1] > 1e-12).any() or (T[:, 0]-1).abs().max() > 1e-12):
            raise FloatingPointError('invalid hazard probability or likelihood')
        for j in range(len(T)):
            error = T[j]-batch['reference_T'][j]
            row = {'max_abs_T_error': float(error.abs().max()),
                   'mse_T': float(error.square().mean()),
                   'escape_error': float(error[-1]),
                   'distance_nll': float(prediction['distance_nll'][j].mean())}
            if curves:
                row['T'] = T[j].cpu().tolist()
                row['log_rates'] = prediction['log_rates'][j].cpu().tolist()
            rows.append(row)
    return {'rays': len(profiles), 'samples': count, 'hits': hits,
            **{k: v/count for k, v in totals.items()},
            'max_abs_T_error': max(r['max_abs_T_error'] for r in rows),
            'mean_ray_max_abs_T_error': float(np.mean([r['max_abs_T_error'] for r in rows])),
            'rmse_T': float(np.sqrt(np.mean([r['mse_T'] for r in rows]))),
            'mean_abs_escape_error': float(np.mean([abs(r['escape_error']) for r in rows])),
            'max_abs_escape_error': max(abs(r['escape_error']) for r in rows), 'per_ray': rows}


def parameter_hash(model):
    sha = hashlib.sha256()
    for name, value in model.named_parameters():
        sha.update(name.encode('utf-8'))
        sha.update(value.detach().cpu().numpy().tobytes())
    return sha.hexdigest()


def audit_data(data, config):
    if data['split_unit'] != 'ray' or data['reference_sampler']['crossing_rule'] != 'tungsten_linear':
        raise ValueError('requires independent camera rays and the existing Tungsten-linear labels')
    ids, pixels = set(), set()
    for p in data['profiles']:
        pixel = pixel_of(p)
        if p['id'] in ids or pixel in pixels:
            raise ValueError('ray leakage or duplicate profiles')
        ids.add(p['id']); pixels.add(pixel)
        if (not torch.all((p['initial_sign'] == 1) | (p['initial_sign'] == -1))
                or (p['distance'] <= 0).any() or (p['distance'] > p['horizon']+1e-10).any()
                or not torch.allclose(p['distance'][~p['hit']], torch.full_like(p['distance'][~p['hit']], p['horizon']))):
            raise ValueError('invalid censored first-hit labels')
    return {'dataset': config['dataset'], 'dataset_sha256': digest(resolve(config['dataset'])),
            'reference_sampler': data['reference_sampler'], 'ray_disjoint_splits': True,
            'distance_convention': 'q=s/ell from medium entry; excludes camera-to-entry distance',
            'local_definition': 'GP local-conditioning rate using the original mean gradient; no local_tangent normalization',
            'baseline_kernel': 'continuous stationary SE, unit variance and derivative variance in normalized coordinates',
            'teacher_note': 'finite-history H1/B8 Tungsten-linear teacher; residual can also capture its approximation error',
            'splits': {s: {'rays': sum(p['split'] == s for p in data['profiles']),
                            'samples': sum(len(p['hit']) for p in data['profiles'] if p['split'] == s)} for s in SPLITS}}


@torch.no_grad()
def audit_quadrature(data, config):
    c = config['model']
    records = []
    # Only validation geometry is used to audit the prescribed numerical grid.
    for p in (p for p in data['profiles'] if p['split'] == 'validation'):
        # Prime-sized grid also probes bin interiors, not just shared edges.
        q = torch.linspace(0, p['horizon'], 4099, dtype=torch.float64)
        prior = torch.special.log_ndtr(p['features'][0, 0]*torch.tensor([1., -1.]))
        prior -= torch.logsumexp(prior, 0)
        def transmittance(bins, order):
            rates = local_bin_log_rates(p['features'], bins, order)
            H, _ = piecewise_distribution(rates, q[None], p['horizon'])
            return torch.logsumexp(prior[:, None]-H, 0).exp()
        coarse = transmittance(c['bins'], c['quadrature_order'])
        accurate = transmittance(c['bins'], 32)
        dense = transmittance(2048, 16)
        records.append({'pixel': list(pixel_of(p)),
                        'max_T_quadrature_error': float((coarse-accurate).abs().max()),
                        'max_T_grid_error': float((coarse-dense).abs().max())})
    result = {'split': 'validation', 'grid_bins': c['bins'], 'quadrature_order': c['quadrature_order'],
              'reference_bins': 2048, 'reference_order': 16, 'per_ray': records,
              'max_T_quadrature_error': max(r['max_T_quadrature_error'] for r in records),
              'max_T_grid_error': max(r['max_T_grid_error'] for r in records)}
    result['passed'] = result['max_T_grid_error'] <= config['max_baseline_T_discretization_error']
    return result


def load_hazard(path, device='cpu'):
    bundle = torch.load(path, map_location=device, weights_only=True)
    if bundle.get('format') != 'first_hit_hazard_distance_experiment':
        raise ValueError('not a distance-only hazard checkpoint')
    model = HazardNetwork(HazardConfig(**bundle['model_config'])).to(device)
    model.load_state_dict(bundle['model_state'])
    model.eval()
    return model, bundle


def train_one(config, data_hash, splits, mode, seed, output):
    destination = output/f'{mode}_{seed}'
    if (destination/'run_report.json').exists():
        report = read_json(destination/'run_report.json')
        if report['dataset_sha256'] != data_hash or report['study_config'] != config:
            raise ValueError('existing run belongs to a different experiment')
        return report
    if (destination/'best.pt').exists():
        raise ValueError(f'incomplete run exists at {destination}; use a new study output directory')
    destination.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    model = HazardNetwork(HazardConfig(mode=mode, **config['model'])).to(config['device'])
    initial_hash = parameter_hash(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=config['minimum_learning_rate'])
    history, best, selected_epoch = [], math.inf, 0
    began = time.perf_counter()
    for epoch in range(1, config['epochs']+1):
        model.train()
        generator = torch.Generator().manual_seed(seed+epoch)
        order = torch.randperm(len(splits['train']), generator=generator).tolist()
        total = count = 0
        for begin in range(0, len(order), config['profiles_per_batch']):
            profiles = [splits['train'][i] for i in order[begin:begin+config['profiles_per_batch']]]
            indices = [torch.randperm(len(p['hit']), generator=generator)[:config['samples_per_profile_per_epoch']]
                       .to(config['device']) for p in profiles]
            batch = make_batch(profiles, indices)
            optimizer.zero_grad(set_to_none=True)
            result = batch_predictions(model, batch)
            loss = result['training_nll'].mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite hazard training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'], error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach())*batch['hit'].numel()
            count += batch['hit'].numel()
        scheduler.step()
        if epoch % config['validation_every'] and epoch != config['epochs']:
            continue
        validation = evaluate(model, splits['validation'])
        score = validation['distance_nll']
        improved = score < best
        if improved:
            best, selected_epoch = score, epoch
            bundle = {'format': 'first_hit_hazard_distance_experiment', 'version': 1,
                      'distance_only': True, 'dataset_sha256': data_hash, 'seed': seed,
                      'model_config': model.configuration(), 'model_state': model.state_dict(),
                      'epoch': epoch, 'validation': {k: v for k, v in validation.items() if k != 'per_ray'},
                      'study_config': config}
            torch.save(bundle, destination/'best.pt')
        row = {'epoch': epoch, 'training_conditional_nll': total/count,
               'validation': {k: v for k, v in validation.items() if k != 'per_ray'},
               'selected_so_far': improved, 'elapsed_seconds': time.perf_counter()-began,
               'learning_rate': optimizer.param_groups[0]['lr']}
        history.append(row)
        write_json(destination/'history.json', history)
        if epoch % 25 == 0 or epoch == config['epochs']:
            print(f"{mode} seed={seed} epoch={epoch}: val NLL={score:.6f}, "
                  f"mean/max T={validation['mean_ray_max_abs_T_error']:.4f}/{validation['max_abs_T_error']:.4f}, "
                  f"elapsed={row['elapsed_seconds']:.1f}s", flush=True)
    training_seconds = time.perf_counter()-began
    model, _ = load_hazard(destination/'best.pt', config['device'])
    # Test is evaluated only after validation has frozen checkpoint selection.
    metrics = {s: {k: v for k, v in evaluate(model, ps).items() if k != 'per_ray'} for s, ps in splits.items()}
    report = {'mode': mode, 'seed': seed, 'dataset_sha256': data_hash, 'study_config': config,
              'model_config': model.configuration(), 'parameters': sum(p.numel() for p in model.parameters()),
              'initial_parameter_sha256': initial_hash, 'training_seconds': training_seconds,
              'selected_epoch': selected_epoch, 'completed_epochs': config['epochs'],
              'selection_criterion': 'minimum validation marginal distance NLL', 'metrics': metrics}
    write_json(destination/'run_report.json', report)
    return report


def summarize_runs(runs, baseline, historical):
    summary = {'local': {key: {'mean': baseline[key], 'std': 0.} for key in METRICS}}
    for mode in MODES:
        chosen = [r for r in runs if r['mode'] == mode]
        summary[mode] = {key: {'mean': float(np.mean([r['metrics']['test'][key] for r in chosen])),
                              'std': float(np.std([r['metrics']['test'][key] for r in chosen], ddof=1)) if len(chosen) > 1 else 0.}
                         for key in METRICS}
    summary['spline_reference'] = {key: {'mean': historical[key], 'std': 0.} for key in METRICS}
    return summary


def historical_reference(config, data_hash, profiles):
    previous = read_json(resolve(config['spline_analysis']))
    if previous['dataset_sha256'] != data_hash:
        raise ValueError('spline reference uses a different dataset')
    selected = previous['selected']
    by_id = {r['profile']: r for r in previous['curves'][selected]}
    curves = [by_id[p['id']] for p in profiles]
    escapes = [r['terminal_spline_T']-r['terminal_reference_T'] for r in curves]
    result = {'name': selected, 'note': 'historical joint distance+gradient training; reference only, not a controlled ablation',
              'distance_nll': previous['runs'][selected]['metrics']['test']['distance_nll'],
              'max_abs_T_error': max(r['max_abs_T_error'] for r in curves),
              'mean_ray_max_abs_T_error': float(np.mean([r['max_abs_T_error'] for r in curves])),
              'rmse_T': float(np.sqrt(np.mean([r['rmse_T']**2 for r in curves]))),
              'mean_abs_escape_error': float(np.mean(np.abs(escapes))),
              'max_abs_escape_error': float(np.max(np.abs(escapes)))}
    return result, curves


def evaluate_study(config, data, output, splits):
    data_hash = digest(resolve(config['dataset']))
    baseline = {s: evaluate(None, ps, curves=(s == 'test')) for s, ps in splits.items()}
    profiles = [p for p in data['profiles'] if p['split'] == 'test']
    historical, historical_curves = historical_reference(config, data_hash, profiles)
    runs, predictions = [], {}
    for mode in MODES:
        for seed in config['seeds']:
            name = f'{mode}_{seed}'
            model, bundle = load_hazard(output/name/'best.pt', config['device'])
            if bundle['dataset_sha256'] != data_hash or bundle['study_config'] != config:
                raise ValueError('checkpoint belongs to a different study')
            runs.append(read_json(output/name/'run_report.json'))
            predictions[name] = evaluate(model, splits['test'], curves=True)
    for seed in config['seeds']:
        pair = [r for r in runs if r['seed'] == seed]
        if len({r['initial_parameter_sha256'] for r in pair}) != 1 or len({r['parameters'] for r in pair}) != 1:
            raise AssertionError('paired network initialization or parameter count differs')
    records = []
    for i, p in enumerate(profiles):
        q = np.linspace(0, p['horizon'], config['curve_points'])
        observed = splits['test'][i]['reference_T'].cpu().numpy()
        low, high = wilson(observed, len(p['hit']))
        records.append({'pixel': list(pixel_of(p)), 'profile': p['id'], 'samples': len(p['hit']),
                        'q': q.tolist(), 'distance': (q*p['geometry']['ell']).tolist(),
                        'reference': observed.tolist(), 'reference_low': low.tolist(), 'reference_high': high.tolist(),
                        'local': baseline['test']['per_ray'][i],
                        'spline': {'T': historical_curves[i]['spline'],
                                   'max_abs_T_error': historical_curves[i]['max_abs_T_error']},
                        **{name: result['per_ray'][i] for name, result in predictions.items()}})
    paired = []
    for seed in config['seeds']:
        d, r = predictions[f'direct_{seed}'], predictions[f'residual_{seed}']
        paired.append({'seed': seed, 'residual_minus_direct_NLL': r['distance_nll']-d['distance_nll'],
                       'residual_minus_direct_mean_ray_max_T_error': r['mean_ray_max_abs_T_error']-d['mean_ray_max_abs_T_error'],
                       'residual_better_rays': sum(b['max_abs_T_error'] < a['max_abs_T_error']
                                                  for a, b in zip(d['per_ray'], r['per_ray'])), 'rays': len(profiles)})
    report = {'config': config, 'data_audit': read_json(output/'data_audit.json'),
              'quadrature_audit': read_json(output/'quadrature_audit.json'),
              'baseline_metrics': {s: {k: v for k, v in result.items() if k != 'per_ray'} for s, result in baseline.items()},
              'runs': runs, 'historical_reference': historical,
              'summary': summarize_runs(runs, baseline['test'], historical), 'paired_comparison': paired, 'curves': records}
    write_json(output/'analysis.json', report)
    csv_rows = []
    for r in records:
        for name in ('local', *predictions):
            csv_rows.append({'pixel_x': r['pixel'][0], 'pixel_y': r['pixel'][1], 'method': name,
                             **{key: r[name][key] for key in ('max_abs_T_error', 'mse_T', 'escape_error', 'distance_nll')}})
    write_csv(output/'per_ray_metrics.csv', csv_rows)
    return report


@torch.no_grad()
def benchmark(config, data, output):
    profiles = [p for p in data['profiles'] if p['split'] == 'test']
    prepared = prepare({'profiles': profiles}, config, 'cpu')['test']
    seed = config['seeds'][0]  # Prescribed before inspecting any test result.
    models = {mode: load_hazard(output/f'{mode}_{seed}'/'best.pt')[0] for mode in MODES}
    models['spline_reference'], _ = load_model(resolve(config['spline_checkpoint']))
    methods = ('local', *MODES, 'spline_reference')
    rng = np.random.default_rng(20261018)
    randoms = rng.uniform(1e-6, 1-1e-6, (len(profiles), 2))
    signs, uniforms, branches = [], [], []
    for p, pair in zip(profiles, randoms):
        branch = int(pair[0] >= float(torch.special.ndtr(p['features'][0, 0])))
        branches.append(branch)
        signs.append(torch.tensor([1. if branch == 0 else -1.], dtype=torch.float64))
        uniforms.append(torch.tensor([[pair[1]]], dtype=torch.float64))

    def sample(method, i, cached):
        p, cache = profiles[i], prepared[i]
        branch = branches[i]
        if method == 'local':
            rates = (cache['baseline'] if cached else local_bin_log_rates(
                p['features'], config['model']['bins'], config['model']['quadrature_order']))[branch:branch+1]
        else:
            model = models[method]
            fixed = cache['fixed'] if cached else resample_profile(p['features'], config['model']['profile_points'])
            if method == 'spline_reference':
                _, raw = model.encode(fixed[None], signs[i])
                x, y, derivative, _, log_hit = spline_parameters(raw, model.config)
                hit_probability = log_hit.exp()
                if uniforms[i][0, 0] >= hit_probability[0]:
                    return float(cache['horizon']), False
                q, _, _ = rational_quadratic(uniforms[i]/hit_probability[:, None],
                                              x[:, None], y[:, None], derivative[:, None], inverse=True)
                return float(q[0, 0]*cache['horizon']), True
            baseline = None
            if method == 'residual':
                baseline = (cache['baseline'] if cached else local_bin_log_rates(
                    p['features'], config['model']['bins'], config['model']['quadrature_order']))[branch:branch+1]
            rates = model.log_rates(fixed[None], signs[i], baseline)
        q, hit = inverse_piecewise(rates, cache['horizon'], uniforms[i])
        return float(q[0, 0]), bool(hit[0, 0])

    cases = [(method, cached) for cached in (True, False) for method in methods]
    for method, cached in cases:
        for i in range(len(profiles)):
            sample(method, i, cached)
    times = {f'{method}_{"prepared" if cached else "from_cubics"}': [] for method, cached in cases}
    checksums = {name: [] for name in times}
    for repeat in range(config['benchmark_repeats']):
        # Rotate order to reduce warm-up / temperature / ordering bias.
        for method, cached in cases[repeat % len(cases):]+cases[:repeat % len(cases)]:
            name = f'{method}_{"prepared" if cached else "from_cubics"}'
            checksum = 0.
            start = time.perf_counter()
            for _ in range(config['benchmark_inner_repeats']):
                for i in range(len(profiles)):
                    q, hit = sample(method, i, cached)
                    checksum += q+int(hit)
            seconds = time.perf_counter()-start
            times[name].append(seconds*1e6/(len(profiles)*config['benchmark_inner_repeats']))
            checksums[name].append(checksum)
    if any(max(values)-min(values) > 1e-9 for values in checksums.values()):
        raise AssertionError('benchmark changed its deterministic samples')
    for method in methods:
        if abs(checksums[f'{method}_prepared'][0]-checksums[f'{method}_from_cubics'][0]) > 1e-9:
            raise AssertionError('cached and constructed inputs sampled different distributions')
    result = {'unit': 'microseconds per ray', 'threads': torch.get_num_threads(), 'seed': seed,
              'torch': str(torch.__version__), 'platform': platform.platform(),
              'processor': os.environ.get('PROCESSOR_IDENTIFIER', platform.processor()),
              'scope': 'Python CPU, preselected entry sign and pre-generated uniform, one distance sample; no gradient head, VDB access, or rendering',
              'prepared': 'fixed inputs and local bin rates cached; includes network, prefix sum, and inverse sampling',
              'from_cubics': 'starts with the existing cubic mean profile; constructs fixed inputs and/or quadrature baseline as required',
              'local_branches': 'both local branches constructed together; only selected sign is passed through the network',
              'measurement_note': 'Wall-clock microbenchmark on a shared Windows workstation. Raw ranges are retained; small timing differences are not conclusive.',
              'training_gpu': torch.cuda.get_device_name() if torch.cuda.is_available() else None,
              'timings': {name: {'median_us': statistics.median(values), 'minimum_us': min(values),
                                  'maximum_us': max(values), 'runs_us': values} for name, values in times.items()}}
    write_json(output/'benchmark.json', result)
    return result


@torch.no_grad()
def validate_samples(config, report, output):
    seed = config['seeds'][0]
    names = ('local', f'direct_{seed}', f'residual_{seed}')
    data = load_data(resolve(config['dataset']))
    profiles = {p['id']: p for p in data['profiles']}
    generator = torch.Generator().manual_seed(20261019)
    samples = 8192
    records = []
    for row in report['curves']:
        p = profiles[row['profile']]
        prior = torch.special.ndtr(p['features'][0, 0])
        branch = torch.rand(samples, dtype=torch.float64, generator=generator) >= prior
        uniform = torch.rand(samples, dtype=torch.float64, generator=generator)
        q_grid = torch.tensor(row['q'], dtype=torch.float64)
        for name in names:
            rates = torch.tensor(row[name]['log_rates'], dtype=torch.float64)
            draws = torch.empty(samples, dtype=torch.float64)
            hit = torch.zeros(samples, dtype=torch.bool)
            for k in (0, 1):
                chosen = branch == bool(k)
                if chosen.any():
                    q, event = inverse_piecewise(rates[k:k+1], p['horizon'], uniform[chosen][None])
                    draws[chosen], hit[chosen] = q[0], event[0]
            empirical = 1-torch.searchsorted(draws[hit].sort().values, q_grid, right=True).double()/samples
            truth = torch.tensor(row[name]['T'], dtype=torch.float64)
            error = float((empirical-truth).abs().max())
            if not torch.isfinite(draws).all() or (draws < 0).any() or (draws > p['horizon']+1e-12).any():
                raise FloatingPointError('invalid trained-model inverse sample')
            records.append({'pixel': row['pixel'], 'method': name, 'samples': samples, 'max_CDF_error': error})
    # DKW + union bound: 99% simultaneous bound across every checked ray/model.
    threshold = math.sqrt(math.log(2*len(records)/.01)/(2*samples))
    maximum = max(r['max_CDF_error'] for r in records)
    result = {'maximum_CDF_error': maximum, 'simultaneous_99_percent_DKW_bound': threshold,
              'passed': maximum <= threshold, 'records': records}
    write_json(output/'sampling_validation.json', result)
    if not result['passed']:
        raise AssertionError('inverse sampler exceeded the simultaneous empirical-CDF bound')
    return result


def make_report(config, output):
    report = read_json(output/'analysis.json')
    timing = read_json(output/'benchmark.json')
    sampling = (read_json(output/'sampling_validation.json') if (output/'sampling_validation.json').exists()
                else validate_samples(config, report, output))
    if not sampling['passed']:
        raise AssertionError('sample validation has not passed')
    seed = config['seeds'][0]
    labels = {'local': 'Local GP', 'direct': 'Direct hazard', 'residual': 'Local + residual',
              'spline_reference': 'Existing spline (reference)'}
    colors = {'local': '#2563eb', 'direct': '#e87920', 'residual': '#059669', 'spline_reference': '#8b5cf6'}
    plt = plotting()

    def plot_curves(rows, columns, name):
        height = math.ceil(len(rows)/columns)
        fig, axes = plt.subplots(height, columns, figsize=(5.1*columns, 3.15*height), squeeze=False)
        for ax, row in zip(axes.flat, rows):
            x = row['distance']
            ax.fill_between(x, row['reference_low'], row['reference_high'], color='#64748b', alpha=.22,
                            label='GPIS MC pointwise 95% CI')
            ax.plot(x, row['reference'], color='#18212f', lw=1.6, label='GPIS MC')
            ax.plot(x, row['local']['T'], color=colors['local'], lw=1.2, label=labels['local'])
            for mode in MODES:
                predictions = np.asarray([row[f'{mode}_{s}']['T'] for s in config['seeds']])
                ax.fill_between(x, predictions.min(0), predictions.max(0), color=colors[mode], alpha=.12)
                ax.plot(x, row[f'{mode}_{seed}']['T'], color=colors[mode], lw=1.5, label=labels[mode])
            ax.plot(x, row['spline']['T'], color=colors['spline_reference'], lw=1., ls='--', label='Existing spline')
            errors = [row['local']['max_abs_T_error'], row[f'direct_{seed}']['max_abs_T_error'],
                      row[f'residual_{seed}']['max_abs_T_error']]
            ax.set(title=f"Pixel {tuple(row['pixel'])} | max error L/D/R: {errors[0]:.3f}/{errors[1]:.3f}/{errors[2]:.3f}",
                   xlabel='Distance from medium entry (world units)', ylabel='Transmittance', ylim=(-.025, 1.025))
            ax.tick_params(labelsize=8)
            ax.title.set_fontsize(9)
        for ax in list(axes.flat)[len(rows):]:
            ax.set_visible(False)
        handles, legend_labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, legend_labels, loc='upper center', ncol=3, fontsize=9, bbox_to_anchor=(.5, 1.005))
        fig.suptitle(f'Independent test rays | lines: seed {seed}; colored bands: range across 3 seeds', y=1.045)
        fig.tight_layout(rect=(0, 0, 1, .96))
        save_figure(plt, fig, output, name)

    wanted = {(180, 279), (361, 332), (368, 256), (247, 268), (404, 294), (256, 256)}
    plot_curves([r for r in report['curves'] if tuple(r['pixel']) in wanted], 2, 'transmittance_examples')
    plot_curves(report['curves'], 4, 'transmittance_all')

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    shown = ('local', 'direct', 'residual', 'spline_reference')
    for ax, key, title in zip(axes, ('mean_ray_max_abs_T_error', 'rmse_T', 'distance_nll'),
                              ('Mean of per-ray maximum T errors', 'Transmittance RMSE', 'Marginal distance NLL')):
        mean = [report['summary'][method][key]['mean'] for method in shown]
        std = [report['summary'][method][key]['std'] for method in shown]
        ax.bar(np.arange(4), mean, yerr=std, capsize=4, color=[colors[m] for m in shown], alpha=.85)
        ax.set_xticks(np.arange(4), ['Local', 'Direct', 'Residual', 'Spline ref.'])
        ax.set_title(title)
        for i, value in enumerate(mean):
            ax.text(i, value+std[i], f'{value:.4f}', ha='center', va='bottom', fontsize=9)
        ax.margins(y=.18)
    fig.suptitle('24 held-out rays | neural bars: mean +/- sample SD across 3 training seeds')
    fig.tight_layout()
    save_figure(plt, fig, output, 'metrics')

    fig, axes = plt.subplots(1, len(config['seeds']), figsize=(15, 4.1), squeeze=False)
    baseline_val = report['baseline_metrics']['validation']['distance_nll']
    for ax, s in zip(axes.flat, config['seeds']):
        ax.axhline(baseline_val, color=colors['local'], ls='--', label='Local validation NLL')
        for mode in MODES:
            history = read_json(output/f'{mode}_{s}'/'history.json')
            run_report = next(r for r in report['runs'] if r['mode'] == mode and r['seed'] == s)
            selected = next(r for r in history if r['epoch'] == run_report['selected_epoch'])
            ax.plot([r['epoch'] for r in history], [r['validation']['distance_nll'] for r in history],
                    color=colors[mode], label=labels[mode])
            ax.scatter([selected['epoch']], [selected['validation']['distance_nll']], color=colors[mode], s=40, zorder=5)
        ax.set(title=f'Seed {s}', xlabel='Epoch', ylabel='Validation distance NLL')
    axes.flat[0].legend(fontsize=8)
    fig.suptitle('Identical 500-epoch budgets | dots: checkpoints selected only by validation NLL')
    fig.tight_layout()
    save_figure(plt, fig, output, 'training')

    fig, ax = plt.subplots(figsize=(14, 5))
    at = np.arange(len(report['curves']))
    for offset, mode in zip((-.25, 0, .25), ('local', *MODES)):
        if mode == 'local':
            values = np.asarray([[r['local']['max_abs_T_error'] for r in report['curves']]])
        else:
            values = np.asarray([[r[f'{mode}_{s}']['max_abs_T_error'] for r in report['curves']] for s in config['seeds']])
        ax.bar(at+offset, values.mean(0), width=.24, color=colors[mode], label=labels[mode],
               yerr=values.std(0, ddof=1) if len(values) > 1 else None, capsize=2)
    ax.set_xticks(at, [str(tuple(r['pixel'])) for r in report['curves']], rotation=60, ha='right', fontsize=8)
    ax.set(ylabel='Maximum absolute T error along the ray', title='Every held-out ray | mean +/- SD across training seeds')
    ax.legend()
    fig.tight_layout()
    save_figure(plt, fig, output, 'per_ray_errors')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, scope, title in zip(axes, ('prepared', 'from_cubics'), ('Prepared inputs', 'Including preparation from cubic mean profile')):
        values = [timing['timings'][f'{m}_{scope}']['median_us'] for m in shown]
        ax.bar(np.arange(4), values, color=[colors[m] for m in shown], alpha=.85)
        for i, value in enumerate(values):
            ax.text(i, value, f'{value:.1f}', ha='center', va='bottom')
        ax.set_xticks(np.arange(4), ['Local', 'Direct', 'Residual', 'Spline ref.'])
        ax.set(title=title, ylabel='Microseconds / one distance sample')
        ax.margins(y=.16)
    fig.suptitle('Python CPU, 1 thread | no VDB lookup, gradient head, or full rendering')
    fig.tight_layout()
    save_figure(plt, fig, output, 'timing')

    def number(method, key):
        value = report['summary'][method][key]
        return f"{value['mean']:.5f}"+(f" ± {value['std']:.5f}" if method in MODES else '')
    table_rows = ''.join('<tr><td>'+html.escape(labels[method])+'</td>'+''.join(
        f'<td>{number(method, key)}</td>' for key in METRICS)+'</tr>' for method in shown)
    benchmark_rows = ''.join(f'<tr><td>{html.escape(labels[method])}</td>'+
        ''.join(f'<td>{timing["timings"][f"{method}_{scope}"]["median_us"]:.1f}</td>' for scope in ('prepared', 'from_cubics'))
        +'</tr>' for method in shown)
    run_rows = ''.join(f'<tr><td>{r["mode"]}</td><td>{r["seed"]}</td><td>{r["selected_epoch"]}</td>'
                       f'<td>{r["training_seconds"]:.2f}</td><td>{r["metrics"]["test"]["max_abs_T_error"]:.5f}</td>'
                       f'<td><a href="{r["mode"]}_{r["seed"]}/run_report.json">metrics</a> · '
                       f'<a href="{r["mode"]}_{r["seed"]}/best.pt">checkpoint</a></td></tr>' for r in report['runs'])
    paired_text = '；'.join(f"seed {p['seed']}：{p['residual_better_rays']}/{p['rays']} 条射线改善" for p in report['paired_comparison'])
    summary = report['summary']
    improvement_direct = 1-summary['residual']['mean_ray_max_abs_T_error']['mean']/summary['direct']['mean_ray_max_abs_T_error']['mean']
    improvement_local = 1-summary['residual']['mean_ray_max_abs_T_error']['mean']/summary['local']['mean_ray_max_abs_T_error']['mean']
    numerical = report['quadrature_audit']
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>First-hit local / direct / residual comparison</title>
<style>body{{font:16px/1.65 system-ui,"Microsoft YaHei",sans-serif;color:#18212f;background:#f5f7fb;margin:0}}
main{{max-width:1250px;margin:32px auto;padding:32px;background:white;border-radius:14px}}h1{{font-size:28px}}h2{{margin-top:36px}}
table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:14px}}td,th{{padding:10px;border-bottom:1px solid #dce2eb;text-align:right}}
td:first-child,th:first-child{{text-align:left}}img{{width:100%;height:auto}}a{{color:#1d4ed8}}code,pre{{background:#edf1f7;border-radius:4px}}
pre{{padding:15px;overflow:auto}}.callout{{padding:18px;border-left:4px solid #059669;background:#ecfdf5}}.scroll{{overflow:auto}}.note{{color:#475569}}</style>
<main><h1>首碰距离：local、直接消光率、local + 残差</h1>
<p class="callout">在同一组独立测试射线上，残差方案的逐射线最大透射率误差均值，相比直接网络降低
<strong>{improvement_direct:.1%}</strong>，相比纯 local 降低 <strong>{improvement_local:.1%}</strong>。
这是三个配对训练种子的结果；下面同时列出速度成本。</p>
<p>复用已有 sigma=0.05、SE 核、相机首碰 Tungsten-linear 数据。训练 / 验证 / 测试为 191 / 32 / 24 条独立射线，
测试每条 8192 个样本。距离始终从介质入口起算。两个网络的参数量均为 {report['runs'][0]['parameters']:,}，
输入 256 点剖面、两层宽度 64 的编码器、32 维潜变量和 64 个输出。
输出插值到 {config['model']['bins']} 个消光率区间；1024 是积分/采样网格，网络仍然只有 64 个输出。</p>
<p><code>direct: log λ = rθ</code>；<code>residual: log λ = log λlocal + rθ</code>。
两个网络的初始参数、逐 epoch 抽样、学习率计划和距离损失相同；均训练 500 epochs，仅按验证集边际距离 NLL 选择 checkpoint。
训练使用已观测入口符号的条件距离似然，命中损失 H−log λ，逃逸损失 H。未加入梯度头、CDF 辅助损失或仅针对残差的额外正则。</p>
<p class="note">Local 基线使用原始 GP 均值梯度，并非渲染配置中会归一化均值梯度的 local_tangent。
连续 GP local 基线与有限历史 Tungsten-linear 标签也有离散化/采样器差异，学习到的修正不应全部解释成精确 GP 的历史相关性。
现有 spline 是此前联合训练距离与梯度的结果，仅作历史参考。</p>
<h2>独立测试集精度</h2><p>所有指标越小越好。神经网络数值为三个种子的均值 ± 样本标准差；“最大误差”也是每次训练最大误差的均值，非集合模型。</p>
<div class="scroll"><table><tr><th>方案</th><th>距离 NLL</th><th>最大 |ΔT|</th><th>逐射线最大 |ΔT| 均值</th><th>T RMSE</th><th>逃逸误差均值</th><th>逃逸最大误差</th></tr>{table_rows}</table></div>
<p>{paired_text}。胜负按每条射线的最大 |ΔT| 判断。</p><img src="metrics.png" alt="Test error comparison">
<h2>透射率曲线</h2><p>黑线为 GPIS 经验透射率，灰带为逐点 95% Monte Carlo 区间。神经网络实线固定使用 seed {seed}，彩色带为三个训练种子的范围；没有根据测试结果挑选种子。</p>
<a href="transmittance_examples.svg"><img src="transmittance_examples.png" alt="Six representative test rays"></a>
<details><summary>查看全部 24 条测试射线</summary><a href="transmittance_all.svg"><img src="transmittance_all.png" alt="All held-out rays"></a></details>
<img src="per_ray_errors.png" alt="Per-ray errors">
<h2>CPU 距离采样耗时</h2><p>单位 μs/射线，{config['benchmark_repeats']} 次轮换顺序测量的中位数，PyTorch CPU 单线程。
入口符号与均匀随机数预先给定，每次只采样一个距离。“已准备输入”缓存固定剖面和 local 系数；“从三次剖面开始”包含相应的剖面/基线构建。
不含 VDB 查询、梯度头和后续路径，<strong>不能等同于 C++ 渲染时间</strong>。工作站上的计时波动较大，原始范围保存在 JSON 中，小差异不作结论。</p>
<table><tr><th>方案</th><th>已准备输入 μs</th><th>从三次剖面开始 μs</th></tr>{benchmark_rows}</table>
<img src="timing.png" alt="CPU prototype timing">
<p class="note">当前基线构建采用逐个插值单元分裂后的 Gauss-Legendre 积分，并同时计算正负两支。
这有利于本次精度归因，但仍有优化空间，不能据此承诺残差渲染一定更快。</p>
<h2>训练与校验</h2><img src="training.png" alt="Paired validation training curves">
<table><tr><th>方案</th><th>种子</th><th>选中 epoch</th><th>完整训练秒数</th><th>测试最大 |ΔT|</th><th>文件</th></tr>{run_rows}</table>
<p>积分在 VDB 三次剖面边界处分裂，避免跨越导数跳变。只用验证几何检查数值分辨率；相对于 2048 格、16 点积分，
1024 格的最大 T 偏差为 {numerical['max_T_grid_error']:.7f}；固定网格上 8 点与 32 点积分的最大差为 {numerical['max_T_quadrature_error']:.3g}。
训练后对 24 条射线 × 3 种方案各生成 8192 个样本，经验 CDF 最大差 {sampling['maximum_CDF_error']:.5f}，
通过覆盖全部检查的 99% DKW 界 {sampling['simultaneous_99_percent_DKW_bound']:.5f}。</p>
<p>这些 checkpoint 仅用于距离分布实验，尚未包含梯度头或接入 C++ 渲染器。</p>
<h2>复现与数据</h2><pre>python python/first_hit_se_cli.py hazard-study --config python/configs/first_hit_se_hazard_comparison_sigma005.json</pre>
<p><a href="analysis.json">完整精度与曲线 JSON</a> · <a href="per_ray_metrics.csv">逐射线 CSV</a> · <a href="benchmark.json">原始计时</a> ·
<a href="data_audit.json">数据来源</a> · <a href="quadrature_audit.json">积分检查</a> · <a href="sampling_validation.json">采样检查</a> ·
<a href="study_config.json">实验配置</a></p>
<p class="note">数据 SHA-256：<code>{report['data_audit']['dataset_sha256']}</code></p></main></html>'''
    (output/'summary.html').write_text(page, encoding='utf-8')
    return {'summary': str(output/'summary.html'), 'accuracy': report['summary'],
            'sampling_validation_passed': sampling['passed']}


def run(config_path, phase='all'):
    config = read_json(config_path)
    output = resolve(config['output_directory'])
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(config.get('cpu_threads', 1))
    torch.backends.cuda.matmul.allow_tf32 = False
    if (output/'study_config.json').exists() and read_json(output/'study_config.json') != config:
        raise ValueError('use a new output directory when changing the study configuration')
    if phase == 'report':
        return make_report(config, output)
    data = load_data(resolve(config['dataset']))
    if phase == 'benchmark':
        return benchmark(config, data, output)
    audit = audit_data(data, config)
    write_json(output/'data_audit.json', audit)
    numerical = audit_quadrature(data, config)
    write_json(output/'quadrature_audit.json', numerical)
    print(f"local baseline audit: grid T error {numerical['max_T_grid_error']:.7f}, "
          f"quadrature T error {numerical['max_T_quadrature_error']:.7f}", flush=True)
    if not numerical['passed']:
        raise ArithmeticError('local baseline grid failed the prescribed numerical tolerance')
    write_json(output/'study_config.json', config)
    splits = prepare(data, config, config['device'])
    if phase in ('all', 'train'):
        # Same seed has identical weights, minibatches, and fixed LR schedule.
        for seed in config['seeds']:
            for mode in MODES:
                train_one(config, audit['dataset_sha256'], splits, mode, seed, output)
    if phase in ('all', 'evaluate'):
        evaluate_study(config, data, output, splits)
    if phase == 'all':
        benchmark(config, data, output)
        make_report(config, output)
    return {'output_directory': str(output), 'phase': phase,
            'models': [f'{mode}_{seed}' for seed in config['seeds'] for mode in MODES]}
