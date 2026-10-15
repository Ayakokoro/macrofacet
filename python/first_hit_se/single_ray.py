"""Single-ray diagnostic: independent GP realizations, identical h16/K8 models.

This deliberately splits realizations of ONE physical ray, unlike the ordinary
ray-held-out dataset. It is not evidence for generalization to other rays.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import argparse
import hashlib
import math
import os
import statistics
import subprocess
import sys

import numpy as np
import torch

from renewal.collect import digest, resolve, write_json
from . import FORMAT, VERSION, EVENT, SE_KERNEL
from .audit import audit
from .data import load_data, read_json, rows
from .data_ablation import immutable_json
from .gradient_head import cache_profiles, verify_frozen_predictions
from .train import load_model


SPLITS = ('train', 'validation', 'test')


def bin_edges(training_profile, count):
    """Choose bins from TRAIN hits only; include both tails of the full ray."""
    q = training_profile['distance'][training_profile['hit']].double()
    if len(q) < count or count < 2:
        raise ValueError('insufficient training hits for distance bins')
    interior = torch.quantile(q, torch.arange(1, count, dtype=torch.float64)/count)
    edges = torch.cat((q.new_zeros(1), interior, q.new_tensor([training_profile['horizon']])))
    if not torch.all(edges[1:] > edges[:-1]):
        raise ValueError('distance bins must be strictly increasing')
    return edges


def bin_indices(distance, edges):
    if (distance < edges[0]).any() or (distance > edges[-1]).any():
        raise ValueError('distance outside the ray')
    return torch.bucketize(distance.contiguous(), edges[1:-1], right=True)


def validate_realization_sets(seed_sets):
    names = list(seed_sets)
    for i, name in enumerate(names):
        for other in names[:i]:
            if seed_sets[name] & seed_sets[other]:
                raise ValueError(f'GP realization leakage: {name} and {other}')


def collect(config_path):
    spec = read_json(config_path)
    root = resolve(spec['output_directory'])
    root.mkdir(parents=True, exist_ok=True)
    immutable_json(root/'specification.json', spec)
    jobs = []
    for split, settings in spec['splits'].items():
        folder = root/'collection'/split
        folder.mkdir(parents=True, exist_ok=True)
        request = {'seed': settings['seed'], 'output_directory': str(folder),
                   'split_unit': 'ray', 'realizations_per_profile': settings['realizations'],
                   'reference_steps': spec['reference_steps'], 'threads': spec['threads_per_reference'],
                   'sources': [{'scene_id': 'single_ray_sigma005', 'split': split,
                                'render_config': spec['render_config'], 'pixel': spec['pixel']}]}
        config = folder/'collection_config.json'
        immutable_json(config, request)
        jobs.append((split, folder, config))

    def generate(job):
        split, folder, config = job
        with (folder/'collection.log').open('w', encoding='utf-8') as stream:
            subprocess.run([sys.executable, str(resolve('python/first_hit_se_cli.py')), 'collect',
                            '--config', str(config)], cwd=resolve('.'), stdout=stream,
                           stderr=subprocess.STDOUT, check=True)
        return split

    with ThreadPoolExecutor(max_workers=spec['collection_workers']) as executor:
        futures = [executor.submit(generate, job) for job in jobs]
        for future in as_completed(futures):
            print('COLLECTED', future.result(), flush=True)

    profiles, identities, seeds, audit_sources = [], {}, {}, []
    previous = {r['seed'] for r in rows(resolve(spec['previous_reference']))}
    for split in SPLITS:
        folder = root/'collection'/split
        records = load_data(folder/'dataset.pt')['profiles']
        if len(records) != 1:
            raise ValueError('single-ray collection must contain exactly one profile')
        p = records[0]
        p['id'] = f'single_ray_{spec["pixel"][0]}_{spec["pixel"][1]}:{split}'
        if profiles:
            torch.testing.assert_close(p['features'], profiles[0]['features'], atol=0, rtol=0)
            if p['geometry'] != profiles[0]['geometry']:
                raise ValueError('the physical ray changed across realizations')
        labels = rows(Path(p['source'])/'first_passage_samples.csv')
        all_seeds = [r['seed'] for r in labels]
        if len(set(all_seeds)) != len(all_seeds) or previous.intersection(all_seeds):
            raise ValueError('duplicate GP seed or overlap with the previous reference')
        seeds[split] = set(all_seeds)
        fine = [r['seed'] for r in labels if r['training_resolution'] == '1']
        if len(fine) != spec['splits'][split]['realizations']:
            raise ValueError('incomplete GP realization split')
        identities[split] = {'samples': len(p['hit']), 'hits': int(p['hit'].sum()),
            'negative_entries': int((p['initial_sign'] < 0).sum()),
            'reference_samples_sha256': p['samples_sha256'],
            'finest_seeds_sha256': hashlib.sha256('\n'.join(fine).encode()).hexdigest(),
            'unique_seeds_all_resolutions': len(all_seeds)}
        profiles.append(p)
        audit_sources.extend(read_json(folder/'manifest.json')['sources'])
    validate_realization_sets(seeds)
    edges = bin_edges(profiles[0], spec['distance_bins'])
    for p in profiles:
        labels = bin_indices(p['distance'][p['hit']], edges)
        identities[p['split']]['hits_per_distance_bin'] = torch.bincount(
            labels, minlength=spec['distance_bins']).tolist()
    header = {'format': FORMAT, 'version': VERSION, 'event': EVENT, 'kernel': SE_KERNEL,
              'profile_mode': 'cubic', 'split_unit': 'realization',
              'scope': 'Single-ray specialization; no held-out-ray generalization claim.'}
    if not (root/'dataset.pt').exists():
        torch.save({**header, 'profiles': profiles}, root/'dataset.pt')
    else:
        old = load_data(root/'dataset.pt')['profiles']
        for a, b in zip(old, profiles):
            for key in ('features', 'distance', 'hit', 'initial_sign', 'speed'):
                torch.testing.assert_close(a[key], b[key], atol=0, rtol=0, equal_nan=True)
    manifest = {'sources': audit_sources}
    write_json(root/'reference_manifest.json', manifest)
    audit(root/'reference_manifest.json', root/'reference_audit.json')
    report = {'pixel': spec['pixel'], 'split_unit': 'realization', 'splits': identities,
              'distance_bin_edges_q': edges.tolist(), 'ell': profiles[0]['geometry']['ell'],
              'sigma': profiles[0]['geometry']['sigma'], 'horizon_q': profiles[0]['horizon'],
              'disjoint_seeds': True, 'same_physical_ray': True,
              'source_checkpoint_sha256': digest(resolve(spec['initialize_from'])),
              'dataset_sha256': digest(root/'dataset.pt'),
              'event_definition': EVENT,
              'bin_definition': 'Training first-hit distance quantiles; bins are intervals, not exact fixed-q conditioning.'}
    write_json(root/'data_audit.json', report)
    print('DATA', {k: (v['samples'], v['hits']) for k, v in identities.items()}, flush=True)
    return report


def prepare_training(config_path):
    from .joint_state import prepare
    spec = read_json(config_path)
    root = resolve(spec['output_directory'])
    torch.set_num_threads(spec['cpu_threads'])
    source, _ = load_model(resolve(spec['initialize_from']), spec['device'])
    data = load_data(root/'dataset.pt')['profiles']
    identity = {'dataset_sha256': digest(root/'dataset.pt'),
                'checkpoint_sha256': digest(resolve(spec['initialize_from']))}
    training_file, evaluation_file = root/'training_contexts.pt', root/'evaluation_contexts.pt'
    if not training_file.exists() or not evaluation_file.exists():
        contexts = {split: cache_profiles(source, [p for p in data if p['split'] == split],
                                         batch_size=1) for split in SPLITS}
        torch.save({'identity': identity, 'train': contexts['train']}, training_file)
        torch.save({'identity': {**identity, 'scene_dataset_sha256': identity['dataset_sha256']},
                    'splits': contexts}, evaluation_file)
    settings = {'output_directory': str(root/'training'), 'model_directory': spec['model_directory'],
                'dataset': str(root/'dataset.pt'), 'evaluation_dataset': str(root/'dataset.pt'),
                'scene_dataset': str(root/'dataset.pt'), 'initialize_from': spec['initialize_from'],
                'training_contexts': str(training_file), 'evaluation_contexts': str(evaluation_file),
                'device': spec['device'], 'cpu_threads': spec['cpu_threads'], 'seeds': spec['seeds'],
                'representative_seed': spec['representative_seed'], 'modes': ['head_only', 'joint'],
                'training': spec['training']}
    path = root/'training_config.json'
    immutable_json(path, settings)
    prepare(path)
    return path


def train(config_path):
    path = prepare_training(config_path)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    spec = read_json(config_path)
    root = resolve(spec['output_directory'])
    (root/'logs').mkdir(exist_ok=True)
    for seed in spec['seeds']:
        for mode in ('head_only', 'joint'):
            name = f'{mode}_s{seed}'
            print('TRAIN', name, flush=True)
            with (root/'logs'/f'{name}.log').open('w', encoding='utf-8') as stream:
                subprocess.run([sys.executable, str(resolve('python/first_hit_se_cli.py')),
                                'joint-state-train', '--config', str(path), '--name', name],
                               cwd=resolve('.'), stdout=stream, stderr=subprocess.STDOUT, check=True)
            print('TRAINED', name, flush=True)
    return {'completed_runs': 2*len(spec['seeds'])}


def ks_uniform(values):
    values = np.sort(np.asarray(values, dtype=float))
    if not len(values):
        return None
    ranks = np.arange(1, len(values)+1)/len(values)
    return float(max(np.max(ranks-values), np.max(values-ranks+1/len(values))))


@torch.no_grad()
def averaged_cdf(mixture, grid, chunk_size=512):
    """Average conditional CDFs at the SAME reference event locations."""
    from renewal.model import mixture_cdf
    count = len(mixture[0])
    if not count:
        raise ValueError('CDF averaging requires reference events')
    total = torch.zeros(len(grid), dtype=torch.float64, device=mixture[0].device)
    for begin in range(0, count, chunk_size):
        parameters = [v[begin:begin+chunk_size, None, :] for v in mixture]
        total += mixture_cdf(*parameters, grid[None, :]).sum(0)
    return (total/count).cpu().numpy()


@torch.no_grad()
def evaluate_model(model, profiles, edges, grid, array_path):
    from renewal.model import mixture_cdf, mixture_log_prob
    from .batch import query_many, make_plan
    from .model import losses_from_prediction
    device = next(model.parameters()).device
    result = {'splits': {}}
    for split in SPLITS:
        profile = next(p for p in profiles if p['split'] == split)
        plan = make_plan(profile['features'], model.config.maximum_step, device)
        p = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in profile.items()}
        prediction = query_many(model, [p['features']], [p['distance']], plans=[plan])[0]
        losses = losses_from_prediction(prediction, p['distance'], p['hit'], p['initial_sign'], p['speed'])
        observed = p['speed'][p['hit']]
        q = p['distance'][p['hit']]
        branches = (p['initial_sign'][p['hit']] < 0).long()
        rows_at_hits = p['hit'].nonzero().flatten()
        mixture = [prediction[key][branches, rows_at_hits] for key in ('log_weights', 'means', 'scales')]
        speed_nll = -mixture_log_prob(*mixture, observed).cpu().numpy()
        pit = mixture_cdf(*mixture, observed).cpu().numpy()
        if not np.isfinite(speed_nll).all() or not np.isfinite(pit).all():
            raise FloatingPointError('nonfinite single-ray likelihood or PIT')
        record = {'samples': len(p['hit']), 'hits': len(observed),
                  'speed_nll_on_hits': float(speed_nll.mean()), 'speed_pit_ks': ks_uniform(pit),
                  'distance_nll': float(losses['distance'].mean()),
                  'joint_nll': float(losses['joint'].mean())}
        if split == 'test':
            labels = bin_indices(q.cpu(), edges).numpy()
            w = observed.cpu().numpy()
            bins = []
            for i in range(len(edges)-1):
                mask = labels == i
                if mask.sum() < 64:
                    raise ValueError('test distance bin has too few independent hits')
                chosen = torch.from_numpy(mask).to(device)
                model_cdf = averaged_cdf([v[chosen] for v in mixture], grid.to(device))
                empirical = np.searchsorted(np.sort(w[mask]), grid.numpy(), side='right')/mask.sum()
                thresholds = [0.1, 0.25, 0.5, 2., 3., 4.]
                bins.append({'index': i, 'q_begin': float(edges[i]), 'q_end': float(edges[i+1]),
                    'mean_reference_q': float(q[chosen].mean()), 'hits': int(mask.sum()),
                    'speed_nll_on_hits': float(speed_nll[mask].mean()),
                    'speed_pit_ks': ks_uniform(pit[mask]),
                    'cdf_grid_max_abs_error': float(np.max(np.abs(model_cdf-empirical))),
                    'predicted_cdf': model_cdf.tolist(), 'empirical_cdf': empirical.tolist(),
                    'reference_cdf_dkw_95': math.sqrt(math.log(40)/(2*int(mask.sum()))),
                    'pit_uniform_dkw_95': math.sqrt(math.log(40)/(2*int(mask.sum()))),
                    'pit_uniform_dkw_95_simultaneous_8_bins': math.sqrt(math.log(40*(len(edges)-1))/(2*int(mask.sum()))),
                    'thresholds': [{'W': value, 'observed': float(np.mean(w[mask] <= value)),
                                    'predicted': float(model_cdf[int(torch.argmin((grid-value).abs()))])}
                                   for value in thresholds]})
            record['bins'] = bins
            record['mean_bin_pit_ks'] = statistics.mean(b['speed_pit_ks'] for b in bins)
            record['maximum_bin_pit_ks'] = max(b['speed_pit_ks'] for b in bins)
            record['mean_bin_cdf_max_error'] = statistics.mean(b['cdf_grid_max_abs_error'] for b in bins)
            record['maximum_bin_cdf_error'] = max(b['cdf_grid_max_abs_error'] for b in bins)
            record['global_cdf'] = averaged_cdf(mixture, grid.to(device)).tolist()
            record['global_empirical_cdf'] = (np.searchsorted(np.sort(w), grid.numpy(), side='right')/len(w)).tolist()
            np.savez_compressed(array_path, speed_nll=speed_nll, pit=pit, distance=q.cpu().numpy(),
                                speed=w, distance_bin=labels)
            locations = torch.linspace(0, profile['horizon'], 513, dtype=torch.float64)
            transmittance = query_many(model, [profile['features']], [locations], plans=[plan], with_mixture=False)[0]['T'].cpu()
            sorted_hits = profile['distance'][profile['hit']].sort().values
            empirical_T = 1-torch.searchsorted(sorted_hits, locations, right=True).double()/len(profile['hit'])
            if abs(float(transmittance[0])-1) > 1e-12 or (transmittance[1:] > transmittance[:-1]+1e-12).any():
                raise AssertionError('first-hit transmittance lost normalization or monotonicity')
            record['transmittance'] = {'q': locations.tolist(), 'predicted': transmittance.tolist(),
                'empirical': empirical_T.tolist(), 'maximum_absolute_error': float((transmittance-empirical_T).abs().max()),
                'reference_dkw_95': math.sqrt(math.log(40)/(2*len(profile['hit'])))}
        result['splits'][split] = record
    return result


def evaluate(config_path):
    from .verify_cpp import verify_cpp
    spec = read_json(config_path)
    root = resolve(spec['output_directory'])
    torch.set_num_threads(spec['cpu_threads'])
    data = load_data(root/'dataset.pt')['profiles']
    edges = bin_edges(next(p for p in data if p['split'] == 'train'), spec['distance_bins'])
    train_profile = next(p for p in data if p['split'] == 'train')
    upper = max(6., float(torch.quantile(train_profile['speed'][train_profile['hit']], .9995)))
    grid = torch.unique(torch.cat((torch.linspace(0, upper, 181, dtype=torch.float64),
                                  torch.tensor([.1, .25, .5, 2., 3., 4.], dtype=torch.float64))))
    folder = root/'evaluations'
    folder.mkdir(exist_ok=True)
    records = [('source', resolve(spec['initialize_from']))]
    for seed in spec['seeds']:
        for mode in ('head_only', 'joint'):
            for checkpoint in ('last', 'best'):
                name = f'{mode}_s{seed}'
                records.append((f'{name}_{checkpoint}', resolve(spec['model_directory'])/name/f'{checkpoint}.pt'))
    source, _ = load_model(resolve(spec['initialize_from']), spec['device'])
    model_results = {}
    for name, path in records:
        target = folder/f'{name}.json'
        identity = {'checkpoint_sha256': digest(path), 'dataset_sha256': digest(root/'dataset.pt'),
                    'implementation_sha256': digest(Path(__file__)), 'specification_sha256': digest(config_path)}
        if target.exists() and read_json(target).get('identity') == identity:
            model_results[name] = read_json(target)
            continue
        model, bundle = load_model(path, spec['device'])
        record = evaluate_model(model, data, edges, grid, folder/f'{name}.npz')
        record.update(name=name, checkpoint=str(path), identity=identity, epoch=bundle.get('epoch'),
                      model_config=model.configuration(), speed_cdf_grid=grid.tolist())
        if name.startswith('head_only'):
            record['frozen_predictions'] = verify_frozen_predictions(source, model, [data[-1]])
        write_json(target, record)
        model_results[name] = record
        print('EVALUATED', name, {key: record['splits']['test'][key]
                                for key in ('speed_nll_on_hits', 'speed_pit_ks', 'mean_bin_pit_ks')}, flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    for mode in ('head_only', 'joint'):
        name = f'{mode}_s{spec["representative_seed"]}'
        destination = root/'checks'/name
        if not (destination/'verification.json').exists():
            verify_cpp(resolve(spec['model_directory'])/name/'last.pt', root/'dataset.pt',
                       resolve('build/Release/macrofacet_query_first_hit_se.exe'), destination, 1, 512)
    summaries = {}
    metrics = ('speed_nll_on_hits', 'speed_pit_ks', 'mean_bin_pit_ks',
               'maximum_bin_pit_ks', 'mean_bin_cdf_max_error', 'maximum_bin_cdf_error')
    source_losses = np.load(folder/'source.npz')['speed_nll']
    for checkpoint in ('last', 'best'):
        summaries[checkpoint] = {}
        for mode in ('head_only', 'joint'):
            group = [model_results[f'{mode}_s{seed}_{checkpoint}']['splits']['test'] for seed in spec['seeds']]
            aggregate = {key: {'mean': statistics.mean(r[key] for r in group),
                               'standard_deviation': statistics.stdev(r[key] for r in group)} for key in metrics}
            aggregate['maximum_T_error'] = {'mean': statistics.mean(r['transmittance']['maximum_absolute_error'] for r in group),
                'standard_deviation': statistics.stdev(r['transmittance']['maximum_absolute_error'] for r in group)}
            paired = {}
            for seed in spec['seeds']:
                difference = np.load(folder/f'{mode}_s{seed}_{checkpoint}.npz')['speed_nll']-source_losses
                half = 1.95996398454*difference.std(ddof=1)/math.sqrt(len(difference))
                paired[str(seed)] = {'mean_nll_difference': float(difference.mean()),
                    'paired_mc_95_normal_interval': [float(difference.mean()-half), float(difference.mean()+half)]}
            aggregate['paired_test_nll_vs_source'] = paired
            summaries[checkpoint][mode] = aggregate
    runs = {f'{mode}_s{seed}': read_json(resolve(spec['model_directory'])/f'{mode}_s{seed}'/'run_report.json')
            for seed in spec['seeds'] for mode in ('head_only', 'joint')}
    paired_training = all(runs[f'head_only_s{seed}']['sample_trace'] == runs[f'joint_s{seed}']['sample_trace']
                          for seed in spec['seeds'])
    if not paired_training:
        raise AssertionError('paired training modes consumed different trajectory batches')
    unchanged = digest(resolve(spec['initialize_from'])) == read_json(root/'data_audit.json')['source_checkpoint_sha256']
    if not unchanged:
        raise AssertionError('the original checkpoint was modified')
    summary = {'specification': spec, 'source': model_results['source']['splits']['test'],
               'aggregate': summaries, 'primary_checkpoint': 'last.pt',
               'secondary_checkpoint': 'best.pt: minimum independent validation joint NLL',
               'paired_training_indices_identical': paired_training, 'source_checkpoint_unchanged': unchanged,
               'training_seconds': sum(r['training_seconds'] for r in runs.values()),
               'scope': 'One fixed ray, new GP realizations. Conditional speed at reference hits in training-defined distance intervals.',
               'uncertainty': 'Across-seed SD measures optimization only; paired NLL intervals measure test-realization sampling conditional on the trained models.',
               'limitations': ['No held-out-ray or scene generalization claim.',
                               'Distance bins are not exact fixed-endpoint first-hit conditioning.',
                               'Grid reference retains continuous-path discretization error.',
                               'CDF curves use reference hit locations/signs, not model-sampled collision locations.']}
    write_json(root/'summary.json', summary)
    return summary


def report(config_path):
    spec = read_json(config_path)
    root = resolve(spec['output_directory'])
    summary = read_json(root/'summary.json')
    audit_data = read_json(root/'data_audit.json')
    seed = spec['representative_seed']
    names = ['source', f'head_only_s{seed}_last', f'joint_s{seed}_last']
    records = {name: read_json(root/'evaluations'/f'{name}.json') for name in names}
    os.environ.setdefault('MPLCONFIGDIR', str(root/'matplotlib_cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'savefig.bbox': 'tight'})
    plots = root/'plots'
    plots.mkdir(exist_ok=True)
    colors = ['#94a3b8', '#0284c7', '#db2777']
    labels = ['Original RNO', 'Single-ray head only', 'Single-ray joint']

    def save(fig, name):
        fig.savefig(plots/f'{name}.svg')
        fig.savefig(plots/f'{name}.png', dpi=160)
        plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(15, 7.3), sharex=True, sharey=True)
    for i, ax in enumerate(axes.flat):
        reference = records['source']['splits']['test']['bins'][i]
        grid = np.array(records['source']['speed_cdf_grid'])
        observed = np.array(reference['empirical_cdf'])
        half = reference['reference_cdf_dkw_95']
        ax.fill_between(grid, np.maximum(0, observed-half), np.minimum(1, observed+half),
                        color='#cbd5e1', alpha=.6, linewidth=0)
        ax.plot(grid, observed, color='#0f172a', lw=1.8, label='Independent GP test')
        for name, label, color in zip(names, labels, colors):
            ax.plot(grid, records[name]['splits']['test']['bins'][i]['predicted_cdf'],
                    label=label, color=color, lw=1.4)
        ax.set_title(f'q in [{reference["q_begin"]:.3f}, {reference["q_end"]:.3f}) | n={reference["hits"]}')
        ax.grid(alpha=.15)
        ax.set(xlim=(0, min(5.5, grid[-1])), ylim=(0, 1.01), xlabel='Normalized crossing speed W', ylabel='CDF')
    axes.flat[0].legend(fontsize=8, loc='lower right')
    fig.suptitle('One ray, eight first-hit distance intervals | fixed representative seed 20261042', fontsize=14)
    fig.tight_layout()
    save(fig, 'distance_cdfs')

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    for name, label, color in zip(names, labels, colors):
        bins = records[name]['splits']['test']['bins']
        axes[0].plot(range(1, 9), [b['speed_pit_ks'] for b in bins], 'o-', label=label, color=color)
        axes[1].plot(range(1, 9), [b['thresholds'][2]['predicted']-b['thresholds'][2]['observed'] for b in bins],
                     'o-', label=label, color=color)
    axes[0].plot(range(1, 9), [b['pit_uniform_dkw_95_simultaneous_8_bins'] for b in bins],
                 '--', color='#0f172a', lw=1, label='95% simultaneous DKW threshold')
    axes[0].set(ylabel='Conditional PIT KS', xlabel='First-hit distance interval')
    axes[1].axhline(0, color='#0f172a', lw=.7)
    axes[1].set(ylabel='Predicted minus observed P(W <= 0.5)', xlabel='First-hit distance interval')
    for ax in axes:
        ax.grid(alpha=.15)
        ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, 'calibration')

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    ref = records['source']['splits']['test']['transmittance']
    q, empirical = np.array(ref['q']), np.array(ref['empirical'])
    axes[0].fill_between(q, np.clip(empirical-ref['reference_dkw_95'], 0, 1),
                         np.clip(empirical+ref['reference_dkw_95'], 0, 1), color='#cbd5e1', alpha=.6)
    axes[0].plot(q, empirical, color='#0f172a', label='Independent GP test')
    for name, label, color in zip(names, labels, colors):
        row = records[name]['splits']['test']
        axes[0].plot(q, row['transmittance']['predicted'], label=label, color=color)
        axes[1].plot(row['transmittance']['q'], np.array(row['transmittance']['predicted'])-empirical,
                     label=label, color=color)
    for ax in axes:
        ax.set_xlabel('Distance from medium entry q = s / ell')
        ax.grid(alpha=.15)
        ax.legend(fontsize=8)
    axes[0].set_ylabel('Transmittance')
    axes[1].set_ylabel('Predicted minus reference transmittance')
    axes[1].axhline(0, color='#0f172a', lw=.7)
    fig.tight_layout()
    save(fig, 'transmittance')

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    for mode, label, color in zip(('head_only', 'joint'), labels[1:], colors[1:]):
        for current_seed in spec['seeds']:
            history = read_json(resolve(spec['model_directory'])/f'{mode}_s{current_seed}'/'history.json')
            style = {'color': color, 'alpha': 1 if current_seed == seed else .35,
                     'label': label if current_seed == seed else None}
            axes[0].plot([r['updates'] for r in history],
                         [r['validation']['speed_nll_on_hits'] for r in history], **style)
            axes[1].plot([r['updates'] for r in history],
                         [r['validation']['joint_nll'] for r in history], **style)
    axes[0].axhline(records['source']['splits']['validation']['speed_nll_on_hits'], color=colors[0], ls='--', label='Original RNO')
    axes[1].axhline(records['source']['splits']['validation']['joint_nll'], color=colors[0], ls='--', label='Original RNO')
    for ax in axes:
        ax.set_xlabel('Optimizer updates')
        ax.grid(alpha=.15)
        ax.legend(fontsize=8)
    axes[0].set_ylabel('Independent validation gradient NLL')
    axes[1].set_ylabel('Independent validation joint NLL')
    fig.tight_layout()
    save(fig, 'learning')

    lines = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8">',
             '<title>单光线 first-hit RNO 拟合诊断</title>',
             '<style>body{font:16px/1.7 system-ui,sans-serif;max-width:1400px;margin:32px auto;padding:0 24px;color:#172033}'
             'h1,h2{line-height:1.3}table{width:100%;border-collapse:collapse;font-size:14px}'
             'th,td{padding:9px 12px;border-bottom:1px solid #dbe2ea;text-align:left}th{background:#edf2f7}'
             'img{width:100%;height:auto}.note{padding:16px;background:#eef6ff;border-left:4px solid #0284c7}'
             'a{color:#0369a1}code{font-size:13px}</style>',
             '<h1>一条光线，多个首碰距离：h16 / K8 RNO 拟合诊断</h1>',
             '<p>固定 shader_ball_full_0.05.nvdb，相机像素 (376, 96)，从 activeDomain 入口计距离。'
             '固定 SE；入口不条件化，取第一次过零，保留正负入口分支。原模型为上一轮联合训练的固定代表权重。</p>',
             '<p class="note">这是单光线专门拟合实验：训练、验证、测试共享同一条确定的均值曲线，'
             '使用完全独立的 GP 实现。它检验这条光线上的分布学习能力，不证明未见光线或跨场景泛化。'
             '每个实现只有一个首次碰撞标签；8 个距离区间由训练集首碰距离的八分位点决定。'
             '图中是有限距离区间条件分布，不是精确固定距离的条件 GP 桥。'
             '对一条固定光线，距离本身就能标识均值历史，因此拟合成功也不单独证明隐藏状态已学会可迁移的历史压缩。</p>',
             '<p><a href="summary.json">汇总数据</a> · <a href="data_audit.json">数据与独立性检查</a> · '
             '<a href="reference_audit.json">参考网格检查</a> · <a href="specification.json">实验设置</a></p>',
             '<h2>数据与训练</h2><table><tr><th>划分</th><th>GP 实现</th><th>首碰标签</th><th>每距离区间的命中数</th></tr>']
    for split in SPLITS:
        row = audit_data['splits'][split]
        lines.append(f'<tr><td>{split}</td><td>{row["samples"]:,}</td><td>{row["hits"]:,}</td><td>{row["hits_per_distance_bin"]}</td></tr>')
    lines += ['</table>',
              f'<p>参考网格目标步长为 ell/128 和 ell/256，训练/评估只用更细网格。三组种子各做梯度头微调和联合微调；'
              f'每组 {spec["training"]["epochs"]} 次更新，每次均匀抽取 2048 条训练轨迹，包含删失样本。'
              '配对模式使用完全相同的轨迹索引与完整联合 NLL。梯度头组冻结初始化、状态演化和 hazard；联合组更新全部模块。'
              '总参数量保持 39,545，梯度头组可训练 22,168。学习率 3e-4→3e-5。</p>',
              '<p>该光线入口均值较高，本次三个划分均未观察到负入口实现；首次上穿分支未由此实验验证。</p>',
              '<p>主结果固定使用最后一轮；辅助结果使用独立验证集联合 NLL 最小的检查点。'
              '所有曲线固定展示种子 20261042，不按测试结果挑选。均值 ± 标准差描述三个优化种子；'
              'GP 采样误差另由测试样本数量、参考 CDF 区间和配对 NLL 区间描述。</p>']
    for checkpoint, title in (('last', '主结果：固定最后一轮'), ('best', '辅助结果：验证集选择检查点')):
        lines.append(f'<h2>{title}</h2><table><tr><th>模型</th><th>测试梯度 NLL</th><th>整体 PIT KS</th>'
                     '<th>8 区间平均 PIT KS</th><th>最大区间 PIT KS</th><th>最大透射率误差</th></tr>')
        base = summary['source']
        lines.append(f'<tr><td>原 RNO</td><td>{base["speed_nll_on_hits"]:.6f}</td><td>{base["speed_pit_ks"]:.6f}</td>'
                     f'<td>{base["mean_bin_pit_ks"]:.6f}</td><td>{base["maximum_bin_pit_ks"]:.6f}</td>'
                     f'<td>{base["transmittance"]["maximum_absolute_error"]:.6f}</td></tr>')
        for mode, label in (('head_only', '仅微调梯度头'), ('joint', '联合微调')):
            row = summary['aggregate'][checkpoint][mode]
            keys = ('speed_nll_on_hits', 'speed_pit_ks', 'mean_bin_pit_ks', 'maximum_bin_pit_ks', 'maximum_T_error')
            cells = ''.join(f'<td>{row[k]["mean"]:.6f} ± {row[k]["standard_deviation"]:.6f}</td>' for k in keys)
            lines.append(f'<tr><td>{label}</td>{cells}</tr>')
        lines.append('</table>')
    lines += ['<h2>不同距离区间的梯度分布</h2><p>横轴是正的归一化穿越速度 W；'
              '模型 CDF 在该区间每个参考首碰位置和入口符号处求值，再取平均，因此与该区间的经验 CDF 匹配。'
              '不能用区间中心一次预测代替这个平均。灰带是各参考经验 CDF 的 95% DKW 区间，仅计 Monte Carlo 不确定性。</p>',
              '<img src="plots/distance_cdfs.svg" alt="8 个距离区间的梯度 CDF">',
              '<h2>局部概率校准</h2><p>PIT 在每个参考位置计算后分区间检验；虚线使用 8 区间 Bonferroni/DKW 校正。'
              '满足阈值只表示未检出明显偏差，不证明模型精确。右图专门查看小穿越速度 W≤0.5 的概率。</p>',
              '<img src="plots/calibration.svg" alt="逐区间校准">',
              '<h2>透射率</h2><p>梯度头微调组的透射率与原模型完全相同；联合组同时学习首碰距离分布。</p>',
              '<img src="plots/transmittance.svg" alt="透射率">',
              '<h2>训练收敛</h2><img src="plots/learning.svg" alt="独立验证损失曲线">',
              '<h2>复核与解释范围</h2>',
              f'<p>配对轨迹索引一致：{summary["paired_training_indices_identical"]}；原始权重未修改：{summary["source_checkpoint_unchanged"]}。'
              '代表模型均完成 Python/C++ 参数预测及 512 次碰撞采样检查。横向梯度继续使用固定 SE 解析分布；'
              '这里主要验证所学沿光线梯度头，未把其指标解释为模型自采样碰撞位置后的完整法线误差。</p>',
              '<p>粗细参考网格使用独立 GP 随机流。网格之间的差异同时包含 Monte Carlo 波动；'
              '两级网格不能证明连续路径离散化偏差为零。对照中的种子变化也不代表不同场景。</p>',
              '<p><a href="checks/head_only_s20261042/verification.json">梯度头 C++ 复核</a> · '
              '<a href="checks/joint_s20261042/verification.json">联合模型 C++ 复核</a></p></html>']
    (root/'comparison.html').write_text('\n'.join(lines), encoding='utf-8')
    return {'report': str(root/'comparison.html')}


def run(config_path, phase='all'):
    result = None
    if phase in ('collect', 'all'):
        result = collect(config_path)
    if phase in ('train', 'all'):
        result = train(config_path)
    if phase in ('evaluate', 'all'):
        evaluate(config_path)
        result = {'summary': str(resolve(read_json(config_path)['output_directory'])/'summary.json')}
    if phase in ('report', 'all'):
        result = report(config_path)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--phase', choices=('collect', 'train', 'evaluate', 'report', 'all'), default='all')
    args = parser.parse_args()
    print(run(args.config, args.phase))


if __name__ == '__main__':
    main()
