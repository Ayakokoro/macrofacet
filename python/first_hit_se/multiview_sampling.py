"""Joint distance/speed diagnostics and separately scoped inference timings."""
import statistics
import time

import numpy as np
import torch

from renewal.collect import digest, write_json
from .distance_models import sample_distance
from .distance_study import load_run, synchronize
from .frozen_gradient_study import cache_profiles, head_parameters, load_head, points
from .gradient_cdf import cdf, sample
from .multiview_study import test_groups
from .spline import local_profile


@torch.no_grad()
def measure(head, kind, conditions, uniforms, device):
    params = head_parameters(head, conditions)
    functions = {
        'inverse_only': lambda: sample(kind, params, uniforms),
        'head_and_inverse': lambda: sample(kind, head_parameters(head, conditions), uniforms),
    }
    results = {}
    iterations, repeats = (100 if device == 'cpu' else 50), 7
    for name, function in functions.items():
        for _ in range(20):
            function()
        durations = []
        for _ in range(repeats):
            synchronize(device)
            began = time.perf_counter()
            for _ in range(iterations):
                function()
            synchronize(device)
            durations.append((time.perf_counter()-began)/iterations)
        median = statistics.median(durations)
        results[name] = {'batch_microseconds': median*1e6,
                         'microseconds_per_sample': median*1e6/len(uniforms),
                         'repeat_batch_microseconds': [s*1e6 for s in durations]}
    return results


@torch.no_grad()
def sampling_checks(spec, config, data, output):
    seed = spec['seeds'][0]  # Declared first seed, without test-driven selection.
    checkpoint = output/'runs'/f'residual_{seed}'/'best.pt'
    model, _ = load_run(checkpoint, config['device'])
    model.requires_grad_(False)
    splits = cache_profiles(model, data, config)
    heads = {kind: load_head(spec, config, output, kind, seed, config['device'])
             for kind in ('mixture', 'cdf')}
    device = config['device']
    training_speeds = torch.cat([p['speed'][p['hit']] for p in splits['train']])
    cuts = torch.quantile(training_speeds, training_speeds.new_tensor([.25, .5, .75]))
    fractions = cuts.new_tensor([.1, .25, .5, .75, 1.])
    samples = 8192
    generator = torch.Generator(device=device).manual_seed(seed+700000)
    rows = []
    originals = [p for p in data['profiles'] if p['split'] == 'test']
    for i, (p, original) in enumerate(zip(splits['test'], originals)):
        uniform = torch.rand(samples, 4, generator=generator, device=device, dtype=torch.float64)
        sign = torch.where(uniform[:, 0] < torch.special.ndtr(p['b0']), 1., -1.)
        q = p['horizon'].expand(samples).clone()
        hit = torch.zeros(samples, dtype=torch.bool, device=device)
        for j, branch in enumerate((1., -1.)):
            selected = sign == branch
            if not selected.any():
                continue
            distance, valid = sample_distance(model, p['fixed'][None], sign.new_tensor([branch]),
                p['baseline'][j:j+1], p['horizon'][None], uniform[selected, 1][None])
            q[selected], hit[selected] = distance[0], valid[0]
        b, db = local_profile(original['features'].to(device), q[hit])
        conditions = {'z': p['latent'][(sign[hit] < 0).long()], 'q': q[hit],
                      'horizon': p['horizon'].expand(int(hit.sum())),
                      'sign': sign[hit], 'b': b, 'db': db}
        reference = ((p['distance'][p['hit'], None, None]/p['horizon'] <= fractions[None, :, None]) &
                     (p['speed'][p['hit'], None, None] <= cuts[None, None, :])).sum(0).double()/len(p['hit'])
        within = q[hit, None]/p['horizon'] <= fractions[None]
        record = {'ray': i, 'reference_rectangles': reference.cpu().tolist(),
                  'escape_error': float((~hit).double().mean()-(~p['hit']).double().mean()),
                  'sampled_hits': int(hit.sum()), 'heads': {}}
        for kind, head in heads.items():
            params = head_parameters(head, conditions)
            speed = sample(kind, params, uniform[hit, 2:])
            if not torch.isfinite(speed).all() or (speed < 0).any():
                raise FloatingPointError('invalid sampled speed')
            rectangles = (within[:, :, None] & (speed[:, None, None] <= cuts[None, None])).sum(0).double()/samples
            # Conditional expectation at the same sampled distances: checks the
            # inverse sampler against its own CDF with less Monte Carlo noise.
            expected = torch.stack([(within*cdf(kind, params, c.expand(len(speed)))[:, None]).sum(0)/samples
                                    for c in cuts], -1)
            errors = (rectangles-reference).abs()
            record['heads'][kind] = {'rectangles': rectangles.cpu().tolist(),
                'mean_abs_joint_cdf_error': float(errors.mean()), 'max_abs_joint_cdf_error': float(errors.max()),
                'mean_abs_expected_joint_cdf_error': float((expected-reference).abs().mean()),
                'max_sampler_vs_own_cdf_error': float((rectangles-expected).abs().max())}
        rows.append(record)
        if (i+1) % 40 == 0:
            print(f'joint sampling {i+1}/{len(originals)} rays', flush=True)
    grouped = {}
    for group, ids in test_groups(data).items():
        grouped[group] = {kind: {key: float(np.mean([rows[i]['heads'][kind][key] for i in ids]))
                          for key in ('mean_abs_joint_cdf_error', 'max_abs_joint_cdf_error',
                                      'mean_abs_expected_joint_cdf_error')}
                          for kind in heads}
        grouped[group]['mean_abs_escape_error'] = float(np.mean([abs(rows[i]['escape_error']) for i in ids]))
    # Validation conditions determine timing inputs. Random generation, the
    # backbone, local profile lookup and distance sampling are outside timings.
    p = points(splits['validation'])
    timings = {}
    for target, count in (('cpu', 1), (device, 4096)):
        ids = torch.linspace(0, len(p['q'])-1, count, device=device).long()
        conditions = {k: p[k][ids].to(target) for k in ('z', 'q', 'horizon', 'sign', 'b', 'db')}
        uniforms = torch.rand(count, 2, generator=generator, device=device, dtype=torch.float64).to(target)
        timings[f'{target}_batch_{count}'] = {kind: measure(
            load_head(spec, config, output, kind, seed, target), kind, conditions, uniforms, target)
            for kind in heads}
        print(f'gradient sampling benchmark finished: {target}, batch={count}', flush=True)
    result = {'seed': seed, 'checkpoint_sha256': digest(checkpoint), 'samples_per_ray': samples,
        'q_over_horizon_thresholds': fractions.cpu().tolist(), 'training_speed_quantiles': cuts.cpu().tolist(),
        'reference_and_model_worst_case_pointwise_mc_standard_deviation': float(np.sqrt(.5/samples)),
        'groups': grouped, 'per_ray': rows, 'timings': timings,
        'max_sampler_vs_own_cdf_error': {kind: max(r['heads'][kind]['max_sampler_vs_own_cdf_error'] for r in rows) for kind in heads},
        'scope': 'first declared seed; longitudinal speed and distance joint rectangles, no rendered images',
        'timing_scope': 'PyTorch eager; precomputed latent/local inputs and uniforms; excludes full ray setup, field access and distance sampling',
        'timing_cpu_threads': torch.get_num_threads(), 'torch_version': str(torch.__version__),
        'gpu': torch.cuda.get_device_name() if torch.cuda.is_available() else None}
    write_json(output/'sampling_evaluation.json', result)
    return {'groups': grouped, 'timings': timings}
