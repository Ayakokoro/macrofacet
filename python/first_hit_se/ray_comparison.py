"""Frozen, deployed GRU versus its analytic local baseline on individual rays.

No training, label generation, checkpoint selection, or renderer changes.
The analytic local gradient law is the flux-weighted positive normal, not the
learned 'mixture_local' ablation or the renderer's unit-gradient local_tangent.
"""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

from renewal.collect import digest, resolve, write_json
from .causal_data import batch_profiles
from .causal_gradient import points
from .causal_gradient_study import load_head
from .causal_residual import branch_values
from .causal_study import load_run
from .distance_probe import ks
from .gradient_cdf import cdf, log_prob
from .hazard import log_negative_part_mean
from .hazard_study import parameter_hash
from .spline import local_profile


def local_speed_log_prob(w, mu):
    """p(w) = w phi(w-mu) / E[max(N(mu,1),0)], w>0."""
    w, mu = torch.broadcast_tensors(w.double(), mu.double())
    result = w.clamp_min(1e-300).log() - .5*(w-mu).square()
    result -= .5*math.log(2*math.pi) + log_negative_part_mean(-mu)
    return torch.where(w > 0, result, -torch.inf)


def local_speed_cdf(w, mu):
    """Stable analytic CDF, including very unlikely crossing directions.

    With t=w-mu, the survival numerator phi(t)+mu Phi(-t) is evaluated
    as E[max(-N(t,1),0)] + w Phi(-t), a sum of positive terms.
    """
    w, mu = torch.broadcast_tensors(w.double(), mu.double())
    positive = w.clamp_min(0)
    t = positive-mu
    log_tail = torch.logaddexp(log_negative_part_mean(t),
                              positive.log()+torch.special.log_ndtr(-t))
    result = -torch.expm1((log_tail-log_negative_part_mean(-mu)).clamp_max(0))
    return torch.where(w > 0, result.clamp(0, 1), 0.)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def array(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value).round(8).tolist()


def ecdf(values, grid):
    return torch.searchsorted(values.sort().values, grid.contiguous(), right=True).double()/len(values)


def camera_metadata(raw, catalog):
    meta_path = Path(raw['source']).parent/'ray.json'
    meta = read_json(meta_path)
    # Dataset merging renumbers the terminal camera_pixel id. Match the
    # recorded physical camera/pixel instead of interpreting that id as a pixel.
    if raw['scene_id'] == 'shader_ball_sigma002_camera_only':
        return {'camera': 'original_training_camera', 'pixel': meta['pixel'],
                'resolution': [512, 512], 'group': 'original_24'}
    camera, group = raw['scene_id'].removeprefix('sigma002_').rsplit('_', 1)
    selected = catalog.get((camera, tuple(meta['pixel'])))
    if selected:
        if int(selected['hits']) != int(raw['hit'].sum()):
            raise ValueError('camera metadata hit count mismatch')
        config = read_json(Path(selected['render_config']))
        return {'camera': selected['camera'], 'pixel': json.loads(selected['pixel']),
                'resolution': [config['render']['width'], config['render']['height']],
                'group': selected['group']}
    raise ValueError(f'camera metadata missing: {camera}, {meta["pixel"]}')


def metric(values, nll):
    return {'pit_ks': ks(values), 'speed_nll': float(nll.mean())}


def summarize(rows):
    qualified = [r for r in rows if r['hits'] >= 1024]
    def ids(key, reverse=True, eligible=None):
        return [r['ray'] for r in sorted(qualified if eligible is None else eligible,
                                        key=key, reverse=reverse)[:6]]
    rankings = {
        'T_worst': ids(lambda r: r['T_gru'], eligible=rows),
        'T_regression': ids(lambda r: r['T_gru']-r['T_local'], eligible=rows),
        'T_improvement': ids(lambda r: r['T_local']-r['T_gru']),
        'T_best_with_hits': ids(lambda r: r['T_gru'], False),
        'gradient_worst': ids(lambda r: r['gradient_gru']['pit_ks']),
        'gradient_regression': ids(lambda r: r['gradient_gru']['pit_ks']-r['gradient_local']['pit_ks']),
        'gradient_improvement': ids(lambda r: r['gradient_local']['pit_ks']-r['gradient_gru']['pit_ks']),
        'gradient_best': ids(lambda r: r['gradient_gru']['pit_ks'], False)}
    total = sum(r['samples'] for r in rows)
    hits = sum(r['hits'] for r in rows)
    summary = {'rays': len(rows), 'gradient_rays_with_1024_hits': len(qualified),
               'reference_trials': total, 'reference_hits': hits, 'rankings': rankings}
    for name in ('local', 'gru'):
        summary[name] = {
            'mean_ray_max_T_error': float(np.mean([r['T_'+name] for r in rows])),
            'mean_ray_gradient_pit_ks_1024': float(np.mean([r['gradient_'+name]['pit_ks'] for r in qualified])),
            'speed_nll_per_hit': sum(r['gradient_'+name]['speed_nll']*r['hits'] for r in rows if r['hits'])/hits,
            'joint_distance_nll_per_trial': sum(r['joint_distance_'+name]*r['samples'] for r in rows)/total}
        summary[name]['joint_gradient_nll_per_trial'] = summary[name]['speed_nll_per_hit']*hits/total
        summary[name]['joint_nll_per_trial'] = (summary[name]['joint_distance_nll_per_trial']
                                               + summary[name]['joint_gradient_nll_per_trial'])
    summary['T_improved_count'] = sum(r['T_gru'] < r['T_local']-1e-9 for r in rows)
    summary['T_regressed_count'] = sum(r['T_gru'] > r['T_local']+1e-9 for r in rows)
    summary['gradient_improved_count_1024'] = sum(r['gradient_gru']['pit_ks'] < r['gradient_local']['pit_ks'] for r in qualified)
    return summary


@torch.no_grad()
def evaluate(args, output):
    bundle_path = resolve(args.bundle)
    bundle = read_json(bundle_path)
    source = bundle['provenance']
    backbone_path, head_path = Path(source['backbone']), Path(source['gradient'])
    for key, path in (('backbone', backbone_path), ('gradient', head_path)):
        if digest(path) != source[key+'_sha256']:
            raise ValueError('deployed checkpoint changed: '+key)
    backbone, saved = load_run(backbone_path)
    head, head_saved = load_head(head_path)
    seed = saved['identity']['seed']
    if seed != head_saved['identity']['seed']:
        raise ValueError('unpaired deployed models')
    distance_dir, gradient_dir = backbone_path.parents[2], head_path.parents[2]
    protocol_path = gradient_dir/'protocol.json'
    protocol = read_json(protocol_path)
    if digest(protocol_path) != source['gradient_protocol_sha256']:
        raise ValueError('gradient protocol changed')
    for path, expected in protocol['source_sha256'].items():
        if digest(Path(path)) != expected:
            raise ValueError('frozen implementation changed: '+path)
    dataset_path = resolve(protocol['specification']['dataset'])
    if digest(dataset_path) != protocol['dataset_sha256']:
        raise ValueError('dataset changed')
    print('Loading frozen models, test curves, and reference-position contexts', flush=True)
    data = torch.load(dataset_path, weights_only=True, map_location='cpu')
    raw = [p for p in data['profiles'] if p['split'] == 'test']
    train_w = torch.cat([p['speed'][p['hit']] for p in data['profiles'] if p['split'] == 'train'])
    display_max = float(torch.quantile(train_w, .995))*1.15
    context_path = gradient_dir/f'contexts_{seed}_test.pt'
    cache = torch.load(context_path, weights_only=True, map_location='cpu')
    if (cache['identity'] != {'protocol_sha256': digest(protocol_path), 'seed': seed, 'split': 'test'}
            or cache['backbone_parameter_sha256'] != parameter_hash(backbone)):
        raise ValueError('cached contexts do not match the deployed backbone')
    prepared = torch.load(distance_dir/'prepared_test.pt', weights_only=True, map_location='cpu')
    if prepared['identity']['dataset_sha256'] != protocol['dataset_sha256']:
        raise ValueError('distance cache dataset mismatch')
    distance = read_json(distance_dir/'evaluation.json')
    gradients = read_json(gradient_dir/'evaluation.json')
    curves = next(r for r in distance['results']['gru_nll'] if r['seed'] == seed)['metrics']['per_ray']
    local_curves = distance['results']['local_causal_grid'][0]['metrics']['per_ray']
    old_gradient = next(r for r in gradients['results']['mixture_gru'] if r['seed'] == seed)['metrics']['per_ray']
    if (distance['checkpoint_sha256'][str(backbone_path)] != source['backbone_sha256']
            or gradients['checkpoint_sha256'][str(head_path)] != source['gradient_sha256']):
        raise ValueError('saved evaluation is for a different model')
    with resolve('outputs/first_hit_camera_holdout_sigma002_20261015/test_rays.csv').open() as stream:
        catalog = {(r['camera'], tuple(json.loads(r['pixel']))): r for r in csv.DictReader(stream)}
    head.to(args.device).eval()
    rows, details, max_pit_difference = [], [], 0.
    for index, (p, ctx, prepared_row) in enumerate(zip(raw, cache['rows'], prepared['rows'])):
        n, hits = len(p['hit']), int(p['hit'].sum())
        q = torch.linspace(0, 1, len(curves[index]['T']), dtype=torch.float64)*p['horizon']
        ref_T = 1-torch.searchsorted(p['distance'][p['hit']].sort().values, q, right=True).double()/n
        b, db = local_profile(p['features'], q)
        batch = batch_profiles([prepared_row])
        local_H, local_rate = branch_values(torch.zeros_like(batch['mass']), batch, batch['events'])
        branch = (p['initial_sign'] < 0).long()
        sample_index = torch.arange(n)
        local_log_T = torch.logsumexp(batch['log_prior'][0, :, None]-local_H[0], 0)
        local_joint = torch.where(p['hit'], local_H[0, branch, sample_index]-local_rate[0, branch, sample_index]
                                 -batch['log_prior'][0, branch], -local_log_T)
        delta = backbone(batch)[0]
        row = {'ray': index, 'id': p['id'], **camera_metadata(p, catalog), 'samples': n, 'hits': hits,
               'negative_starts': int((p['initial_sign'] < 0).sum()), 'horizon_q': float(p['horizon']),
               'ell': p['geometry']['ell'], 'entry': p['geometry']['origin'], 'direction': p['geometry']['direction'],
               'T_local': local_curves[index]['max_abs_T_error'], 'T_gru': curves[index]['max_abs_T_error'],
               'T_gru_seed_range': [min(r['metrics']['per_ray'][index]['max_abs_T_error'] for r in distance['results']['gru_nll']),
                                    max(r['metrics']['per_ray'][index]['max_abs_T_error'] for r in distance['results']['gru_nll'])],
               'reference_escape': float(ref_T[-1]), 'local_escape': local_curves[index]['T'][-1],
               'gru_escape': curves[index]['T'][-1], 'joint_distance_local': float(local_joint.mean()),
               'joint_distance_gru': float(ctx['joint_distance_sum']/n)}
        for name, T in (('local', local_curves[index]['T']), ('gru', curves[index]['T'])):
            error = float((torch.tensor(T, dtype=torch.float64)-ref_T).abs().max())
            if abs(error-row['T_'+name]) > 1e-10:
                raise AssertionError('distance reference/evaluation mismatch')
        detail = {'ray': index, 'q': array(q), 'b': array(b), 'db': array(db), 'reference_T': array(ref_T),
                  'local_T': array(local_curves[index]['T']), 'gru_T': array(curves[index]['T']),
                  'delta_q': array(torch.arange(delta.shape[1])*backbone.config.step),
                  'delta_positive': array(delta[0]), 'delta_negative': array(delta[1]), 'conditions': []}
        if hits:
            cp = points([ctx])
            device_points = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in cp.items()}
            params = head(device_points)
            w, mu = device_points['w'], -device_points['sign']*device_points['db']
            values = {'local': local_speed_cdf(w, mu), 'gru': cdf('mixture', params, w)}
            losses = {'local': -local_speed_log_prob(w, mu), 'gru': -log_prob('mixture', params, w)}
            for name in values:
                if not torch.isfinite(values[name]).all() or not torch.isfinite(losses[name]).all():
                    raise FloatingPointError('invalid gradient law')
                row['gradient_'+name] = metric(values[name], losses[name])
            difference = abs(row['gradient_gru']['pit_ks']-old_gradient[index]['pit_ks'])
            max_pit_difference = max(max_pit_difference, difference)
            if difference > 1e-5:
                raise AssertionError('deployed gradient metrics differ from saved evaluation')
            row['pit_sampling_scale'] = 1.36/math.sqrt(hits)
            grid = torch.linspace(0, max(display_max, float(w.max())*1.02), 129,
                                  dtype=torch.float64, device=args.device)
            pit_grid = torch.linspace(0, 1, 101, dtype=torch.float64, device=args.device)
            quantiles = torch.quantile(cp['q'], torch.tensor([0, .25, .5, .75, 1.], dtype=torch.float64))
            labels = ['all', 'q1', 'q2', 'q3', 'q4']
            masks = [torch.ones(hits, dtype=torch.bool, device=args.device)]
            for j in range(4):
                masks.append(((device_points['q'] >= float(quantiles[j])) &
                              ((device_points['q'] < float(quantiles[j+1])) if j < 3
                               else (device_points['q'] <= float(quantiles[j+1])))))
            sums = {name: torch.zeros(5, len(grid), dtype=torch.float64, device=args.device)
                    for name in ('local', 'gru')}
            for start in range(0, hits, 1024):
                stop = min(start+1024, hits)
                grid_cdfs = {'local': local_speed_cdf(grid[None], mu[start:stop, None]),
                             'gru': cdf('mixture', tuple(v[start:stop, None] for v in params), grid[None])}
                weights = torch.stack([mask[start:stop] for mask in masks]).double()
                for name, predicted in grid_cdfs.items():
                    sums[name] += weights@predicted
            detail['speed_grid'], detail['pit_grid'] = array(grid), array(pit_grid)
            for j, mask in enumerate(masks):
                count = int(mask.sum())
                condition = {'name': labels[j], 'hits': count}
                if count:
                    condition.update(q_range=[float(device_points['q'][mask].min()), float(device_points['q'][mask].max())],
                                     reference_cdf=array(ecdf(w[mask], grid)),
                                     negative_fraction=float((device_points['sign'][mask] < 0).double().mean()))
                    for name in sums:
                        condition[name+'_cdf'] = array(sums[name][j]/count)
                        condition[name+'_pit_cdf'] = array(ecdf(values[name][mask], pit_grid))
                        condition[name+'_pit_ks'] = ks(values[name][mask])
                        condition[name+'_speed_nll'] = float(losses[name][mask].mean())
                detail['conditions'].append(condition)
        else:
            row.update(gradient_local=None, gradient_gru=None, pit_sampling_scale=None)
        rows.append(row)
        details.append(detail)
        if (index+1) % 20 == 0:
            print(f'Compared {index+1}/{len(raw)} rays', flush=True)
    summary = summarize(rows)
    result = {'seed': seed, 'bundle': str(bundle_path), 'model_config': bundle['model_config'],
              'gradient_config': bundle['gradient_config'], 'summary': summary, 'rows': rows,
              'audits': {'checkpoint_and_dataset_hashes_match': True,
                         'cached_backbone_parameters_match': True, 'training_sources_unchanged': True,
                         'max_saved_gradient_PIT_difference': max_pit_difference},
              'provenance': {**source, 'dataset_sha256': protocol['dataset_sha256'],
                             'bundle_sha256': digest(bundle_path), 'evaluation_source_sha256': digest(Path(__file__)),
                             'context_sha256': digest(context_path)},
              'reference_sampler': data['reference_sampler'],
              'notes': ['Indices are zero-based test ray indices, not image pixel indices.',
                        'Analytic local uses the raw mean gradient and both entry signs; it is not renderer classic_local.',
                        'Gradient CDFs average predictions at the same reference hit positions and signs.',
                        'q1..q4 are empirical hit-position quartiles; they are not equal-length intervals.',
                        'Gradients mean the longitudinal secant speed W, not validated three-dimensional normals.',
                        'Examples are selected after evaluation for diagnosis, not for training or checkpoint selection.',
                        'Two error metrics cannot be directly compared by their magnitude; joint NLL has an additive decomposition.',
                        'Reference is finite-history, finite-step GPIS, not exact continuous-GP ground truth.']}
    write_json(output/'summary.json', result)
    write_json(output/'curves.json', details)
    with (output/'per_ray.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        fields = ['ray', 'id', 'camera', 'pixel', 'hits', 'T_local', 'T_gru', 'T_delta',
                  'gradient_local_PIT', 'gradient_gru_PIT', 'gradient_PIT_delta']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({**{k: r[k] for k in fields[:7]}, 'T_delta': r['T_gru']-r['T_local'],
                             'gradient_local_PIT': r['gradient_local']['pit_ks'] if r['hits'] else '',
                             'gradient_gru_PIT': r['gradient_gru']['pit_ks'] if r['hits'] else '',
                             'gradient_PIT_delta': r['gradient_gru']['pit_ks']-r['gradient_local']['pit_ks'] if r['hits'] else ''})
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', default='outputs/models/first_hit_causal_gru_sigma002/first_hit_model.json')
    parser.add_argument('--output', default='outputs/first_hit_causal_ray_comparison_sigma002_20261016')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--phase', choices=('all', 'evaluate', 'report'), default='all')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    output = resolve(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.phase in ('all', 'evaluate'):
        evaluate(args, output)
    if args.phase in ('all', 'report'):
        from .ray_comparison_report import report
        report(output)


if __name__ == '__main__':
    main()
