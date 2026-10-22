"""Analyze fixed-state contact-flight probabilities and matched-distance normals.

Usage: python tools/analyze_contact_b.py outputs/diagnostics/contact_b_20261022
Requires numpy and scipy; leaves renderer/model settings unchanged.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance


VARIANTS = (
    'neural', 'joint_at_gpis_distance', 'joint_at_gpis_distance_no_memory',
    'gpis_speed_full_memory', 'gpis_speed_no_memory',
)


def read_pfm(path):
    with Path(path).open('rb') as stream:
        if stream.readline().strip() != b'PF':
            raise ValueError('expected RGB PFM')
        width, height = map(int, stream.readline().split())
        scale = float(stream.readline())
        values = np.fromfile(stream, '<f4' if scale < 0 else '>f4')
    return values.reshape(height, width, 3)[::-1].astype(float)*abs(scale)


def samples(directory, index, method):
    data = np.fromfile(directory/f'state_{index}_{method}.bin', '<f8').reshape(-1, 5)
    if not np.isfinite(data).all() or not np.isin(data[:, 0], [0, 1, 2, 3]).all():
        raise ValueError('invalid sample file')
    return data


def hit_features(data, state):
    hit = data[data[:, 0] == 1]
    direction = np.array(state['direction'])
    g = hit[:, 2:5]
    lengths = np.linalg.norm(g, axis=-1)
    if (lengths <= 0).any() or (g@direction >= 0).any():
        raise ValueError('accepted collision violates support')
    normal = g/lengths[:, None]
    return {
        'q': hit[:, 1]/state['ell'],
        'speed': -(g@direction)*state['ell']/state['sigma'],
        'cos_incidence': -(normal@direction),
        'normal_t1': normal@np.array(state['e1']),
        'normal_t2': normal@np.array(state['e2']),
        'normal': normal,
    }


def compare_features(actual, reference):
    n, m = len(actual['q']), len(reference['q'])
    if min(n, m) < 2:
        return {'hits': n, 'reference_hits': m, 'usable': False}
    result = {'hits': n, 'reference_hits': m, 'usable': min(n, m) >= 64}
    for key in ('q', 'speed', 'cos_incidence', 'normal_t1', 'normal_t2'):
        a, b = actual[key], reference[key]
        result[key] = {
            'ks': float(ks_2samp(a, b, method='asymp').statistic),
            'wasserstein': float(wasserstein_distance(a, b)),
            'mean': float(a.mean()), 'reference_mean': float(b.mean()),
            'quantiles_10_50_90': np.quantile(a, [.1, .5, .9]).tolist(),
            'reference_quantiles_10_50_90': np.quantile(b, [.1, .5, .9]).tolist(),
        }
    result['mean_normal_difference_norm'] = float(np.linalg.norm(
        actual['normal'].mean(0)-reference['normal'].mean(0)))
    return result


def probability_interval(p, n):
    # Wilson interval, individual fixed state only.
    z = 1.959963984540054
    denominator = 1+z*z/n
    center = (p+z*z/(2*n))/denominator
    radius = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/denominator
    return [float(max(0, center-radius)), float(min(1, center+radius))]


def aggregate(rows):
    weights = np.array([r['population_weight'] for r in rows])
    average = lambda values: float(np.average(values, weights=weights))
    result = {
        'states': len(rows), 'represented_B_visits': float(weights.sum()),
        'joint_collision_probability': average([r['joint_collision_probability'] for r in rows]),
        'gpis_collision_probability': average([r['gpis_collision_probability'] for r in rows]),
        'gpis_rejected_probability': average([r['gpis_rejected_probability'] for r in rows]),
        'joint_escape_probability': average([r['joint_escape_probability'] for r in rows]),
        'gpis_escape_probability': average([r['gpis_escape_probability'] for r in rows]),
        'mean_abs_collision_probability_difference': average([abs(r['collision_difference']) for r in rows]),
        'outside_training_derivative_residual_fraction': average([abs(r['derivative_residual']) > 3 for r in rows]),
        'max_abs_collision_probability_difference': max(abs(r['collision_difference']) for r in rows),
    }
    result['collision_probability_difference'] = result['joint_collision_probability']-result['gpis_collision_probability']
    result['escape_probability_difference'] = result['joint_escape_probability']-result['gpis_escape_probability']
    result['normals'] = {}
    for variant in VARIANTS:
        selected = [r for r in rows if r['normal_comparisons'][variant]['usable']]
        if not selected:
            result['normals'][variant] = {'usable_states': 0}
            continue
        # Condition on a GPIS collision, using the same state/distance measure
        # for all matched variants. Per-state KS avoids cancellation by pooling.
        w = [r['population_weight']*r['gpis_collision_probability'] for r in selected]
        metric = {'usable_states': len(selected)}
        for key in ('q', 'speed', 'cos_incidence', 'normal_t1', 'normal_t2'):
            for name in ('ks', 'wasserstein', 'mean', 'reference_mean'):
                metric[f'{key}_{name}'] = float(np.average(
                    [r['normal_comparisons'][variant][key][name] for r in selected], weights=w))
        metric['mean_normal_difference_norm'] = float(np.average(
            [r['normal_comparisons'][variant]['mean_normal_difference_norm'] for r in selected], weights=w))
        result['normals'][variant] = metric
    return result


def analyze(directory):
    directory = Path(directory)
    identity = json.loads((directory/'identity.json').read_text())
    captured = json.loads((directory/'capture.json').read_text())
    complete = json.loads((directory/'comparison.json').read_text())
    image = read_pfm(identity['request']['source_image'])
    differences = []
    for pixel in captured['pixels']:
        x, y = pixel['pixel']
        differences.append(np.max(np.abs(np.array(pixel['radiance'])-image[y, x])))
    replay = {'pixels': len(differences), 'max_abs_radiance_error': float(max(differences)),
              'within_float32_rounding': bool(max(differences) < 1e-7)}
    if not replay['within_float32_rounding']:
        raise ValueError(f'camera replay differs from source image: {replay}')
    records = {r['id']: r for r in complete['states']}
    rows = []
    largest_sampling_z = 0.
    largest_paired_speed_error = 0.
    for state in captured['states']:
        i = state['id']; raw = records[i]; n = raw['trials']
        gp = samples(directory, i, 'gpis')
        nn = samples(directory, i, 'neural')
        if len(gp) != n or len(nn) != n or raw['gpis_counts'][3]:
            raise ValueError('missing samples or invalid GPIS gradients')
        for values, counts in ((gp, raw['gpis_counts']), (nn, raw['neural_counts'])):
            if np.bincount(values[:, 0].astype(int), minlength=4).tolist() != counts:
                raise ValueError('binary outcomes differ from summary counts')
            if (values[:, 1] < 0).any() or (values[:, 1] > state['horizon']*state['ell']+1e-6).any():
                raise ValueError('sample outside flight domain')
        exact_p = 1-raw['predicted_escape']
        deviation = abs(float((nn[:, 0] == 1).mean())-exact_p)
        # One-count floor handles probabilities rounded to zero or one.
        z = deviation/max(np.sqrt(exact_p*(1-exact_p)/n), 1/n)
        largest_sampling_z = max(largest_sampling_z, z)
        if z > 6:
            raise ValueError('neural inverse samples disagree with integrated probability')
        reference = hit_features(gp, state)
        row = {k: state[k] for k in ('id', 'region', 'kind', 'depth_group', 'population_weight',
                                    'pixel', 'camera_sample', 'completed_collisions', 'b0',
                                    'bprime0', 'beta_g', 'derivative_residual', 'throughput')}
        row.update(
            joint_collision_probability=1-raw['predicted_escape'],
            joint_sampled_collision_probability=float((nn[:, 0] == 1).mean()),
            gpis_collision_probability=raw['gpis_counts'][1]/n,
            gpis_rejected_probability=raw['gpis_counts'][2]/n,
            joint_escape_probability=raw['predicted_escape'],
            gpis_escape_probability=raw['gpis_counts'][0]/n,
            gpis_hit_interval_95=probability_interval(raw['gpis_counts'][1]/n, n),
            near_birth_joint_collision_probability=float(((nn[:, 0] == 1) & (nn[:, 1]/state['ell'] < .125)).mean()),
            near_birth_gpis_collision_probability=float(((gp[:, 0] == 1) & (gp[:, 1]/state['ell'] < .125)).mean()),
        )
        row['collision_difference'] = row['joint_collision_probability']-row['gpis_collision_probability']
        row['escape_difference'] = row['joint_escape_probability']-row['gpis_escape_probability']
        row['normal_comparisons'] = {}
        feature_sets = {}
        for variant in VARIANTS:
            values = nn if variant == 'neural' else samples(directory, i, variant)
            if variant != 'neural' and len(values) != len(reference['q']):
                raise ValueError('matched-distance sample counts differ')
            features = hit_features(values, state)
            feature_sets[variant] = features
            row['normal_comparisons'][variant] = compare_features(features, reference)
        if len(reference['q']):
            for left, right in (('joint_at_gpis_distance', 'joint_at_gpis_distance_no_memory'),
                                ('gpis_speed_full_memory', 'gpis_speed_no_memory')):
                a, b = feature_sets[left]['speed'], feature_sets[right]['speed']
                error = float(np.abs(a-b).max())
                largest_paired_speed_error = max(largest_paired_speed_error, error)
                if not np.allclose(a, b, atol=1e-10, rtol=1e-10):
                    raise ValueError('transverse ablation changed longitudinal speed')
            if not np.allclose(feature_sets['gpis_speed_no_memory']['speed'], reference['speed'], atol=1e-10, rtol=1e-10):
                raise ValueError('GPIS-speed control changed the GPIS longitudinal derivative')
        rows.append(row)
    regions = {r['name']: aggregate([x for x in rows if x['region'] == r['name']])
               for r in identity['request']['regions']}
    groups = {}
    for kind in ('contact', 'control'):
        for depth in ('first_B', 'later_B'):
            selected = [r for r in rows if r['kind'] == kind and r['depth_group'] == depth]
            if selected:
                groups[kind+'_'+depth] = aggregate(selected)
    worst = sorted(rows, key=lambda r: abs(r['collision_difference']), reverse=True)
    report = {
        'replay': replay, 'trials_per_state': identity['request']['trials_per_state'],
        'experiment_checks': {'largest_neural_probability_sampling_z': float(largest_sampling_z),
                              'largest_paired_longitudinal_speed_error': largest_paired_speed_error,
                              'binary_counts_and_support': 'passed', 'gpis_invalid_gradients': 0},
        'strata': captured['strata'], 'regions': regions, 'by_kind_and_depth': groups,
        'largest_probability_differences': [r['id'] for r in worst[:20]], 'states': rows,
        'definitions': {
            'collision_probability': 'accepted next downcrossing; GPIS rejected upcrossings reported separately',
            'weights': 'B-visit population / selected states within each region and first/later B stratum',
            'normal_statistics': 'GPIS-hit-weighted mean of per-state metrics; require >=64 hits on both sides',
            'normal_ks': 'empirical CDF distance; nonzero Monte Carlo floor, not a brightness error',
            'matched_distance': 'joint and GPIS-speed variants use exactly the GPIS accepted-hit distances',
            'no_memory': 'only transverse birth-gradient conditioning removed; paired standard-normal draws',
            'gpis_speed_variants': 'retain GPIS distance and longitudinal derivative; use analytic cell mean gradient',
            'scope': 'states sampled from neural paths; differences do not quantify the final-image causal contribution',
        },
    }
    (directory/'analysis.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    with (directory/'probabilities.csv').open('w', newline='', encoding='utf-8') as stream:
        keys = ('id', 'region', 'depth_group', 'completed_collisions', 'population_weight', 'b0', 'beta_g',
                'derivative_residual', 'joint_collision_probability', 'gpis_collision_probability',
                'collision_difference', 'gpis_rejected_probability', 'joint_escape_probability',
                'gpis_escape_probability', 'escape_difference')
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps({'replay': replay, 'regions': {k: {p: v[p] for p in (
        'states', 'joint_collision_probability', 'gpis_collision_probability', 'gpis_rejected_probability',
        'collision_probability_difference', 'escape_probability_difference',
        'outside_training_derivative_residual_fraction')} for k, v in regions.items()},
        'worst': [{k: r[k] for k in ('id', 'region', 'depth_group', 'joint_collision_probability',
                  'gpis_collision_probability', 'gpis_rejected_probability', 'derivative_residual')} for r in worst[:8]]}, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    analyze(parser.parse_args().directory)
