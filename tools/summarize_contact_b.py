"""Build a compact report, including optional full-ray and GPIS-step probes."""
import argparse
import csv
import json
from pathlib import Path


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def summarize(directory):
    directory = Path(directory)
    analysis = load(directory/'analysis.json')
    captured = load(directory/'capture.json')
    states = {s['id']: s for s in captured['states']}
    results = {s['id']: s for s in load(directory/'comparison.json')['states']}
    probes = {}
    for path in sorted(directory.glob('gpis_*_probe')):
        if not (path/'comparison.json').exists():
            continue
        req = load(path/'request.json')
        source = {s['id']: s['source_state_id'] for s in load(path/'capture.json')['states']}
        probes[str(req['gpis_step_size'])] = {
            source[s['id']]: s for s in load(path/'comparison.json')['states']}
    reviewed = []
    for path in directory.glob('state_*_reference/first_passage_summary.csv'):
        index = int(path.parent.name.split('_')[1]); state = states[index]
        with path.open(newline='', encoding='utf-8') as stream:
            data = list(csv.DictReader(stream))
        fine = next(row for row in data if row['training_resolution'] == '1')
        row = {
            'id': index, 'region': state['region'], 'pixel': state['pixel'],
            'camera_sample': state['camera_sample'], 'completed_collisions': state['completed_collisions'],
            'joint_collision_probability': 1-results[index]['predicted_escape'],
            'full_ray_reference_collision_probability': float(fine['event_probability']),
            'full_ray_reference_trials': int(fine['trajectories']),
            'reference_probabilities_coarse_fine': [float(r['event_probability']) for r in data],
            'reference_median_hit_world_distance': float(fine['event_q50'])*state['ell'],
            'gpis_by_world_step': {},
        }
        for step, r in {'0.01': results[index], **{step: r[index] for step, r in probes.items() if index in r}}.items():
            row['gpis_by_world_step'][step] = {
                name: r['gpis_counts'][i]/r['trials']
                for i, name in enumerate(('escape', 'accepted_hit', 'rejected_upcrossing', 'invalid_gradient'))}
        reviewed.append(row)
    reviewed.sort(key=lambda r: r['id'])
    regions = {}
    for name, region in analysis['regions'].items():
        regions[name] = {key: region[key] for key in (
            'states', 'represented_B_visits', 'joint_collision_probability', 'gpis_collision_probability',
            'gpis_rejected_probability', 'joint_escape_probability', 'gpis_escape_probability',
            'escape_probability_difference', 'mean_abs_collision_probability_difference')}
        regions[name]['normal_cosine_ks_at_gpis_distances'] = {
            variant: region['normals'][variant].get('cos_incidence_ks') for variant in (
                'joint_at_gpis_distance', 'joint_at_gpis_distance_no_memory',
                'gpis_speed_full_memory', 'gpis_speed_no_memory')}
        regions[name]['longitudinal_speed_ks_at_gpis_distances'] = region['normals']['joint_at_gpis_distance'].get('speed_ks')
    report = {
        'replay': analysis['replay'], 'experiment_checks': analysis.get('experiment_checks'),
        'camera_paths': sum(s['camera_paths'] for s in captured['strata'] if s['depth_group'] == 'first_B'),
        'captured_B_visits': sum(s['population'] for s in captured['strata']),
        'selected_B_states': len(states), 'trials_per_sampler_per_state': analysis['trials_per_state'],
        'regions': regions, 'selected_full_ray_checks': reviewed,
        'interpretation': {
            'probabilities': 'Region averages describe stratified B states from neural paths, weighted by B-visit frequency.',
            'upcrossings': 'GPIS rejects these outcomes with zero path radiance; they are not escapes or accepted hits.',
            'normal_cosine_ks': 'GPIS-hit-weighted mean per-state empirical CDF difference; smaller means closer, not a brightness percentage.',
            'reference_checks': 'Eight states selected after inspecting discrepancies; these are diagnostic counterexamples, not unbiased regional estimates.',
            'step_probe': 'Smaller GPIS step changes both crossing detection and renewal frequency; it is not a full-history GP convergence guarantee.',
            'image_scope': 'No full-render component replacement was performed, so brightness contributions are not quantified.',
        },
    }
    (directory/'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (directory/'full_ray_checks.json').write_text(json.dumps(reviewed, indent=2), encoding='utf-8')
    with (directory/'region_summary.csv').open('w', newline='', encoding='utf-8') as stream:
        keys = ('region', 'joint_collision_probability', 'gpis_collision_probability', 'gpis_rejected_probability',
                'joint_escape_probability', 'gpis_escape_probability', 'escape_probability_difference')
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction='ignore')
        writer.writeheader()
        writer.writerows({'region': name, **r} for name, r in regions.items())
    print(json.dumps({'output': str(directory/'summary.json'), 'full_ray_checks': len(reviewed),
                      'experiment_checks': report['experiment_checks']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    summarize(parser.parse_args().directory)
