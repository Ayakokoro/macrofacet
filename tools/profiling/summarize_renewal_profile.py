"""Summarize raw host profiles and Nsight CSVs without conflating the clocks."""
from pathlib import Path
import collections
import csv
import hashlib
import json
import statistics
import sys

root = Path(sys.argv[1])
read = lambda name: json.loads((root / name).read_text(encoding='utf-8'))
groups = {mode: [read(f'{mode}_{i}.json')['runs'][0] for i in range(1, 4)]
          for mode in ('baseline', 'wall')}
summary = {'timing': {}, 'host_stages': {}, 'gpu_kernel_seconds': {}, 'gpu_kernel_calls': {}}
for mode, runs in groups.items():
    times = [r['seconds'] for r in runs]
    summary['timing'][mode] = {'seconds': times, 'median': statistics.median(times)}
for name in groups['wall'][0]['stages']:
    summary['host_stages'][name] = {
        'median_inclusive_seconds': statistics.median(r['stages'][name]['inclusive_seconds'] for r in groups['wall']),
        'median_exclusive_seconds': statistics.median(r['stages'][name]['exclusive_seconds'] for r in groups['wall']),
        'median_inclusive_fraction': statistics.median(r['stages'][name]['inclusive_seconds']/r['seconds'] for r in groups['wall']),
        'median_exclusive_fraction': statistics.median(r['stages'][name]['exclusive_seconds']/r['seconds'] for r in groups['wall']),
    }
gpu, calls = collections.Counter(), collections.Counter()
with (root / 'nsys_stable_stats_nvtx_kern_sum.csv').open(newline='', encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        name = row['NVTX Range'].lstrip(':')
        gpu[name] += float(row['Total Time (ns)'])/1e9
        calls[name] += int(row['Kern Inst'])
summary['gpu_kernel_seconds'] = dict(gpu)
summary['gpu_kernel_calls'] = dict(calls)
summary['trace_wall_seconds'] = read('trace_stable.json')['runs'][0]['seconds']
summary['trace_note'] = 'GPU kernel times are measured by Nsight, summed within each NVTX range. Nested ranges overlap; never add parents to children. Host timers for asynchronous module calls measure launch/dispatch, not GPU execution.'
summary['counts'] = {k: v for k, v in groups['wall'][0].items() if k not in ('stages','worker_stages','seconds')}
summary['worker_stages'] = groups['wall'][0].get('worker_stages', {})
summary['worker_note'] = 'Worker stages sum thread time and must not be added to coordinator wall time.'
summary['image_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('*.pfm')}
summary['all_diagnostic_images_identical'] = len(set(summary['image_sha256'].values())) == 1
production = root / 'production_parity/render_neural_renewal_directional.pfm'
summary['production_image_sha256'] = hashlib.sha256(production.read_bytes()).hexdigest()
summary['matches_production_exactly'] = summary['production_image_sha256'] == summary['image_sha256']['wall_3.json.pfm']
assert summary['all_diagnostic_images_identical'] and summary['matches_production_exactly']
for runs in groups.values():
    for run in runs:
        assert run['segments'] == groups['wall'][0]['segments']
        assert run['hits'] == groups['wall'][0]['hits']
        assert run['numerical_failures'] == 0 and run['safety_cap_terminations'] == 0
for run in groups['wall']:
    exclusive_sum = sum(v['exclusive_seconds'] for v in run['stages'].values())
    assert abs(exclusive_sum-run['stages']['render.measured']['inclusive_seconds']) < 1e-6
(root / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(json.dumps(summary, indent=2))
