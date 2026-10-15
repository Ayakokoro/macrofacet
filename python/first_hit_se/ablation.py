"""Reproducible, paired continuation experiments for loss and state capacity."""
from copy import deepcopy
from pathlib import Path
import subprocess
import sys

import torch

from renewal.collect import digest, resolve, write_json
from .data import read_json, load_data
from .model import FirstHitNetwork, ModelConfig
from .train import load_model


def prepare(config_path):
    spec = read_json(config_path)
    root = resolve(spec['output_directory'])
    root.mkdir(parents=True, exist_ok=True)
    runs = []
    torch.set_num_threads(1)
    original, _ = load_model(resolve(spec['common']['initialize_from']))
    data = load_data(resolve(spec['common']['dataset']))
    # Only training inputs are used to verify the initialization transform.
    training = [p for p in data['profiles'] if p['split'] == 'train']
    probes = [training[0], max(training, key=lambda p: p['horizon'])]
    with torch.no_grad():
        queries = [torch.linspace(0, p['horizon'], 17, dtype=torch.float64) for p in probes]
        reference = [original.query(p['features'], q) for p, q in zip(probes, queries)]
        for seed in spec['seeds']:
            for variant in spec['variants']:
                name = f"{variant['name']}_s{seed}"
                config = deepcopy(spec['common'])
                config.update(seed=seed, curve_loss_weight=variant['curve_loss_weight'],
                              output_directory=f"{spec['model_directory']}/{name}")
                config['model']['hidden'] = variant['hidden']
                config['expand_hidden_state'] = variant['hidden'] != original.config.hidden
                torch.manual_seed(seed)
                initialized = FirstHitNetwork(ModelConfig(**config['model']))
                if config['expand_hidden_state']:
                    initialized.expand_hidden_from(original)
                else:
                    initialized.load_state_dict(original.state_dict())
                maximum = 0.
                for p, q, expected in zip(probes, queries, reference):
                    actual = initialized.query(p['features'], q)
                    maximum = max(maximum, *(float((actual[k]-expected[k]).abs().max()) for k in expected))
                if maximum > 1e-5:
                    raise ValueError(f'initialization changed predictions for {name}: {maximum}')
                path = root/'configs'/f'{name}.json'
                if path.exists() and read_json(path) != config:
                    raise ValueError(f'existing ablation configuration differs: {path}')
                write_json(path, config)
                runs.append({'name': name, 'variant': variant['name'], 'seed': seed,
                             'config': str(path), 'directory': str(resolve(config['output_directory'])),
                             'parameters': sum(p.numel() for p in initialized.parameters()),
                             'initial_max_abs_prediction_difference': maximum})
    experiment = {'specification': spec, 'specification_sha256': digest(config_path),
                  'dataset_sha256': digest(resolve(spec['common']['dataset'])),
                  'initial_checkpoint_sha256': digest(resolve(spec['common']['initialize_from'])),
                  'baseline_sha256': {k: digest(resolve(v)) for k, v in spec['baselines'].items()},
                  'scope': 'Paired continuation from the same v2 function; not a from-scratch capacity sweep.',
                  'curve_target': '33 uniform queries including endpoints; all stored training realizations',
                  'primary_selection': 'Last epoch at the same fixed update budget; no test-based selection',
                  'runs': runs}
    manifest = root/'experiment.json'
    if manifest.exists() and read_json(manifest) != experiment:
        raise ValueError('ablation manifest changed; use a new output directory')
    write_json(manifest, experiment)
    return experiment


def run(config_path, prepare_only=False):
    experiment = prepare(config_path)
    if prepare_only:
        return {'runs': len(experiment['runs']), 'initial_max_abs_prediction_difference':
                max(r['initial_max_abs_prediction_difference'] for r in experiment['runs'])}
    root = resolve(experiment['specification']['output_directory'])
    (root/'logs').mkdir(exist_ok=True)
    cli = Path(__file__).resolve().parents[1]/'first_hit_se_cli.py'
    for i, record in enumerate(experiment['runs'], 1):
        directory = Path(record['directory'])
        expected = read_json(record['config'])
        report = directory/'run_report.json'
        if report.exists():
            saved = read_json(report)
            if (read_json(directory/'training_config.json') != expected or
                    saved['dataset_sha256'] != experiment['dataset_sha256']):
                raise ValueError(f'completed run does not match experiment: {record["name"]}')
            if saved['completed_epochs'] == expected['epochs']:
                print(f'SKIP {i}/{len(experiment["runs"])} {record["name"]}: complete', flush=True)
                continue
        command = [sys.executable, '-u', str(cli), 'train', '--config', record['config']]
        if (directory/'last.pt').exists():
            command.append('--resume')
        print(f'START {i}/{len(experiment["runs"])} {record["name"]}', flush=True)
        with (root/'logs'/f'{record["name"]}.log').open('a', encoding='utf-8') as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
        saved = read_json(report)
        if saved['completed_epochs'] != expected['epochs']:
            raise ValueError('run did not complete the fixed epoch budget')
        print(f'DONE {i}/{len(experiment["runs"])} {record["name"]}: '
              f'{saved["optimizer_updates"]} updates, {saved["training_seconds"]:.1f} seconds', flush=True)
    return {'completed_runs': len(experiment['runs']), 'output': str(root)}
