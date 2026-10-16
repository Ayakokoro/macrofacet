"""Controlled local-head / causal-GRU comparison on the fixed sigma=.02 data."""
from pathlib import Path
import time

import numpy as np
import torch

from renewal.collect import digest, resolve, write_json
from .causal_data import batch_profiles, prepare_profile, to_device
from .causal_residual import CausalConfig, CausalResidual, objective, prediction
from .data import load_data, read_json
from .hazard_study import parameter_hash
from .multiview_study import test_groups
from .residual_supervision import ray_metrics


def model_config(spec, variant):
    return CausalConfig(**spec['model'], history=variant['history'])


def synchronize(device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()


def register(spec, output):
    sources = [Path(__file__).parent/name for name in ('causal_study.py', 'causal_data.py', 'causal_residual.py')]
    old = read_json(resolve(spec['supervision_directory'])/'evaluation.json')
    reference = {k: v for k, v in old['checkpoint_sha256'].items()
                 if 'l2_0p01_T_0' not in k}
    for path, expected in reference.items():
        if digest(Path(path)) != expected:
            raise ValueError('baseline checkpoint changed')
    identity = {'specification': spec, 'dataset_sha256': digest(resolve(spec['dataset'])),
        'training_source_sha256': {str(p): digest(p) for p in sources},
        'baseline_checkpoint_sha256': reference,
        'baseline_evaluation_sha256': digest(resolve(spec['supervision_directory'])/'evaluation.json'),
        'selection': 'all variants fixed in advance; minimum validation marginal NLL; test only after training is frozen'}
    if identity['dataset_sha256'] != spec['dataset_sha256']:
        raise ValueError('fixed dataset changed')
    if spec['selection'] != 'minimum_validation_marginal_distance_nll':
        raise ValueError('unsupported checkpoint selection')
    path = output/'protocol.json'
    if path.exists() and read_json(path) != identity:
        raise ValueError('registered inputs or training implementation changed; use a new output directory')
    write_json(path, identity)
    return identity


def prepared(spec, data, split, output):
    path = output/f'prepared_{split}.pt'
    identity = {'dataset_sha256': spec['dataset_sha256'], 'model': spec['model'],
                'curve_points': spec['curve_points'], 'source_sha256': digest(Path(__file__).with_name('causal_data.py'))}
    if path.exists():
        cache = torch.load(path, weights_only=True)
        if cache['identity'] != identity:
            raise ValueError('prepared geometry has a different identity')
        rows = cache['rows']
    else:
        rows = []
        config = CausalConfig(**spec['model'])
        raw = [p for p in data['profiles'] if p['split'] == split]
        for i, profile in enumerate(raw):
            rows.append(prepare_profile(profile, config, spec['curve_points']))
            if (i+1) % 100 == 0 or i+1 == len(raw):
                print(f'prepared {split}: {i+1}/{len(raw)} rays', flush=True)
        torch.save({'identity': identity, 'rows': rows}, path)
    return [to_device(row, spec['device']) for row in rows]


@torch.no_grad()
def evaluate(model, profiles, curves=False, batch_size=4):
    if model is not None:
        model.eval()
    rows = []
    for begin in range(0, len(profiles), batch_size):
        batch = batch_profiles(profiles[begin:begin+batch_size])
        pred = prediction(model, batch)
        T = pred['T']
        if (not torch.isfinite(T).all() or not torch.isfinite(pred['distance_nll']).all()
                or (T[:, 1:]-T[:, :-1] > 1e-12).any() or (T[:, 0]-1).abs().max() > 1e-10):
            raise FloatingPointError('invalid causal probability distribution')
        for j in range(len(T)):
            error = T[j]-batch['reference_T'][j]
            rows.append({'max_abs_T_error': float(error.abs().max()), 'mse_T': float(error.square().mean()),
                'escape_error': float(error[-1]), 'distance_nll': float(pred['distance_nll'][j].mean()),
                **({'T': T[j].cpu().tolist()} if curves else {})})
    return {**ray_metrics(rows), 'per_ray': rows}


def load_run(path, device='cpu'):
    saved = torch.load(path, weights_only=True, map_location=device)
    if saved['format'] != 'first_hit_causal_residual':
        raise ValueError('not a causal residual checkpoint')
    model = CausalResidual(CausalConfig(**saved['model_config'])).to(device)
    model.load_state_dict(saved['model_state'])
    model.eval()
    return model, saved


def train_one(spec, variant, seed, splits, output):
    directory = output/'runs'/f"{variant['name']}_{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    identity = {'specification': spec, 'variant': variant, 'seed': seed,
                'protocol_sha256': digest(output/'protocol.json')}
    report_path = directory/'run_report.json'
    if report_path.exists():
        report = read_json(report_path)
        if report['identity'] != identity:
            raise ValueError('completed run has a different identity')
        return report
    torch.manual_seed(seed)
    model = CausalResidual(model_config(spec, variant)).to(spec['device'])
    initial_hash = parameter_hash(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec['learning_rate'], weight_decay=spec['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, spec['epochs'], eta_min=spec['minimum_learning_rate'])
    weights = {k: variant[k] for k in ('residual_l2_weight', 'transmittance_weight')}
    history, best, first, elapsed = [], float('inf'), 1, 0.
    latest = directory/'latest.pt'
    if latest.exists():
        saved = torch.load(latest, weights_only=True, map_location=spec['device'])
        if saved['identity'] != identity:
            raise ValueError('resume identity changed')
        model.load_state_dict(saved['model_state'])
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        history, best, first, elapsed = saved['history'], saved['best_score'], saved['epoch']+1, saved['elapsed']
    synchronize(spec['device'])
    start = time.perf_counter()
    for epoch in range(first, spec['epochs']+1):
        model.train()
        generator = torch.Generator().manual_seed(seed+epoch)
        order = torch.randperm(len(splits['train']), generator=generator).tolist()
        sums, count = {k: 0. for k in ('loss', 'nll', 'residual_l2', 'transmittance_mse')}, 0
        for begin in range(0, len(order), spec['profiles_per_batch']):
            profiles = [splits['train'][i] for i in order[begin:begin+spec['profiles_per_batch']]]
            indices = [torch.randperm(len(p['hit']), generator=generator)[:spec['samples_per_profile_per_epoch']].to(spec['device']) for p in profiles]
            batch = batch_profiles(profiles, indices)
            optimizer.zero_grad(set_to_none=True)
            terms = objective(model, batch, weights)
            if not torch.isfinite(terms['loss']):
                raise FloatingPointError('nonfinite causal training objective')
            terms['loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), spec['gradient_clip'], error_if_nonfinite=True)
            optimizer.step()
            for key in sums:
                sums[key] += float(terms[key].detach())*len(profiles)
            count += len(profiles)
        scheduler.step()
        if epoch % spec['validation_every'] and epoch != spec['epochs']:
            continue
        validation = evaluate(model, splits['validation'])
        validation = {k: v for k, v in validation.items() if k != 'per_ray'}
        synchronize(spec['device'])
        seconds = elapsed+time.perf_counter()-start
        bundle = {'format': 'first_hit_causal_residual', 'identity': identity,
            'model_config': model.configuration(), 'model_state': model.state_dict(),
            'epoch': epoch, 'validation': validation}
        if validation['distance_nll'] < best:
            best = validation['distance_nll']
            torch.save(bundle, directory/'best.pt')
        history.append({'epoch': epoch, 'elapsed_seconds': seconds, 'validation': validation,
                        **{'train_'+k: v/count for k, v in sums.items()}})
        torch.save({**bundle, 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                    'best_score': best, 'history': history, 'elapsed': seconds}, latest)
        write_json(directory/'history.json', history)
        if epoch % 50 == 0 or epoch == spec['epochs']:
            print(f"{variant['name']} seed={seed} epoch={epoch}: val NLL={best:.6f} (best), current mean max T={validation['mean_ray_max_abs_T_error']:.5f}, {seconds:.1f}s", flush=True)
    model, chosen = load_run(directory/'best.pt', spec['device'])
    report = {'identity': identity, 'initial_parameter_sha256': initial_hash, 'selected_epoch': chosen['epoch'],
        'training_seconds': history[-1]['elapsed_seconds'], 'parameters': sum(p.numel() for p in model.parameters()),
        'active_parameters': model.active_parameters(), 'validation': chosen['validation'],
        'checkpoint': str(directory/'best.pt'), 'checkpoint_sha256': digest(directory/'best.pt')}
    write_json(report_path, report)
    return report


def train(spec, data, output):
    splits = {split: prepared(spec, data, split, output) for split in ('train', 'validation')}
    runs = []
    for seed in spec['seeds']:
        for variant in spec['variants']:
            print(f"training {variant['name']}, seed={seed}", flush=True)
            runs.append(train_one(spec, variant, seed, splits, output))
        if len({r['initial_parameter_sha256'] for r in runs if r['identity']['seed'] == seed}) != 1:
            raise AssertionError('paired initialization differs')
    result = {'protocol_sha256': digest(output/'protocol.json'), 'runs': runs,
              'test_metrics_used': False, 'checkpoint_selection': spec['selection']}
    write_json(output/'training.json', result)
    return result


def evaluate_selected(spec, data, output):
    trained = read_json(output/'training.json')
    if trained['protocol_sha256'] != digest(output/'protocol.json') or trained['test_metrics_used']:
        raise ValueError('training selection changed')
    profiles = prepared(spec, data, 'test', output)
    old = read_json(resolve(spec['supervision_directory'])/'evaluation.json')
    results = {'local_mlp_grid': old['results']['local'], 'mlp_nll': old['results']['baseline'],
               'mlp_T': old['results']['regularized_T'], 'local_causal_grid': [{'seed': None, 'metrics': evaluate(None, profiles, True)}]}
    hashes = {}
    for variant in spec['variants']:
        results[variant['name']] = []
        for seed in spec['seeds']:
            record = next(r for r in trained['runs'] if r['identity']['seed'] == seed and r['identity']['variant'] == variant)
            path = Path(record['checkpoint'])
            if digest(path) != record['checkpoint_sha256']:
                raise ValueError('selected checkpoint changed')
            hashes[str(path)] = digest(path)
            model, _ = load_run(path, spec['device'])
            score = evaluate(model, profiles, True)
            results[variant['name']].append({'seed': seed, 'metrics': score})
            print(f"fixed test {variant['name']}/{seed}: mean max T={score['mean_ray_max_abs_T_error']:.6f}, tail10={score['worst_decile_mean']:.6f}, worst={score['max_abs_T_error']:.6f}", flush=True)
    groups, summary = test_groups(data), {}
    for name, indices in groups.items():
        summary[name] = {}
        for method, runs in results.items():
            values = [ray_metrics([r['metrics']['per_ray'][i] for i in indices]) for r in runs]
            summary[name][method] = {k: {'mean': float(np.mean([r[k] for r in values])),
                'seed_std': float(np.std([r[k] for r in values]))} for k in values[0] if k != 'rays'}
    result = {'training_sha256': digest(output/'training.json'), 'dataset_sha256': spec['dataset_sha256'],
              'checkpoint_sha256': hashes, 'results': results, 'summary': summary, 'groups': groups}
    write_json(output/'evaluation.json', result)
    return result


def run(config_path, phase='all'):
    spec = read_json(resolve(config_path))
    output = resolve(spec['output_directory'])
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(spec['cpu_threads'])
    torch.backends.cuda.matmul.allow_tf32 = False
    register(spec, output)
    data = load_data(resolve(spec['dataset']))
    if phase in ('all', 'train'):
        trained = train(spec, data, output)
        if phase == 'train':
            return {'runs': len(trained['runs']), 'training': str(output/'training.json')}
    if phase in ('all', 'evaluate'):
        result = evaluate_selected(spec, data, output)
        if phase == 'evaluate':
            return result['summary']['all_280']
    if phase in ('all', 'diagnose'):
        from .causal_diagnostics import diagnose
        result = diagnose(spec, data, output)
        if phase == 'diagnose':
            return result['summary']
    if phase in ('all', 'report'):
        from .causal_report import report
        return report(spec, data, output)
    raise ValueError(phase)
