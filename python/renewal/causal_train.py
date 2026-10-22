"""Train an A/B conditional residual GRU on packed Renewal SE trajectories."""
from dataclasses import asdict
import json
import math
from pathlib import Path
import time

import torch

from first_hit_se.causal_residual import CausalConfig
from .causal_data import ProfileDataset, batch_profiles, prepare_profile, to_device
from .causal_residual import CONDITIONING, FORMAT, VERSION, RenewalResidual, load_model, objective, prediction
from .collect import digest, resolve, write_json
from .data import load_data


def source_identity():
    root = Path(__file__).resolve().parents[1]
    names = ('renewal/causal_train.py', 'renewal/causal_data.py', 'renewal/causal_residual.py',
             'first_hit_se/causal_residual.py', 'first_hit_se/causal_data.py',
             'first_hit_se/hazard.py', 'first_hit_se/spline.py')
    return {name: digest(root/name) for name in names}


def read_spec(config_path, smoke_test=False):
    spec = json.loads(resolve(config_path).read_text(encoding='utf-8-sig'))
    if spec.get('schema_version') != 1:
        raise ValueError('unsupported residual training configuration')
    CausalConfig(**spec['model'])
    for key in ('epochs', 'profiles_per_batch', 'samples_per_profile_per_epoch', 'validation_every',
                'curve_points', 'cpu_threads', 'patience'):
        if type(spec[key]) is not int or spec[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if spec['curve_points'] < 2:
        raise ValueError('curve_points must be at least two')
    for key in ('learning_rate', 'gradient_clip'):
        if not math.isfinite(spec[key]) or spec[key] <= 0:
            raise ValueError(f'{key} must be finite and positive')
    for key in ('weight_decay', 'minimum_learning_rate'):
        if not math.isfinite(spec[key]) or spec[key] < 0:
            raise ValueError(f'{key} must be finite and nonnegative')
    if spec['minimum_learning_rate'] > spec['learning_rate']:
        raise ValueError('minimum learning rate exceeds the initial rate')
    for key in ('residual_l2_weight', 'transmittance_weight'):
        if not math.isfinite(spec['objective'][key]) or spec['objective'][key] < 0:
            raise ValueError(f'invalid objective weight {key}')
    if smoke_test:
        spec.update(epochs=2, validation_every=1, samples_per_profile_per_epoch=64,
                    curve_points=65, limit_per_mode=2, smoke_test=True,
                    output_directory=str(resolve(spec['output_directory'])/'smoke_test'))
    return spec


def prepared(spec, data, split, output, identity):
    cache_identity = {'dataset_sha256': identity['dataset_sha256'], 'model': spec['model'],
                      'curve_points': spec['curve_points'], 'limit_per_mode': spec.get('limit_per_mode'),
                      'sources': identity['sources'], 'conditioning': CONDITIONING}
    path = output/f'prepared_{split}.pt'
    if path.exists():
        cache = torch.load(path, map_location='cpu', weights_only=True)
        if cache['identity'] != cache_identity:
            raise ValueError('prepared data changed; use a new output directory')
        return cache['rows']
    rows = []
    config = CausalConfig(**spec['model'])
    for raw in data.profiles(split, spec.get('limit_per_mode')):
        rows.append(prepare_profile(raw, config, spec['curve_points']))
        if len(rows) % 50 == 0:
            print(f'prepared {split}: {len(rows)} profiles', flush=True)
    if not rows:
        raise ValueError(f'empty {split} split')
    torch.save({'identity': cache_identity, 'rows': rows}, path)
    print(f'prepared {split}: {len(rows)} profiles (complete)', flush=True)
    return rows


def profile_counts(profiles):
    return {name: {'profiles': sum(int(p['mode']) == mode for p in profiles),
                   'samples': sum(len(p['hit']) for p in profiles if int(p['mode']) == mode),
                   'hits': sum(int(p['hit'].sum()) for p in profiles if int(p['mode']) == mode)}
            for mode, name in enumerate(CONDITIONING['modes'])}


@torch.no_grad()
def evaluate_profiles(model, profiles, device, batch_size=8):
    if model is not None:
        model.eval()
    rows = []
    for start in range(0, len(profiles), batch_size):
        batch = to_device(batch_profiles(profiles[start:start+batch_size]), device)
        p = prediction(model, batch)
        T, nll = p['T'], p['distance_nll']
        if (not torch.isfinite(T).all() or not torch.isfinite(nll[batch['valid']]).all()
                or (T[:, 1:]-T[:, :-1] > 1e-10).any() or (T[:, 0]-1).abs().max() > 1e-10
                or (T < 0).any() or (T > 1+1e-10).any()):
            raise FloatingPointError('invalid conditional survival distribution')
        error = T-batch['reference_T']
        values = torch.stack(((nll.masked_fill(~batch['valid'], 0).sum(-1)/batch['valid'].sum(-1)),
                              error.abs().max(-1).values, error.square().mean(-1), error[:, -1]), -1).cpu().tolist()
        for raw, metrics in zip(profiles[start:start+batch_size], values):
            rows.append({'profile': int(raw['profile']), 'mode': int(raw['mode']),
                         **dict(zip(('distance_nll', 'max_abs_T_error', 'mse_T', 'escape_error'), metrics))})

    def summary(selected):
        if not selected:
            return {'profiles': 0}
        n = len(selected)
        return {'profiles': n, 'distance_nll': sum(p['distance_nll'] for p in selected)/n,
                'mean_max_abs_T_error': sum(p['max_abs_T_error'] for p in selected)/n,
                'max_abs_T_error': max(p['max_abs_T_error'] for p in selected),
                'mse_T': sum(p['mse_T'] for p in selected)/n,
                'mean_abs_escape_error': sum(abs(p['escape_error']) for p in selected)/n}
    return {'overall': summary(rows), 'by_mode': {name: summary([p for p in rows if p['mode'] == mode])
            for mode, name in enumerate(CONDITIONING['modes'])}}


def train(config_path, resume=False, smoke_test=False):
    spec = read_spec(config_path, smoke_test)
    torch.set_num_threads(spec['cpu_threads'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    device = spec.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    output, dataset = resolve(spec['output_directory']), resolve(spec['dataset'])
    output.mkdir(parents=True, exist_ok=True)
    identity = {'specification': spec, 'dataset_sha256': digest(dataset), 'sources': source_identity(),
                'conditioning': CONDITIONING, 'selection': 'minimum_validation_conditional_distance_nll'}
    if spec.get('dataset_sha256', identity['dataset_sha256']) != identity['dataset_sha256']:
        raise ValueError('dataset hash differs from the training configuration')
    protocol = output/'protocol.json'
    if protocol.exists() and json.loads(protocol.read_text(encoding='utf-8')) != identity:
        raise ValueError('training protocol changed; use a new output directory')
    latest = output/'last.pt'
    if latest.exists() and not resume:
        raise ValueError('training output already exists; use --resume or a new output directory')
    if resume and not latest.exists():
        raise ValueError('no last.pt checkpoint to resume')
    write_json(protocol, identity)
    packed = load_data(dataset)
    data = ProfileDataset(packed)
    splits = {split: prepared(spec, data, split, output, identity) for split in ('train', 'validation')}
    counts = {split: profile_counts(rows) for split, rows in splits.items()}
    reference = {'kernel': packed['kernel'], 'reference_sampler': packed.get('reference_sampler'),
                 'surface_derivative_sampling': packed.get('surface_derivative_sampling')}
    # Prepared labels own their storage. Release unused test labels before training.
    del packed, data
    torch.manual_seed(spec['seed'])
    model = RenewalResidual(CausalConfig(**spec['model'])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec['learning_rate'], weight_decay=spec['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, spec['epochs'],
                                                          eta_min=spec['minimum_learning_rate'])
    history, best, first, stale, elapsed = [], math.inf, 1, 0, 0.
    if resume:
        saved = torch.load(latest, map_location=device, weights_only=True)
        if saved['identity'] != identity or saved['format'] != FORMAT or saved['version'] != VERSION:
            raise ValueError('resume checkpoint identity changed')
        model.load_state_dict(saved['model_state'])
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        history, best = saved['history'], saved['best_score']
        first, stale, elapsed = saved['epoch']+1, saved['stale'], saved['elapsed_seconds']
    else:
        baseline = evaluate_profiles(None, splits['validation'], device)
        write_json(output/'baseline_validation.json', baseline)
    write_json(output/'environment.json', {'torch': str(torch.__version__), 'device': device,
               'cuda_device': torch.cuda.get_device_name(device) if str(device).startswith('cuda') else None,
               'parameters': model.active_parameters(), 'counts': counts, **reference})
    start_time = last_print = time.perf_counter()
    for epoch in range(first, spec['epochs']+1):
        if stale >= spec['patience']:
            break
        model.train()
        generator = torch.Generator().manual_seed(spec['seed']+epoch)
        order = torch.randperm(len(splits['train']), generator=generator).tolist()
        totals, count = {k: 0. for k in ('loss', 'nll', 'residual_l2', 'transmittance_mse')}, 0
        for begin in range(0, len(order), spec['profiles_per_batch']):
            profiles = [splits['train'][i] for i in order[begin:begin+spec['profiles_per_batch']]]
            indices = [torch.randperm(len(p['hit']), generator=generator)[:spec['samples_per_profile_per_epoch']]
                       for p in profiles]
            batch = to_device(batch_profiles(profiles, indices), device)
            optimizer.zero_grad(set_to_none=True)
            terms = objective(model, batch, spec['objective'])
            if not torch.isfinite(terms['loss']):
                raise FloatingPointError('nonfinite conditional residual training objective')
            terms['loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), spec['gradient_clip'], error_if_nonfinite=True)
            optimizer.step()
            for key in totals:
                totals[key] += float(terms[key].detach())*len(profiles)
            count += len(profiles)
            if time.perf_counter()-last_print >= 30:
                print(f'epoch {epoch}: {count}/{len(order)} profiles', flush=True)
                last_print = time.perf_counter()
        scheduler.step()
        metrics = None
        if epoch == 1 or epoch % spec['validation_every'] == 0 or epoch == spec['epochs']:
            metrics = evaluate_profiles(model, splits['validation'], device)
        seconds = elapsed+time.perf_counter()-start_time
        bundle = {'format': FORMAT, 'version': VERSION, 'conditioning': CONDITIONING, 'identity': identity,
                  'model_config': asdict(model.config), 'model_state': model.state_dict(),
                  'epoch': epoch, 'validation': metrics, 'reference': reference}
        if metrics is not None:
            score = metrics['overall']['distance_nll']
            if score < best:
                best, stale = score, 0
                torch.save(bundle, output/'best.pt')
            else:
                stale += 1
            print(f'epoch {epoch}: validation conditional NLL={score:.6f}, best={best:.6f}', flush=True)
            last_print = time.perf_counter()
        history.append({'epoch': epoch, 'elapsed_seconds': seconds, 'validation': metrics,
                        **{'train_'+k: v/count for k, v in totals.items()}})
        torch.save({**bundle, 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                    'best_score': best, 'history': history, 'stale': stale, 'elapsed_seconds': seconds}, latest)
        write_json(output/'history.json', history)
    _, chosen = load_model(output/'best.pt', 'cpu')
    report = {'checkpoint': str(output/'best.pt'), 'checkpoint_sha256': digest(output/'best.pt'),
              'selected_epoch': chosen['epoch'], 'validation': chosen['validation'], 'counts': counts,
              'epochs_completed': history[-1]['epoch'], 'training_seconds': history[-1]['elapsed_seconds'],
              'test_metrics_used': False, 'smoke_test': spec.get('smoke_test', False)}
    write_json(output/'training.json', report)
    return report


def evaluate(checkpoint, dataset, output, split='test', device='cpu'):
    model, saved = load_model(resolve(checkpoint), device)
    spec = saved['identity']['specification']
    torch.set_num_threads(spec['cpu_threads'])
    dataset = resolve(dataset)
    if digest(dataset) != saved['identity']['dataset_sha256']:
        raise ValueError('evaluation dataset differs from the checkpoint dataset')
    if source_identity() != saved['identity']['sources']:
        raise ValueError('evaluation implementation differs from the training checkpoint')
    data = ProfileDataset(load_data(dataset))
    rows = prepared(spec, data, split, resolve(checkpoint).parent, saved['identity'])
    report = {'split': split, 'checkpoint_sha256': digest(resolve(checkpoint)),
              'smoke_test': spec.get('smoke_test', False), 'counts': profile_counts(rows),
              **evaluate_profiles(model, rows, device)}
    write_json(resolve(output), report)
    return report
