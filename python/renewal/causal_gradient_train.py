"""Fit a hit-gradient head, freezing the selected A/B distance checkpoint."""
import json
import math
from pathlib import Path
import time

import torch

from first_hit_se.causal_gradient import GradientConfig
from first_hit_se.distance_probe import ks
from first_hit_se.gradient_cdf import cdf, log_prob
from first_hit_se.hazard_study import parameter_hash
from .causal_data import ProfileDataset
from .causal_gradient import (FORMAT, VERSION, TARGET, HitGradient, cache_profile, load_joint,
                             loss, points, raw_profiles, to_device)
from .causal_residual import CONDITIONING, load_model as load_backbone
from .causal_train import profile_counts, source_identity as distance_sources
from .collect import digest, resolve, write_json
from .data import load_data


def source_identity():
    result = distance_sources()
    root = Path(__file__).resolve().parents[1]
    for name in ('renewal/causal_gradient.py', 'renewal/causal_gradient_train.py',
                 'first_hit_se/causal_gradient.py', 'first_hit_se/gradient_cdf.py',
                 'first_hit_se/distance_probe.py', 'renewal/model.py'):
        result[name] = digest(root/name)
    return result


def read_spec(config_path, smoke_test=False):
    spec = json.loads(resolve(config_path).read_text(encoding='utf-8-sig'))
    if spec.get('schema_version') != 1:
        raise ValueError('unsupported gradient training configuration')
    for key in ('epochs', 'patience', 'cpu_threads', 'profiles_per_batch',
                'samples_per_profile_per_epoch', 'validation_every'):
        if type(spec[key]) is not int or spec[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    for key in ('learning_rate', 'gradient_clip'):
        if not math.isfinite(spec[key]) or spec[key] <= 0:
            raise ValueError(f'{key} must be finite and positive')
    for key in ('minimum_learning_rate', 'weight_decay'):
        if not math.isfinite(spec[key]) or spec[key] < 0:
            raise ValueError(f'{key} must be finite and nonnegative')
    if spec['minimum_learning_rate'] > spec['learning_rate']:
        raise ValueError('minimum learning rate exceeds the initial rate')
    if smoke_test:
        spec.update(epochs=2, validation_every=1, samples_per_profile_per_epoch=64,
                    limit_per_mode=2, smoke_test=True,
                    backbone_checkpoint=spec['smoke_backbone_checkpoint'],
                    output_directory=str(resolve(spec['output_directory'])/'smoke_test'))
    return spec


def prepared(spec, dataset, backbone, split, output, identity):
    cache_id = {'dataset_sha256': identity['dataset_sha256'],
                'backbone_sha256': identity['backbone_sha256'], 'sources': identity['sources'],
                'limit_per_mode': spec.get('limit_per_mode'), 'split': split}
    path = output/f'contexts_{split}.pt'
    if path.exists():
        cache = torch.load(path, map_location='cpu', weights_only=True)
        if cache['identity'] != cache_id:
            raise ValueError('gradient contexts changed; use a new output directory')
        return cache['rows']
    before = parameter_hash(backbone)
    rows = []
    for raw in raw_profiles(dataset, split, spec.get('limit_per_mode')):
        rows.append(cache_profile(backbone, raw))
        if len(rows) % 50 == 0:
            print(f'cached hit contexts {split}: {len(rows)} profiles', flush=True)
    if not rows:
        raise ValueError(f'empty {split} split')
    if parameter_hash(backbone) != before or any(p.requires_grad for p in backbone.parameters()):
        raise AssertionError('distance backbone changed while caching')
    torch.save({'identity': cache_id, 'rows': rows}, path)
    print(f'cached hit contexts {split}: {len(rows)} profiles (complete)', flush=True)
    return rows


@torch.no_grad()
def evaluate_profiles(head, profiles, device='cpu', calibration=True, batch_size=8):
    head.eval()
    groups = {mode: {'profiles': 0, 'samples': 0, 'hits': 0, 'distance': 0., 'gradient': 0., 'pit': []}
              for mode in (0, 1)}
    for start in range(0, len(profiles), batch_size):
        rows = profiles[start:start+batch_size]
        batch = to_device(points(rows), device)
        if len(batch['w']):
            params = head(batch)
            losses = -log_prob('mixture', params, batch['w'])
            values = cdf('mixture', params, batch['w']) if calibration else None
            if not torch.isfinite(losses).all():
                raise FloatingPointError('nonfinite conditional hit-gradient likelihood')
            if values is not None and (not torch.isfinite(values).all() or (values < 0).any() or (values > 1).any()):
                raise FloatingPointError('invalid conditional hit-gradient CDF')
        else:
            losses, values = batch['w'], None
        for mode, group in groups.items():
            selected = batch['mode'] == mode
            group['hits'] += int(selected.sum())
            group['gradient'] += float(losses[selected].sum())
            if values is not None:
                group['pit'].append(values[selected].cpu())
        for row in rows:
            group = groups[int(row['mode'])]
            group['profiles'] += 1
            group['samples'] += len(row['hit'])
            group['distance'] += float(row['distance_nll_sum'])

    def summarize(g):
        n, h = g['samples'], g['hits']
        result = {k: g[k] for k in ('profiles', 'samples', 'hits')}
        result.update(distance_nll=g['distance']/n if n else None,
                      joint_nll=(g['distance']+g['gradient'])/n if n else None,
                      gradient_term_per_realization=g['gradient']/n if n else None,
                      speed_nll=g['gradient']/h if h else None)
        if calibration:
            pit = torch.cat(g['pit']) if g['pit'] else torch.empty(0)
            result.update(pit_ks=ks(pit),
                          coverage_90=float(((pit >= .05) & (pit <= .95)).double().mean()) if h else None,
                          pit_counts=torch.histc(pit, bins=20, min=0, max=1).long().tolist())
        return result
    overall = {key: sum(g[key] for g in groups.values()) for key in ('profiles', 'samples', 'hits', 'distance', 'gradient')}
    overall['pit'] = [p for g in groups.values() for p in g['pit']]
    return {'overall': summarize(overall), 'by_mode': {
        name: summarize(groups[mode]) for mode, name in enumerate(CONDITIONING['modes'])}}


def train(config_path, resume=False, smoke_test=False):
    spec = read_spec(config_path, smoke_test)
    torch.set_num_threads(spec['cpu_threads'])
    torch.backends.cuda.matmul.allow_tf32 = False
    device = spec.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    path = resolve(spec['backbone_checkpoint'])
    if not path.exists():
        raise FileNotFoundError(f'distance checkpoint missing: {path}; run train-residual first')
    backbone, distance_saved = load_backbone(path, 'cpu')
    backbone.requires_grad_(False)
    if not backbone.config.history:
        raise ValueError('hit-gradient training requires a recurrent backbone')
    dataset_path = resolve(spec['dataset'])
    dataset_hash = digest(dataset_path)
    if (dataset_hash != distance_saved['identity']['dataset_sha256']
            or spec.get('dataset_sha256', dataset_hash) != dataset_hash):
        raise ValueError('gradient and distance datasets differ')
    if distance_saved['identity']['sources'] != distance_sources():
        raise ValueError('frozen distance implementation changed')
    head_config = GradientConfig(hidden=backbone.config.hidden, **spec['head'])
    head = HitGradient(head_config, spec['seed']).to(device)
    before = parameter_hash(backbone)
    identity = {'specification': spec, 'dataset_sha256': dataset_hash, 'backbone_sha256': digest(path),
                'backbone_parameter_sha256': before, 'sources': source_identity(), 'conditioning': CONDITIONING,
                'target': TARGET, 'selection': 'minimum_validation_conditional_speed_nll'}
    output = resolve(spec['output_directory'])
    output.mkdir(parents=True, exist_ok=True)
    protocol, latest = output/'protocol.json', output/'last.pt'
    if protocol.exists() and json.loads(protocol.read_text(encoding='utf-8')) != identity:
        raise ValueError('gradient training protocol changed; use a new output directory')
    if latest.exists() and not resume:
        raise ValueError('gradient output already exists; use --resume or a new output directory')
    if resume and not latest.exists():
        raise ValueError('no gradient last.pt to resume')
    write_json(protocol, identity)
    dataset = ProfileDataset(load_data(dataset_path))
    splits = {split: prepared(spec, dataset, backbone, split, output, identity) for split in ('train', 'validation')}
    del dataset
    if any(not sum(int(p['hit'].sum()) for p in rows) for rows in splits.values()):
        raise ValueError('gradient training and validation each require hit labels')
    counts = {split: profile_counts(rows) for split, rows in splits.items()}
    optimizer = torch.optim.AdamW(head.parameters(), lr=spec['learning_rate'], weight_decay=spec['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, spec['epochs'], eta_min=spec['minimum_learning_rate'])
    history, best, first, stale, elapsed = [], math.inf, 1, 0, 0.
    if resume:
        saved = torch.load(latest, map_location=device, weights_only=True)
        if saved['identity'] != identity or saved['format'] != FORMAT or saved['target'] != TARGET:
            raise ValueError('gradient resume identity changed')
        head.load_state_dict(saved['head_state'])
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        history, best = saved['history'], saved['best_score']
        first, stale, elapsed = saved['epoch']+1, saved['stale'], saved['elapsed_seconds']
    write_json(output/'environment.json', {'torch': str(torch.__version__), 'device': device,
               'cuda_device': torch.cuda.get_device_name(device) if str(device).startswith('cuda') else None,
               'gradient_parameters': sum(p.numel() for p in head.parameters()), 'counts': counts,
               'backbone_frozen': True, 'backbone_sha256': identity['backbone_sha256']})
    began = last_print = time.perf_counter()
    for epoch in range(first, spec['epochs']+1):
        if stale >= spec['patience']:
            break
        head.train()
        generator = torch.Generator().manual_seed(spec['seed']+epoch)
        order = torch.randperm(len(splits['train']), generator=generator).tolist()
        total, samples, hits = 0., 0, 0
        for begin in range(0, len(order), spec['profiles_per_batch']):
            rows = [splits['train'][i] for i in order[begin:begin+spec['profiles_per_batch']]]
            indices = [torch.randperm(len(p['hit']), generator=generator)[:spec['samples_per_profile_per_epoch']]
                       for p in rows]
            batch = to_device(points(rows, indices), device)
            samples += batch['samples']
            hits += len(batch['w'])
            optimizer.zero_grad(set_to_none=True)
            term = loss(head, batch)
            if term is not None:
                if not torch.isfinite(term):
                    raise FloatingPointError('nonfinite hit-gradient objective')
                term.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), spec['gradient_clip'], error_if_nonfinite=True)
                optimizer.step()
                total += float(term.detach())*batch['samples']
            if time.perf_counter()-last_print >= 30:
                print(f'gradient epoch {epoch}: {min(begin+len(rows), len(order))}/{len(order)} profiles', flush=True)
                last_print = time.perf_counter()
        if not hits:
            raise ValueError('epoch sampled no hits; increase samples_per_profile_per_epoch')
        scheduler.step()
        metrics = None
        if epoch == 1 or epoch % spec['validation_every'] == 0 or epoch == spec['epochs']:
            metrics = evaluate_profiles(head, splits['validation'], device, calibration=False)
        seconds = elapsed+time.perf_counter()-began
        bundle = {'format': FORMAT, 'version': VERSION, 'conditioning': CONDITIONING, 'target': TARGET,
                  'identity': identity, 'backbone_config': backbone.configuration(), 'backbone_state': backbone.state_dict(),
                  'backbone_identity': distance_saved['identity'], 'reference': distance_saved['reference'],
                  'head_config': head.configuration(), 'head_state': head.state_dict(), 'epoch': epoch, 'validation': metrics}
        if metrics is not None:
            score = metrics['overall']['speed_nll']
            if score < best:
                best, stale = score, 0
                torch.save(bundle, output/'best.pt')
            else:
                stale += 1
            print(f'gradient epoch {epoch}: validation speed NLL={score:.6f}, best={best:.6f}', flush=True)
            last_print = time.perf_counter()
        history.append({'epoch': epoch, 'elapsed_seconds': seconds, 'train_gradient_term_per_realization': total/samples,
                        'sampled_hits': hits, 'sampled_trajectories': samples, 'validation': metrics})
        torch.save({**bundle, 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                    'history': history, 'best_score': best, 'stale': stale, 'elapsed_seconds': seconds}, latest)
        write_json(output/'history.json', history)
    if parameter_hash(backbone) != before or any(p.grad is not None or p.requires_grad for p in backbone.parameters()):
        raise AssertionError('distance backbone changed during gradient training')
    _, selected, chosen = load_joint(output/'best.pt', device)
    validation = evaluate_profiles(selected, splits['validation'], device)
    report = {'checkpoint': str(output/'best.pt'), 'checkpoint_sha256': digest(output/'best.pt'),
              'selected_epoch': chosen['epoch'], 'epochs_completed': history[-1]['epoch'],
              'validation': validation, 'counts': counts, 'training_seconds': history[-1]['elapsed_seconds'],
              'backbone_unchanged': True, 'test_metrics_used': False, 'smoke_test': spec.get('smoke_test', False)}
    write_json(output/'training.json', report)
    return report


def evaluate(checkpoint, dataset, output, split='test', device='cpu'):
    backbone, head, saved = load_joint(resolve(checkpoint), 'cpu')
    spec, identity = saved['identity']['specification'], saved['identity']
    torch.set_num_threads(spec['cpu_threads'])
    if digest(resolve(dataset)) != identity['dataset_sha256'] or source_identity() != identity['sources']:
        raise ValueError('evaluation data or implementation differs from the joint checkpoint')
    if parameter_hash(backbone) != identity['backbone_parameter_sha256']:
        raise ValueError('joint checkpoint contains a different distance backbone')
    data = ProfileDataset(load_data(resolve(dataset)))
    rows = prepared(spec, data, backbone, split, resolve(checkpoint).parent, identity)
    result = {'split': split, 'checkpoint_sha256': digest(resolve(checkpoint)), 'smoke_test': spec.get('smoke_test', False),
              'counts': profile_counts(rows), **evaluate_profiles(head.to(device), rows, device)}
    write_json(resolve(output), result)
    return result
