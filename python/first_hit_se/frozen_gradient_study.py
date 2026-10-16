"""Paired mixture/CDF gradient heads on each frozen multiview residual model."""
import math
import time

import torch

from renewal.collect import digest, write_json
from .data import read_json
from .distance_models import latent, predictions
from .distance_probe import ks
from .distance_study import load_run, synchronize
from .gradient_cdf import cdf, log_prob, make_head
from .hazard_study import make_batch, parameter_hash, prepare
from .multiview_study import test_groups
from .spline import local_profile


@torch.no_grad()
def cache_profiles(model, data, config):
    splits = prepare(data, config, config['device'])
    for split, prepared in splits.items():
        raw = [p for p in data['profiles'] if p['split'] == split]
        for p, original in zip(prepared, raw):
            b, db = local_profile(original['features'], original['distance'])
            p.update(speed=original['speed'].to(config['device']), b=b.to(config['device']), db=db.to(config['device']))
            signs = p['fixed'].new_tensor([1., -1.])
            p['latent'] = latent(model, p['fixed'][None].expand(2, -1), signs).detach()
            prediction = predictions(model, make_batch([p]))
            p['joint_distance_sum'] = torch.where(p['hit'], prediction['complete_distance_nll'][0],
                                                  prediction['distance_nll'][0]).sum()
    return splits


def points(profiles, indices=None):
    fields = {k: [] for k in ('z', 'q', 'horizon', 'sign', 'b', 'db', 'w', 'ray')}
    samples = 0
    for i, p in enumerate(profiles):
        index = (torch.arange(len(p['hit']), device=p['hit'].device) if indices is None else indices[i])
        samples += len(index)
        selected = index[p['hit'][index]]
        sign = p['initial_sign'][selected]
        fields['z'].append(p['latent'][(sign < 0).long()])
        for key, original in (('q', 'distance'), ('sign', 'initial_sign'), ('b', 'b'), ('db', 'db'), ('w', 'speed')):
            fields[key].append(p[original][selected])
        fields['horizon'].append(p['horizon'].expand(len(selected)))
        fields['ray'].append(torch.full_like(selected, i))
    return {**{k: torch.cat(v) for k, v in fields.items()}, 'samples': samples}


def head_parameters(head, p):
    return head(*(p[k] for k in ('z', 'q', 'horizon', 'sign', 'b', 'db')))


@torch.no_grad()
def evaluate_head(head, kind, profiles):
    head.eval()
    total, joint, samples, hits = 0., 0., 0, 0
    pits, locations, rows = [], [], []
    for start in range(0, len(profiles), 4):
        batch = profiles[start:start+4]
        p = points(batch)
        params = head_parameters(head, p)
        losses = -log_prob(kind, params, p['w'])
        values = cdf(kind, params, p['w'])
        if not torch.isfinite(losses).all() or not ((values >= 0) & (values <= 1)).all():
            raise FloatingPointError('invalid gradient density or CDF')
        total += float(losses.sum())
        joint += float(sum(q['joint_distance_sum'] for q in batch)+losses.sum())
        samples += p['samples']
        hits += len(p['w'])
        pits.append(values)
        locations.append(p['q']/p['horizon'])
        for j in range(len(batch)):
            mask = p['ray'] == j
            n = int(mask.sum())
            rows.append({'hits': n, 'speed_nll': float(losses[mask].mean()) if n else None,
                         'pit_ks': ks(values[mask]),
                         'pit_counts': torch.histc(values[mask], bins=20, min=0, max=1).long().cpu().tolist(),
                         'joint_nll': float((batch[j]['joint_distance_sum']+losses[mask].sum())/len(batch[j]['hit']))})
    values, where = torch.cat(pits), torch.cat(locations)
    bins = []
    for lo, hi in ((0., .25), (.25, .5), (.5, .75), (.75, 1.00000001)):
        mask = (where >= lo) & (where < hi)
        bins.append({'lower': lo, 'upper': min(hi, 1.), 'hits': int(mask.sum()), 'pit_ks': ks(values[mask])})
    return {'speed_nll': total/hits, 'joint_nll': joint/samples, 'pit_ks': ks(values),
            'hits': hits, 'samples': samples, 'distance_bins': bins, 'per_ray': rows}


def train_head(spec, config, model, splits, source, kind, seed, output):
    destination = output/'gradient'/f'{kind}_{seed}'
    destination.mkdir(parents=True, exist_ok=True)
    identity = {'source_sha256': digest(source), 'dataset_sha256': digest(output/'dataset.pt'),
                'spec': spec, 'kind': kind, 'seed': seed}
    report_path = destination/'run_report.json'
    if report_path.exists():
        result = read_json(report_path)
        if result['identity'] != identity:
            raise ValueError('completed gradient run has a different identity')
        return result
    head = make_head(kind, config['model']['hidden'], spec['gradient_width'],
                     spec['gradient_bins'] if kind == 'cdf' else spec['gradient_components'], seed+100000).to(config['device'])
    before = parameter_hash(model)
    common_initial = parameter_hash(head.net[:-1])
    optimizer = torch.optim.AdamW(head.parameters(), lr=.001, weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, spec['gradient_epochs'], eta_min=.00002)
    history, best, first, elapsed = [], math.inf, 1, 0.
    if (destination/'latest.pt').exists():
        saved = torch.load(destination/'latest.pt', weights_only=True, map_location=config['device'])
        if saved['identity'] != identity:
            raise ValueError('resumed gradient run has a different identity')
        head.load_state_dict(saved['head_state'])
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        history, best, first, elapsed = saved['history'], saved['best'], saved['epoch']+1, saved['elapsed']
    synchronize(config['device'])
    began = time.perf_counter()
    for epoch in range(first, spec['gradient_epochs']+1):
        head.train()
        generator = torch.Generator().manual_seed(seed+200000+epoch)
        order = torch.randperm(len(splits['train']), generator=generator).tolist()
        total, samples = 0., 0
        for begin in range(0, len(order), config['profiles_per_batch']):
            ps = [splits['train'][i] for i in order[begin:begin+config['profiles_per_batch']]]
            indices = [torch.randperm(len(p['hit']), generator=generator)[:spec['gradient_samples_per_profile']].to(config['device']) for p in ps]
            p = points(ps, indices)
            if not len(p['w']):
                continue
            optimizer.zero_grad(set_to_none=True)
            losses = -log_prob(kind, head_parameters(head, p), p['w'])
            loss = losses.sum()/p['samples']
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite gradient training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), config['gradient_clip'], error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach())*p['samples']
            samples += p['samples']
        scheduler.step()
        if epoch % 5 and epoch != spec['gradient_epochs']:
            continue
        validation = evaluate_head(head, kind, splits['validation'])
        score = validation['speed_nll']
        bundle = {'identity': identity, 'head_state': head.state_dict(), 'epoch': epoch,
                  'validation': validation, 'config': config}
        if score < best:
            best = score
            torch.save(bundle, destination/'best.pt')
        synchronize(config['device'])
        seconds = elapsed+time.perf_counter()-began
        history.append({'epoch': epoch, 'train_joint_gradient_term': total/samples,
                        'validation': validation, 'elapsed_seconds': seconds})
        torch.save({**bundle, 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                    'history': history, 'best': best, 'elapsed': seconds}, destination/'latest.pt')
        write_json(destination/'history.json', history)
        if epoch % 25 == 0:
            print(f'gradient {kind}/{seed} epoch={epoch}: val NLL={score:.6f}, PIT={validation["pit_ks"]:.5f}, {seconds:.1f}s', flush=True)
    chosen = torch.load(destination/'best.pt', weights_only=True, map_location=config['device'])
    head.load_state_dict(chosen['head_state'])
    if parameter_hash(model) != before or any(p.requires_grad for p in model.parameters()):
        raise AssertionError('frozen distance model changed')
    result = {'identity': identity, 'kind': kind, 'seed': seed, 'selected_epoch': chosen['epoch'],
              'parameters': sum(p.numel() for p in head.parameters()),
              'common_initial_sha256': common_initial, 'distance_model_unchanged': True,
              'training_seconds': history[-1]['elapsed_seconds'],
              'metrics': {split: evaluate_head(head, kind, ps) for split, ps in splits.items()}}
    write_json(report_path, result)
    return result


def load_head(spec, config, output, kind, seed, device):
    chosen = torch.load(output/'gradient'/f'{kind}_{seed}'/'best.pt', map_location=device, weights_only=True)
    head = make_head(kind, config['model']['hidden'], spec['gradient_width'],
                     spec['gradient_bins'] if kind == 'cdf' else spec['gradient_components'], seed+100000).to(device)
    head.load_state_dict(chosen['head_state'])
    return head.eval()


def run(spec, config, data, output):
    results = []
    for seed in spec['seeds']:
        source = output/'runs'/f'residual_{seed}'/'best.pt'
        model, _ = load_run(source, config['device'])
        model.requires_grad_(False)
        splits = cache_profiles(model, data, config)
        paired = [train_head(spec, config, model, splits, source, kind, seed, output) for kind in ('mixture', 'cdf')]
        if paired[0]['common_initial_sha256'] != paired[1]['common_initial_sha256']:
            raise AssertionError('gradient head hidden layers did not start identically')
        results += paired
        del model, splits
    write_json(output/'gradient_evaluation.json', {'runs': results,
        'scope': 'conditional longitudinal speed at reference first-hit positions; backbone frozen',
        'selection': 'minimum validation conditional speed NLL', 'groups': test_groups(data)})
    return {'gradient_runs': len(results), 'frozen_backbones_unchanged': True}
