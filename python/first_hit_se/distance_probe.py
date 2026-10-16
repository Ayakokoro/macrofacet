"""Matched conditional-gradient heads on distance-trained encoders.

The frozen probe diagnoses the retained representation. The finetuned probe
optimizes the complete censored distance/gradient likelihood. Neither replaces
joint sampling and rendering validation.
"""
import math
import time

import torch
from torch import nn
from torch.nn import functional as F

from renewal.collect import digest, resolve, write_json
from renewal.model import _mlp, mixture_cdf, mixture_log_prob
from .data import read_json
from .distance_models import MODES, latent, predictions
from .hazard_study import make_batch, parameter_hash, prepare
from .spline import local_profile


class GradientHead(nn.Module):
    def __init__(self, hidden, width, components):
        super().__init__()
        self.net = _mlp(hidden+4, width, 3*components)

    def forward(self, z, q, horizon, sign, b, db):
        local = torch.stack((q/horizon, q.log1p(), (sign*b).asinh(), (sign*db).asinh()), -1)
        raw = self.net(torch.cat((z, local.to(z.dtype)), -1))
        weight, delta, scale = raw.chunk(3, -1)
        return F.log_softmax(weight, -1), -sign[..., None]*db[..., None]+delta, F.softplus(scale)+.01


def make_probe_batch(profiles, indices=None):
    batch = make_batch(profiles, indices)
    for key in ('speed', 'b', 'db'):
        batch[key] = torch.stack([p[key] if indices is None else p[key][indices[i]]
                                  for i, p in enumerate(profiles)])
    return batch


def probe_predictions(model, head, batch, frozen=False):
    n, count = batch['distance'].shape
    signs = batch['fixed'].new_tensor([1., -1.]).expand(n, -1)
    z = latent(model, batch['fixed'][:, None].expand(-1, 2, -1), signs)
    if frozen:
        z = z.detach()
    index = (batch['initial_sign'] < 0).long()
    z = z.gather(1, index[..., None].expand(n, count, z.shape[-1]))
    hit = batch['hit']
    speed_loss = torch.zeros_like(batch['distance'])
    pit = torch.empty_like(speed_loss)
    if hit.any():
        params = head(z[hit], batch['distance'][hit],
                      batch['horizon'][:, None].expand(-1, count)[hit],
                      batch['initial_sign'][hit], batch['b'][hit], batch['db'][hit])
        speed_loss[hit] = -mixture_log_prob(*params, batch['speed'][hit])
        with torch.no_grad():
            pit[hit] = mixture_cdf(*params, batch['speed'][hit])
    return speed_loss, pit


def ks(values):
    values = values.double().sort().values
    n = len(values)
    if not n:
        return None
    rank = torch.arange(1, n+1, device=values.device, dtype=torch.float64)/n
    return float(torch.maximum(rank-values, values-(rank-1/n)).max())


@torch.no_grad()
def evaluate_probe(model, head, profiles):
    model.eval()
    head.eval()
    totals = {'distance_nll': 0., 'joint_nll': 0., 'speed_nll': 0.}
    count, hits, pits, groups, rows = 0, 0, [], [], []
    for begin in range(0, len(profiles), 4):
        batch = make_probe_batch(profiles[begin:begin+4])
        pred = predictions(model, batch)
        speed_loss, pit = probe_predictions(model, head, batch)
        joint = torch.where(batch['hit'], pred['complete_distance_nll']+speed_loss,
                            pred['distance_nll'])
        totals['distance_nll'] += float(pred['distance_nll'].sum())
        totals['joint_nll'] += float(joint.sum())
        totals['speed_nll'] += float(speed_loss.sum())
        count += batch['hit'].numel()
        hits += int(batch['hit'].sum())
        pits.append(pit[batch['hit']])
        groups.append((batch['distance']/batch['horizon'][:, None])[batch['hit']])
        for j in range(len(batch['hit'])):
            selected = batch['hit'][j]
            if selected.any():
                rows.append({'hits': int(selected.sum()), 'pit_ks': ks(pit[j, selected]),
                             'speed_nll': float(speed_loss[j, selected].mean())})
    values, location = torch.cat(pits), torch.cat(groups)
    bins = []
    for lo, hi in ((0., .25), (.25, .5), (.5, .75), (.75, 1.00000001)):
        selected = (location >= lo) & (location < hi)
        bins.append({'lower_q_over_Q': lo, 'upper_q_over_Q': min(hi, 1.),
                     'hits': int(selected.sum()), 'pit_ks': ks(values[selected])})
    return {'distance_nll': totals['distance_nll']/count,
            'joint_nll': totals['joint_nll']/count,
            'speed_nll': totals['speed_nll']/hits, 'hits': hits, 'samples': count,
            'pit_ks': ks(values), 'distance_bins': bins, 'per_ray': rows}


def run_probe(config, data, output):
    from .distance_study import evaluate, load_run, synchronize
    spec, seed = config['gradient_probe'], config['seeds'][0]
    splits = prepare(data, config, config['device'])
    for split, ps in splits.items():
        original = [p for p in data['profiles'] if p['split'] == split]
        for p, raw in zip(ps, original):
            b, db = local_profile(raw['features'], raw['distance'])
            p.update(speed=raw['speed'].to(config['device']),
                     b=b.to(config['device']), db=db.to(config['device']))
    results = []
    for mode in MODES:
        source = output/'runs'/f'{mode}_{seed}'/'best.pt'
        source_hash = digest(source)
        for regime in spec['regimes']:
            destination = output/'gradient'/f'{mode}_{regime}'
            destination.mkdir(parents=True, exist_ok=True)
            report_path = destination/'run_report.json'
            if report_path.exists():
                completed = read_json(report_path)
                if completed['source_sha256'] != source_hash or completed['config'] != config:
                    raise ValueError('gradient probe identity changed')
                results.append(completed)
                continue
            model, _ = load_run(source, config['device'])
            before_hash = parameter_hash(model)
            frozen = regime == 'frozen'
            if regime not in ('frozen', 'finetune'):
                raise ValueError('unknown gradient probe regime')
            model.requires_grad_(not frozen)
            torch.manual_seed(seed+100000)
            head = GradientHead(config['model']['hidden'], spec['width'], spec['components']).to(config['device'])
            head_hash = parameter_hash(head)
            params = list(head.parameters()) + ([] if frozen else list(model.parameters()))
            optimizer = torch.optim.AdamW(params, lr=spec['learning_rate'], weight_decay=config['weight_decay'])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, spec['epochs'], eta_min=spec['minimum_learning_rate'])
            best, history = math.inf, []
            synchronize(config['device'])
            start = time.perf_counter()
            for epoch in range(1, spec['epochs']+1):
                model.train(not frozen)
                head.train()
                generator = torch.Generator().manual_seed(seed+200000+epoch)
                order = torch.randperm(len(splits['train']), generator=generator).tolist()
                for begin in range(0, len(order), config['profiles_per_batch']):
                    ps = [splits['train'][i] for i in order[begin:begin+config['profiles_per_batch']]]
                    indices = [torch.randperm(len(p['hit']), generator=generator)[
                        :spec['samples_per_profile']].to(config['device']) for p in ps]
                    batch = make_probe_batch(ps, indices)
                    optimizer.zero_grad(set_to_none=True)
                    speed_loss, _ = probe_predictions(model, head, batch, frozen=frozen)
                    loss = speed_loss.mean()
                    if not frozen:
                        loss = loss+predictions(model, batch)['training_nll'].mean()
                    if not torch.isfinite(loss):
                        raise FloatingPointError('nonfinite conditional gradient loss')
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(params, config['gradient_clip'], error_if_nonfinite=True)
                    optimizer.step()
                scheduler.step()
                if epoch % spec['validation_every'] and epoch != spec['epochs']:
                    continue
                metrics = evaluate_probe(model, head, splits['validation'])
                history.append({'epoch': epoch, 'validation': metrics})
                if metrics['joint_nll'] < best:
                    best = metrics['joint_nll']
                    torch.save({'model_state': model.state_dict(), 'head_state': head.state_dict(),
                                'epoch': epoch, 'config': config, 'mode': mode, 'regime': regime,
                                'source_sha256': source_hash, 'validation': metrics}, destination/'best.pt')
                if epoch % 50 == 0 or epoch == spec['epochs']:
                    print(f'gradient {mode}/{regime} epoch={epoch}: val joint={metrics["joint_nll"]:.6f}, '
                          f'speed={metrics["speed_nll"]:.6f}, PIT={metrics["pit_ks"]:.4f}', flush=True)
            synchronize(config['device'])
            seconds = time.perf_counter()-start
            chosen = torch.load(destination/'best.pt', map_location=config['device'], weights_only=True)
            model.load_state_dict(chosen['model_state'])
            head.load_state_dict(chosen['head_state'])
            if frozen and parameter_hash(model) != before_hash:
                raise AssertionError('frozen distance model changed')
            result = {'config': config, 'mode': mode, 'regime': regime, 'seed': seed,
                      'source_sha256': source_hash, 'initial_head_sha256': head_hash,
                      'selected_epoch': chosen['epoch'], 'training_seconds': seconds,
                      'distance_model_unchanged': parameter_hash(model) == before_hash,
                      'head_parameters': sum(p.numel() for p in head.parameters()),
                      'metrics': {s: evaluate_probe(model, head, ps) for s, ps in splits.items()},
                      'distance_metrics': {s: evaluate(model, ps) for s, ps in splits.items()}}
            write_json(destination/'history.json', history)
            write_json(report_path, result)
            results.append(result)
    if len({r['initial_head_sha256'] for r in results}) != 1:
        raise AssertionError('gradient head initialization differs')
    summary = {'scope': 'single paired seed; reference first-hit locations and observed entry signs; '
                        'conditional longitudinal speed, not a rendering or joint-sampling validation',
               'selection': 'minimum validation joint NLL; frozen mode is equivalent to conditional speed selection',
               'finetune': 'distance model and gradient head jointly trained on censored joint likelihood',
               'runs': results}
    write_json(output/'gradient_probe.json', summary)
    return summary
