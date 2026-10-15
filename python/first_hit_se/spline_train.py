"""Train the fixed-profile spline on existing samples, without recollection."""
import math
import time

import torch

from renewal.collect import digest, resolve, write_json
from renewal.model import mixture_log_prob, mixture_cdf
from renewal.reference_sampler import require_same_sampler
from . import FORMAT, VERSION, EVENT, SE_KERNEL
from .data import load_data, read_json
from .spline import SplineConfig, SplineFirstHitNetwork, resample_profile, local_profile
from .train import export_model


def prepare_profiles(profiles, points, device):
    prepared = []
    for p in profiles:
        b, db = local_profile(p['features'], p['distance'])
        row = {k: p[k] for k in ('distance', 'hit', 'initial_sign', 'speed')}
        row.update(fixed=resample_profile(p['features'], points), b=b, db=db,
                   horizon=torch.tensor(p['horizon'], dtype=torch.float64), b0=p['features'][0, 0])
        prepared.append({k: v.to(device) for k, v in row.items()})
    return prepared


def batch_rows(profiles, indices=None):
    labels = ('distance', 'hit', 'initial_sign', 'speed', 'b', 'db')
    return {k: torch.stack([p[k] if indices is None or k not in labels else p[k][indices[i]]
                            for i, p in enumerate(profiles)]) for k in profiles[0]}


def batch_losses(model, batch, with_pit=False):
    q, event, sign = (batch[k] for k in ('distance', 'hit', 'initial_sign'))
    n, samples = q.shape
    signs = q.new_tensor([1., -1.]).expand(n, -1)
    z, raw = model.encode(batch['fixed'][:, None].expand(-1, 2, -1), signs)
    horizon = batch['horizon'][:, None, None]
    H, log_rate, _ = model.branch_distribution(raw[:, :, None], q[:, None], horizon)
    log_prior = torch.special.log_ndtr(signs*batch['b0'][:, None])
    log_prior = log_prior-torch.logsumexp(log_prior, -1, keepdim=True)
    log_mass = log_prior[:, :, None]-H
    log_T = torch.logsumexp(log_mass, 1)
    log_density = torch.logsumexp(log_mass+log_rate, 1)
    branch = (sign < 0).long()[:, None]
    select = lambda values: values.gather(1, branch).squeeze(1)
    selected_H, selected_rate = select(H), select(log_rate)
    selected_prior = log_prior.gather(1, branch.squeeze(1))
    context = z.gather(1, branch.squeeze(1)[..., None].expand(-1, -1, z.shape[-1]))
    log_q = torch.zeros_like(q)
    pits = q.new_empty(0)
    if event.any():
        mix = model.mixture(context[event], q[event], batch['horizon'][:, None].expand_as(q)[event],
                            sign[event], batch['b'][event], batch['db'][event])
        log_q[event] = mixture_log_prob(*mix, batch['speed'][event])
        if with_pit:
            pits = mixture_cdf(*mix, batch['speed'][event])
    complete = -selected_prior+selected_H-event*selected_rate-log_q
    return {'complete_joint': complete, 'joint': torch.where(event, complete, -log_T),
            'distance': torch.where(event, -log_density, -log_T), 'speed': -log_q, 'pit': pits}


@torch.no_grad()
def evaluate(model, profiles, batch_size=4):
    model.eval()
    totals = {k: 0. for k in ('complete_joint', 'joint', 'distance', 'speed')}
    count = hits = negative = 0
    pits = []
    for begin in range(0, len(profiles), batch_size):
        batch = batch_rows(profiles[begin:begin+batch_size])
        result = batch_losses(model, batch, with_pit=True)
        for key in totals:
            totals[key] += float(result[key].sum())
        count += batch['hit'].numel()
        hits += int(batch['hit'].sum())
        negative += int((batch['initial_sign'] < 0).sum())
        pits.append(result['pit'].cpu())
    pit = torch.cat(pits).sort().values
    n = len(pit)
    ks = max(float((torch.arange(1, n+1)/n-pit).max()), float((pit-torch.arange(n)/n).max())) if n else None
    return {'samples': count, 'hits': hits, 'negative_starts': negative,
            'joint_nll': totals['joint']/count, 'distance_nll': totals['distance']/count,
            'complete_joint_nll': totals['complete_joint']/count,
            'speed_nll_on_hits': totals['speed']/hits if hits else None, 'speed_pit_ks': ks}


def train(config_path):
    config = read_json(config_path)
    output, data_path = resolve(config['output_directory']), resolve(config['dataset'])
    output.mkdir(parents=True, exist_ok=True)
    if (output/'best.pt').exists():
        raise ValueError('use a new output directory for a spline experiment')
    torch.set_num_threads(config.get('cpu_threads', 1))
    seed = config.get('seed', 20261015)
    torch.manual_seed(seed)
    device = config.get('device', 'cuda')
    data = load_data(data_path)
    if 'reference_sampler' in config:
        require_same_sampler(config['reference_sampler'], data['reference_sampler'], 'spline training')
    model = SplineFirstHitNetwork(SplineConfig(**config.get('model', {}))).to(device)
    model.reference_sampler = data['reference_sampler']
    dataset_hash = digest(data_path)
    splits = {s: prepare_profiles([p for p in data['profiles'] if p['split'] == s],
                                  model.config.profile_points, device) for s in ('train', 'validation', 'test')}
    if any(not p for p in splits.values()):
        raise ValueError('independent train, validation and test rays are required')
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.get('learning_rate', .001),
                                  weight_decay=config.get('weight_decay', .00001))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=6,
                                                          threshold=.0005, threshold_mode='abs', min_lr=.00002)
    write_json(output/'training_config.json', config)
    history, best, stale, selected_epoch = [], math.inf, 0, 0
    batch_size, limit = config.get('profiles_per_batch', 24), config.get('samples_per_profile_per_epoch', 512)
    began = time.perf_counter()
    for epoch in range(1, config.get('epochs', 500)+1):
        model.train()
        generator = torch.Generator().manual_seed(seed+epoch)
        order = torch.randperm(len(splits['train']), generator=generator).tolist()
        total = count = 0
        for begin in range(0, len(order), batch_size):
            profiles = [splits['train'][i] for i in order[begin:begin+batch_size]]
            indices = [torch.randperm(len(p['hit']), generator=generator)[:limit].to(device) for p in profiles]
            batch = batch_rows(profiles, indices)
            optimizer.zero_grad(set_to_none=True)
            losses = batch_losses(model, batch)
            loss = losses['complete_joint'].mean()
            if epoch <= config.get('distance_warmup_epochs', 20):
                loss = loss-losses['speed'].mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite spline loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.get('gradient_clip', 1.), error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach())*batch['hit'].numel()
            count += batch['hit'].numel()
        if epoch % config.get('validation_every', 5) and epoch != config.get('epochs', 500):
            continue
        validation = evaluate(model, splits['validation'])
        score = validation['joint_nll']
        if not math.isfinite(score):
            raise FloatingPointError('nonfinite spline validation')
        improved = score < best
        if improved:
            best, stale, selected_epoch = score, 0, epoch
        else:
            stale += 1
        scheduler.step(score)
        entry = {'epoch': epoch, 'training_nll': total/count, 'validation': validation,
                 'best': improved, 'elapsed_seconds': time.perf_counter()-began,
                 'learning_rate': optimizer.param_groups[0]['lr']}
        history.append(entry)
        bundle = {'format': FORMAT, 'version': VERSION, 'event': EVENT, 'kernel': SE_KERNEL,
                  'reference_sampler': data['reference_sampler'], 'model_config': model.configuration(),
                  'model_state': model.state_dict(), 'epoch': epoch, 'dataset_sha256': dataset_hash,
                  'training_config': config, 'validation': validation}
        if improved:
            torch.save(bundle, output/'best.pt')
        torch.save({**bundle, 'optimizer_state': optimizer.state_dict(), 'history': history}, output/'last.pt')
        write_json(output/'history.json', history)
        print(f"spline {model.config.profile_points}/{model.config.bins} epoch {epoch}: train={total/count:.5f} "
              f"val={score:.5f} distance={validation['distance_nll']:.5f} W={validation['speed_nll_on_hits']:.5f} "
              f"elapsed={entry['elapsed_seconds']:.1f}s", flush=True)
        if epoch >= config.get('minimum_epochs', 150) and stale >= config.get('patience_checks', 20):
            break
    model.load_state_dict(torch.load(output/'best.pt', map_location=device, weights_only=True)['model_state'])
    report = {'architecture': 'spline', 'dataset_sha256': dataset_hash, 'reference_sampler': data['reference_sampler'],
              'selected_epoch': selected_epoch, 'completed_epochs': epoch,
              'parameters': sum(p.numel() for p in model.parameters()), 'training_seconds': time.perf_counter()-began,
              'model_config': model.configuration(), 'metrics': {s: evaluate(model, p) for s, p in splits.items()}}
    write_json(output/'run_report.json', report)
    export_model(output/'best.pt', output/'first_hit_model.json')
    return report
