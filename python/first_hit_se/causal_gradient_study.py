"""Frozen GRU conditional-speed experiment; validation-only head selection."""
from pathlib import Path
import time

import numpy as np
import torch

from renewal.collect import digest, resolve, write_json
from .causal_data import to_device
from .causal_gradient import CausalGradient, GradientConfig, cache_profile, points
from .causal_study import load_run as load_backbone, synchronize
from .data import load_data, read_json
from .distance_probe import ks
from .gradient_cdf import cdf, log_prob
from .hazard_study import parameter_hash
from .multiview_study import test_groups


def backbone_path(spec, seed):
    return resolve(spec['backbone_directory'])/'runs'/f'gru_nll_{seed}'/'best.pt'


def register(spec, output):
    parent = read_json(resolve(spec['backbone_directory'])/'protocol.json')
    sources = dict(parent['training_source_sha256'])
    for name in ('causal_gradient.py', 'causal_gradient_study.py', 'gradient_cdf.py',
                 'distance_probe.py', 'spline.py', 'hazard.py'):
        p = Path(__file__).with_name(name)
        sources[str(p)] = digest(p)
    p = Path(__file__).parents[1]/'renewal'/'model.py'
    sources[str(p)] = digest(p)
    for p, value in parent['training_source_sha256'].items():
        if digest(Path(p)) != value:
            raise ValueError('frozen distance training implementation changed')
    paths = [backbone_path(spec, seed) for seed in spec['seeds']]
    old = resolve(spec['baseline_directory'])
    paths += [old/'gradient'/f'{kind}_{seed}'/'best.pt' for kind in ('mixture', 'cdf') for seed in spec['seeds']]
    identity = {'specification': spec, 'dataset_sha256': digest(resolve(spec['dataset'])),
        'source_sha256': sources, 'checkpoint_sha256': {str(p): digest(p) for p in paths},
        'backbone_training_sha256': digest(resolve(spec['backbone_directory'])/'training.json'),
        'baseline_evaluation_sha256': digest(old/'gradient_evaluation.json'),
        'protocol': 'Freeze all distance weights; select conditional heads on validation NLL; evaluate fixed test after all runs complete.'}
    if identity['dataset_sha256'] != spec['dataset_sha256']:
        raise ValueError('dataset changed')
    if spec['selection'] != 'minimum_validation_conditional_speed_nll':
        raise ValueError('unsupported checkpoint selection')
    path = output/'protocol.json'
    if path.exists() and read_json(path) != identity:
        raise ValueError('experiment inputs or implementation changed; use a new output directory')
    write_json(path, identity)


def prepared(spec, data, seed, split, output):
    path = output/f'contexts_{seed}_{split}.pt'
    identity = {'protocol_sha256': digest(output/'protocol.json'), 'seed': seed, 'split': split}
    if path.exists():
        cache = torch.load(path, weights_only=True)
        if cache['identity'] != identity:
            raise ValueError('cached contexts changed identity')
    else:
        model, saved = load_backbone(backbone_path(spec, seed), 'cpu')
        model.requires_grad_(False)
        if model.config.hidden != spec['head']['hidden'] or not model.config.history:
            raise ValueError('incompatible frozen GRU')
        before = parameter_hash(model)
        raw = [p for p in data['profiles'] if p['split'] == split]
        rows = [cache_profile(model, p) for p in raw]
        if parameter_hash(model) != before or any(p.requires_grad for p in model.parameters()):
            raise AssertionError('backbone changed while caching')
        cache = {'identity': identity, 'backbone_parameter_sha256': before, 'rows': rows}
        torch.save(cache, path)
        print(f'cached seed={seed} {split}: {len(rows)} rays', flush=True)
    return [to_device(p, spec['device']) for p in cache['rows']]


def calibration(values, losses):
    count = len(values)
    if not count:
        return {'hits': 0, 'speed_nll': None, 'pit_ks': None, 'pit_counts': [0]*20,
                'coverage_68': None, 'coverage_90': None, 'lower_tail_01': None, 'upper_tail_99': None}
    return {'hits': count, 'speed_nll': float(losses.mean()), 'pit_ks': ks(values),
        'pit_counts': torch.histc(values, bins=20, min=0, max=1).long().cpu().tolist(),
        'coverage_68': float(((values >= .16) & (values <= .84)).double().mean()),
        'coverage_90': float(((values >= .05) & (values <= .95)).double().mean()),
        'lower_tail_01': float((values < .01).double().mean()),
        'upper_tail_99': float((values > .99).double().mean())}


@torch.no_grad()
def evaluate(head, profiles, detailed=True):
    head.eval()
    total, hits, samples, joint = 0., 0, 0, 0.
    rows, all_values, all_loss, all_fraction, all_sign, all_db = [], [], [], [], [], []
    for start in range(0, len(profiles), 4):
        batch = profiles[start:start+4]
        p = points(batch)
        samples += p['samples']
        if len(p['w']):
            params = head(p)
            losses = -log_prob(head.config.kind, params, p['w'])
        else:
            losses = p['w']
        if not torch.isfinite(losses).all():
            raise FloatingPointError('nonfinite conditional speed likelihood')
        total += float(losses.sum())
        hits += len(losses)
        joint += float(sum(row['joint_distance_sum'] for row in batch)+losses.sum())
        if not detailed:
            continue
        values = cdf(head.config.kind, params, p['w']) if len(losses) else losses
        if not torch.isfinite(values).all() or not ((values >= 0) & (values <= 1)).all():
            raise FloatingPointError('invalid conditional speed CDF')
        for j, row in enumerate(batch):
            mask = p['ray'] == j
            rows.append({**calibration(values[mask], losses[mask]),
                         'joint_nll': float((row['joint_distance_sum']+losses[mask].sum())/len(row['hit']))})
        all_values.append(values.cpu())
        all_loss.append(losses.cpu())
        all_fraction.append(p['fraction'].cpu())
        all_sign.append(p['sign'].cpu())
        all_db.append((p['sign']*p['db']).cpu())
    if not hits:
        raise ValueError('evaluation split contains no hit labels')
    result = {'speed_nll': total/hits, 'joint_nll': joint/samples, 'hits': hits, 'samples': samples}
    if detailed:
        values, losses, fraction, signs, db = map(torch.cat, (all_values, all_loss, all_fraction, all_sign, all_db))
        result.update(calibration(values, losses))
        result['per_ray'] = rows
        result['distance_bins'] = [{**calibration(values[mask], losses[mask]), 'lower': lo, 'upper': hi}
            for lo, hi in ((0., .25), (.25, .5), (.5, .75), (.75, 1.))
            for mask in [(fraction >= lo) & ((fraction < hi) if hi < 1 else (fraction <= hi))]]
        result['entry_signs'] = {str(sign): calibration(values[signs == sign], losses[signs == sign]) for sign in (1, -1)}
        result['signed_slope_bins'] = {name: calibration(values[mask], losses[mask]) for name, mask in (
            ('below_-2', db < -2), ('-2_to_0', (db >= -2) & (db < 0)),
            ('0_to_2', (db >= 0) & (db < 2)), ('above_2', db >= 2))}
    return result


def train_one(spec, variant, seed, splits, output):
    directory = output/'runs'/f"{variant['name']}_{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    identity = {'protocol_sha256': digest(output/'protocol.json'), 'variant': variant, 'seed': seed}
    if (directory/'run_report.json').exists():
        result = read_json(directory/'run_report.json')
        if result['identity'] != identity:
            raise ValueError('completed gradient run changed identity')
        return result
    config = GradientConfig(**spec['head'], kind=variant['kind'], history=variant['history'])
    head = CausalGradient(config, seed+100000).to(spec['device'])
    initial, hidden = parameter_hash(head), parameter_hash(head.net[:-1])
    optimizer = torch.optim.AdamW(head.parameters(), lr=spec['learning_rate'], weight_decay=spec['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, spec['epochs'], eta_min=spec['minimum_learning_rate'])
    history, best, first, elapsed = [], float('inf'), 1, 0.
    if (directory/'latest.pt').exists():
        saved = torch.load(directory/'latest.pt', weights_only=True, map_location=spec['device'])
        if saved['identity'] != identity:
            raise ValueError('resumed run changed identity')
        head.load_state_dict(saved['head_state'])
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        history, best, first, elapsed = saved['history'], saved['best'], saved['epoch']+1, saved['elapsed']
    synchronize(spec['device'])
    began = time.perf_counter()
    for epoch in range(first, spec['epochs']+1):
        head.train()
        generator = torch.Generator().manual_seed(seed+200000+epoch)
        order = torch.randperm(len(splits['train']), generator=generator).tolist()
        total, samples = 0., 0
        for begin in range(0, len(order), spec['profiles_per_batch']):
            ps = [splits['train'][i] for i in order[begin:begin+spec['profiles_per_batch']]]
            indices = [torch.randperm(len(p['hit']), generator=generator)[:spec['samples_per_profile']].to(spec['device']) for p in ps]
            p = points(ps, indices)
            samples += p['samples']
            if not len(p['w']):
                continue
            optimizer.zero_grad(set_to_none=True)
            loss = -log_prob(config.kind, head(p), p['w']).sum()/p['samples']
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite training objective')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), spec['gradient_clip'], error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach())*p['samples']
        scheduler.step()
        if epoch % spec['validation_every'] and epoch != spec['epochs']:
            continue
        validation = evaluate(head, splits['validation'], False)
        synchronize(spec['device'])
        seconds = elapsed+time.perf_counter()-began
        bundle = {'format': 'first_hit_causal_gradient', 'identity': identity, 'head_config': head.configuration(),
            'head_state': head.state_dict(), 'epoch': epoch, 'validation': validation}
        if validation['speed_nll'] < best:
            best = validation['speed_nll']
            torch.save(bundle, directory/'best.pt')
        history.append({'epoch': epoch, 'elapsed_seconds': seconds,
                        'train_gradient_term_per_realization': total/samples, 'validation': validation})
        torch.save({**bundle, 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                    'history': history, 'best': best, 'elapsed': seconds}, directory/'latest.pt')
        write_json(directory/'history.json', history)
        if epoch % 25 == 0 or epoch == spec['epochs']:
            print(f"{variant['name']}/{seed} epoch={epoch}: val speed NLL={validation['speed_nll']:.6f}, best={best:.6f}, {seconds:.1f}s", flush=True)
    head, saved = load_head(directory/'best.pt', spec['device'])
    result = {'identity': identity, 'parameters': sum(p.numel() for p in head.parameters()),
        'initial_parameter_sha256': initial, 'initial_hidden_sha256': hidden, 'selected_epoch': saved['epoch'],
        'validation': saved['validation'], 'training_seconds': history[-1]['elapsed_seconds'],
        'checkpoint': str(directory/'best.pt'), 'checkpoint_sha256': digest(directory/'best.pt')}
    write_json(directory/'run_report.json', result)
    return result


def load_head(path, device='cpu'):
    saved = torch.load(path, map_location=device, weights_only=True)
    if saved['format'] != 'first_hit_causal_gradient':
        raise ValueError('incompatible conditional gradient checkpoint')
    head = CausalGradient(GradientConfig(**saved['head_config']), saved['identity']['seed']+100000).to(device)
    head.load_state_dict(saved['head_state'])
    return head.eval(), saved


def train(spec, data, output):
    runs = []
    for seed in spec['seeds']:
        splits = {s: prepared(spec, data, seed, s, output) for s in ('train', 'validation')}
        paired = [train_one(spec, v, seed, splits, output) for v in spec['variants']]
        if len({r['initial_hidden_sha256'] for r in paired}) != 1:
            raise AssertionError('hidden layers did not start identically')
        if len({r['initial_parameter_sha256'] for r in paired if r['identity']['variant']['kind'] == 'mixture'}) != 1:
            raise AssertionError('history ablation did not start identically')
        runs += paired
    result = {'protocol_sha256': digest(output/'protocol.json'), 'test_metrics_used': False, 'runs': runs}
    write_json(output/'training.json', result)
    return {'training': str(output/'training.json'), 'completed_runs': len(runs)}


def evaluate_selected(spec, data, output):
    trained = read_json(output/'training.json')
    if trained['protocol_sha256'] != digest(output/'protocol.json') or trained['test_metrics_used']:
        raise ValueError('training identity or selection changed')
    results = {v['name']: [] for v in spec['variants']}
    hashes = {}
    for seed in spec['seeds']:
        profiles = prepared(spec, data, seed, 'test', output)
        for variant in spec['variants']:
            record = next(r for r in trained['runs'] if r['identity']['seed'] == seed and r['identity']['variant'] == variant)
            path = Path(record['checkpoint'])
            if digest(path) != record['checkpoint_sha256']:
                raise ValueError('selected head checkpoint changed')
            hashes[str(path)] = digest(path)
            head, _ = load_head(path, spec['device'])
            score = evaluate(head, profiles)
            results[variant['name']].append({'seed': seed, 'metrics': score})
            print(f"test {variant['name']}/{seed}: NLL={score['speed_nll']:.6f}, global PIT={score['pit_ks']:.6f}", flush=True)
    old = read_json(resolve(spec['baseline_directory'])/'gradient_evaluation.json')
    for kind in ('mixture', 'cdf'):
        results['mlp_'+kind] = [{'seed': r['seed'], 'metrics': r['metrics']['test']} for r in old['runs'] if r['kind'] == kind]
    result = {'training_sha256': digest(output/'training.json'), 'checkpoint_sha256': hashes,
              'results': results, 'groups': test_groups(data)}
    write_json(output/'evaluation.json', result)
    return {name: {'speed_nll': float(np.mean([r['metrics']['speed_nll'] for r in runs])),
                   'global_pit_ks': float(np.mean([r['metrics']['pit_ks'] for r in runs]))} for name, runs in results.items()}


def run(config_path, phase='all'):
    spec = read_json(resolve(config_path))
    output = resolve(spec['output_directory'])
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(spec['cpu_threads'])
    torch.backends.cuda.matmul.allow_tf32 = False
    register(spec, output)
    data = load_data(resolve(spec['dataset']))
    if phase in ('all', 'train'):
        result = train(spec, data, output)
    if phase in ('all', 'evaluate'):
        result = evaluate_selected(spec, data, output)
    if phase in ('all', 'diagnose'):
        from .causal_gradient_diagnostics import diagnose
        result = diagnose(spec, data, output)
    if phase in ('all', 'report'):
        from .causal_gradient_report import report
        result = report(spec, data, output)
    return result
