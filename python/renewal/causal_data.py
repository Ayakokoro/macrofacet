"""Shared-profile A/B labels for conditional residual-hazard training."""
import torch
from torch.nn.utils.rnn import pad_sequence

from first_hit_se.causal_data import geometry as base_geometry, queries, to_device
from . import SE_KERNEL
from .data import FEATURES, SPLITS


class ProfileDataset:
    """Index flat Renewal labels once; never repeat GRU work per trajectory."""
    def __init__(self, data):
        if data['kernel'] != SE_KERNEL or data['feature_names'] != list(FEATURES):
            raise ValueError('conditional residual GRU requires cubic unit-length SE data')
        self.data = data
        count = len(data['initial'])
        if (data['initial'].shape != (count, 4) or len(data['offsets']) != count+1
                or len(data['split']) != count or len(data['horizon']) != count
                or data['offsets'][0] != 0 or data['offsets'][-1] != len(data['features'])
                or (data['offsets'].diff() <= 0).any()
                or not torch.isin(data['split'], torch.tensor(list(SPLITS.values()))).all()):
            raise ValueError('invalid packed profiles')
        ids = data['sample_profile']
        if (ids.ndim != 1 or ids.dtype != torch.long or not len(ids)
                or ids.min() < 0 or ids.max() >= count
                or data['hit'].shape != ids.shape or data['distance'].shape != ids.shape):
            raise ValueError('invalid packed sample/profile mapping')
        counts = torch.bincount(ids, minlength=count)
        if (counts == 0).any():
            raise ValueError('profile without trajectory labels')
        self.offsets = torch.nn.functional.pad(counts.cumsum(0), (1, 0)).tolist()
        self.order = torch.argsort(ids, stable=True) if (ids[1:] < ids[:-1]).any() else None

    def profiles(self, split, limit_per_mode=None):
        if limit_per_mode is not None and limit_per_mode < 1:
            raise ValueError('profile limit must be positive')
        used = [0, 0]
        d = self.data
        for p in torch.where(d['split'] == SPLITS[split])[0].tolist():
            initial = d['initial'][p]
            if initial[0] not in (0, 1):
                raise ValueError('only positive_exterior and collision_state starts are supported')
            mode = int(initial[0])
            if limit_per_mode is not None and used[mode] >= limit_per_mode:
                continue
            used[mode] += 1
            index = slice(self.offsets[p], self.offsets[p+1])
            if self.order is not None:
                index = self.order[index]
            begin, end = d['offsets'][p:p+2].tolist()
            yield {'features': d['features'][begin:end], 'initial': initial,
                   'horizon': d['horizon'][p], 'distance': d['distance'][index],
                   'hit': d['hit'][index], 'profile': torch.tensor(p)}


def geometry(features, initial, config):
    if initial.shape != (4,) or initial[0] not in (0, 1) or not torch.isfinite(initial[:2]).all():
        raise ValueError('invalid A/B initial condition')
    geo = base_geometry(features, config, positive_only=True)
    if not torch.isclose(initial[1], features[0, 0], rtol=1e-9, atol=1e-10):
        raise ValueError('initial b0 differs from the mean profile')
    if initial[0] == 1:
        db0 = features[0, 2]/features[0, 4].exp()
        if (not torch.isfinite(initial).all()
                or not torch.isclose(initial[2], -initial[1], rtol=1e-9, atol=1e-10)
                or initial[3]+db0 <= 0):
            raise ValueError('surface starts require f(0)=0 and an outward g0>0')
    geo['entry'] = initial[None]
    # The start mode is observed; no entry-sign prior is used in this pipeline.
    del geo['log_prior']
    return geo


def prepare_profile(raw, config, curve_points=513):
    if curve_points < 2:
        raise ValueError('at least two survival queries are required')
    geo = geometry(raw['features'], raw['initial'], config)
    hit, distance = raw['hit'], raw['distance']
    if (hit.dtype != torch.bool or hit.ndim != 1 or not len(hit) or hit.shape != distance.shape
            or not torch.isfinite(distance).all() or (distance <= 0).any()
            or (distance > geo['horizon']+1e-9).any()
            or not torch.isclose(distance[~hit], geo['horizon'], rtol=1e-9, atol=1e-10).all()
            or not torch.isclose(raw['horizon'], geo['horizon'], rtol=1e-9, atol=1e-10)):
        raise ValueError('invalid first-downcrossing/censoring labels')
    q = torch.linspace(0, 1, curve_points, dtype=torch.float64)*geo['horizon']
    distance = distance.clamp_max(geo['horizon'])
    # Censored endpoints repeat thousands of times; integrate each query once.
    unique, inverse = torch.unique(torch.cat((distance, q)), sorted=True, return_inverse=True)
    combined = {k: v[..., inverse] for k, v in queries(geo, unique).items()}
    count = len(hit)
    row = {k: geo[k] for k in ('tokens', 'left', 'entry', 'widths', 'horizon', 'mass')}
    row['events'] = {k: v[..., :count] for k, v in combined.items()}
    row['curve'] = {k: v[..., count:] for k, v in combined.items()}
    row['reference_T'] = 1-torch.searchsorted(distance[hit].sort().values, q, right=True).double()/count
    row.update(hit=hit, mode=raw['initial'][0].long(), profile=raw['profile'])
    return row


def batch_profiles(profiles, indices=None):
    if not profiles or (indices is not None and len(indices) != len(profiles)):
        raise ValueError('invalid profile batch')
    batch = {k: pad_sequence([p[k].transpose(0, 1) for p in profiles],
                            batch_first=True).transpose(1, 2) for k in ('tokens', 'left', 'mass')}
    batch['widths'] = pad_sequence([p['widths'] for p in profiles], batch_first=True)
    for k in ('entry', 'horizon', 'reference_T', 'mode', 'profile'):
        batch[k] = torch.stack([p[k] for p in profiles])
    selected = [p['hit'] if indices is None else p['hit'][indices[i]] for i, p in enumerate(profiles)]
    if any(not len(hit) for hit in selected):
        raise ValueError('empty label selection')
    batch['hit'] = pad_sequence(selected, batch_first=True)
    batch['valid'] = pad_sequence([torch.ones_like(hit) for hit in selected], batch_first=True)
    batch['events'] = {}
    for key in profiles[0]['events']:
        values = [p['events'][key] if indices is None else p['events'][key][..., indices[i]]
                  for i, p in enumerate(profiles)]
        if key == 'cell':
            batch['events'][key] = pad_sequence(values, batch_first=True)
        else:
            batch['events'][key] = pad_sequence([v.T for v in values], batch_first=True).transpose(1, 2)
    batch['curve'] = {k: torch.stack([p['curve'][k] for p in profiles]) for k in profiles[0]['curve']}
    return batch


@torch.no_grad()
def query_model(model, features, initial, q, config=None):
    from first_hit_se.causal_residual import branch_values
    geo = geometry(features, initial, config or model.config)
    batch = {k: geo[k][None] for k in ('tokens', 'left', 'entry', 'mass')}
    delta = model(batch) if model is not None else torch.zeros_like(batch['mass'])
    H, rate = branch_values(delta, batch, {k: v[None] for k, v in queries(geo, q).items()})
    return {'T': (-H[0, 0]).exp(), 'H': H[0, 0], 'log_hazard': rate[0, 0], 'delta': delta[0, 0]}
