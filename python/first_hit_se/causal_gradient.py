"""Conditional crossing-speed heads on a frozen causal distance representation.

The left-limit state convention makes a query at a recurrent boundary identical
whether the profile ends there or continues. No horizon or future input enters
the head. Distribution arithmetic and sampling reuse the existing tested code.
"""
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .causal_data import geometry, queries
from .causal_residual import branch_values
from .gradient_cdf import make_head
from .spline import SplineConfig, local_profile, spline_parameters


@dataclass
class GradientConfig:
    hidden: int = 16
    width: int = 64
    size: int = 8
    kind: str = 'mixture'
    history: bool = True


class CausalGradient(nn.Module):
    def __init__(self, config, seed):
        super().__init__()
        self.config = config
        self.net = make_head(config.kind, config.hidden, config.width, config.size, seed).net
        self.spline_config = SplineConfig(bins=config.size) if config.kind == 'cdf' else None

    def forward(self, points):
        z = points['z'] if self.config.history else points['entry']
        q, phase, sign, b, db = (points[k] for k in ('q', 'phase', 'sign', 'b', 'db'))
        local = torch.stack((phase, q.log1p(), (sign*b).asinh(), (sign*db).asinh()), -1)
        raw = self.net(torch.cat((z, local.to(z.dtype)), -1))
        if self.config.kind == 'mixture':
            weight, delta, scale = raw.chunk(3, -1)
            return F.log_softmax(weight, -1), -sign[:, None]*db[:, None]+delta, F.softplus(scale)+.01
        x, y, derivative, _, _ = spline_parameters(raw, self.spline_config)
        scale = F.softplus(raw[..., 0].double()-sign*db)+.05
        return x, y, derivative, scale

    def configuration(self):
        return asdict(self.config)


def entering_states(backbone, geo):
    start = backbone.initial_state(geo['entry'])
    embedded = backbone.segment(geo['tokens'].to(start.dtype))
    states = backbone.gru(embedded, start[None])[0]
    return torch.cat((start[:, None], states[:, :-1]), 1), start


def state_indices(q, step, count):
    # At q=0 use the entry state; at a positive boundary use its left limit.
    index = (torch.ceil(q/step).long()-1).clamp(0, count-1)
    phase = ((q-index*step)/step).clamp(0, 1)
    return index, phase


@torch.no_grad()
def query_points(backbone, features, q, sign):
    geo = geometry(features, backbone.config)
    if q.shape != sign.shape or (q < 0).any() or (q > geo['horizon']+1e-9).any():
        raise ValueError('invalid conditional gradient query')
    if not torch.isfinite(q).all() or not ((sign == 1) | (sign == -1)).all():
        raise ValueError('invalid query distance or entry sign')
    q = q.clamp_max(geo['horizon'])
    states, entry = entering_states(backbone, geo)
    cell, phase = state_indices(q, backbone.config.step, states.shape[1])
    branch = (sign < 0).long()
    b, db = local_profile(features, q)
    return {'z': states[branch, cell], 'entry': entry[branch], 'q': q,
            'phase': phase, 'sign': sign, 'b': b, 'db': db}


@torch.no_grad()
def cache_profile(backbone, raw):
    geo = geometry(raw['features'], backbone.config)
    q = raw['distance'].clamp_max(geo['horizon'])
    states, entry = entering_states(backbone, geo)
    cell, phase = state_indices(q, backbone.config.step, states.shape[1])
    b, db = local_profile(raw['features'], q)
    batch = {k: geo[k][None] for k in ('tokens', 'left', 'entry', 'mass')}
    H, rate = branch_values(backbone(batch), batch, {k: v[None] for k, v in queries(geo, q).items()})
    branch = (raw['initial_sign'] < 0).long()
    index = torch.arange(len(q))
    log_T = torch.logsumexp(geo['log_prior'][:, None]-H[0], 0)
    complete_hit_nll = H[0, branch, index]-rate[0, branch, index]-geo['log_prior'][branch]
    joint_distance = torch.where(raw['hit'], complete_hit_nll, -log_T)
    return {'states': states, 'entry': entry, 'cell': cell, 'q': q, 'phase': phase,
        'b': b, 'db': db, 'sign': raw['initial_sign'], 'hit': raw['hit'], 'w': raw['speed'],
        'horizon': geo['horizon'], 'joint_distance_sum': joint_distance.sum()}


def points(profiles, indices=None):
    fields = {k: [] for k in ('z', 'entry', 'q', 'phase', 'sign', 'b', 'db', 'w', 'ray', 'fraction')}
    samples = 0
    for i, p in enumerate(profiles):
        index = torch.arange(len(p['hit']), device=p['hit'].device) if indices is None else indices[i]
        samples += len(index)
        selected = index[p['hit'][index]]
        branch = (p['sign'][selected] < 0).long()
        fields['z'].append(p['states'][branch, p['cell'][selected]])
        fields['entry'].append(p['entry'][branch])
        for k in ('q', 'phase', 'sign', 'b', 'db', 'w'):
            fields[k].append(p[k][selected])
        fields['ray'].append(torch.full_like(selected, i))
        fields['fraction'].append(p['q'][selected]/p['horizon'])
    return {**{k: torch.cat(v) for k, v in fields.items()}, 'samples': samples}
