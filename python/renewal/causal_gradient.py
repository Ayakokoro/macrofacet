"""Conditional hit-speed distributions on a frozen A/B residual GRU."""
import torch

from first_hit_se.causal_gradient import CausalGradient, GradientConfig, entering_states, state_indices
from first_hit_se.causal_residual import CausalConfig, branch_values
from first_hit_se.gradient_cdf import cdf, log_prob, sample
from first_hit_se.spline import local_profile
from .causal_data import geometry
from .causal_residual import CONDITIONING, RenewalResidual


FORMAT = 'renewal_causal_joint'
VERSION = 1
TARGET = {
    'variable': 'w = -ell * directional_derivative_at_hit / sigma',
    'support': 'w > 0; no upper truncation',
    'physical_derivative': '-sigma * w / ell',
    'event': 'first_downcrossing_after_start',
    'state': 'left_limit_of_fixed_q_interval',
}


class HitGradient(CausalGradient):
    def __init__(self, config, seed=0):
        if (config.kind != 'mixture' or not config.history
                or min(config.hidden, config.width, config.size) < 1):
            raise ValueError('A/B hit gradients require a recurrent positive Gaussian mixture')
        super().__init__(config, seed)

    def forward(self, points):
        # Both A/B modes approach their next collision from the positive side.
        return super().forward({**points, 'sign': torch.ones_like(points['q'])})


def raw_profiles(dataset, split, limit_per_mode=None):
    """Attach speeds with exactly the same index permutation as the distance labels."""
    d = dataset.data
    if 'speed' not in d or d['speed'].shape != d['hit'].shape or not d['speed'].is_floating_point():
        raise ValueError('hit-gradient training requires packed speed labels')
    rows = []
    for raw in dataset.profiles(split):
        p = int(raw['profile'])
        index = slice(dataset.offsets[p], dataset.offsets[p+1])
        if dataset.order is not None:
            index = dataset.order[index]
        rows.append({**raw, 'speed': d['speed'][index]})
    if limit_per_mode is None:
        return rows
    if limit_per_mode < 1:
        raise ValueError('profile limit must be positive')
    selected = []
    for mode in (0, 1):
        group = [p for p in rows if int(p['initial'][0]) == mode]
        # A short check must exercise hit labels, even when the first rays miss.
        priority = [p for p in group if p['hit'].any()][:1]
        priority += [p for p in group if not p['hit'].any()][:1]
        ids = {int(p['profile']) for p in priority}
        priority += [p for p in group if int(p['profile']) not in ids]
        selected.extend(priority[:limit_per_mode])
    return selected


@torch.no_grad()
def query_points(backbone, features, initial, q):
    if not backbone.config.history:
        raise ValueError('hit-gradient head requires a recurrent distance backbone')
    geo = geometry(features, initial, backbone.config)
    if (q.ndim != 1 or not torch.isfinite(q).all() or (q < 0).any()
            or (q > geo['horizon']+1e-9).any()):
        raise ValueError('gradient query outside the profile')
    q = q.clamp_max(geo['horizon'])
    states, entry = entering_states(backbone, geo)
    cell, phase = state_indices(q, backbone.config.step, states.shape[1])
    b, db = local_profile(geo['features'], q)
    return {'z': states[0, cell], 'entry': entry.expand(len(q), -1),
            'q': q, 'phase': phase, 'b': b, 'db': db}


@torch.no_grad()
def cache_profile(backbone, raw):
    if not backbone.config.history:
        raise ValueError('hit-gradient head requires a recurrent distance backbone')
    geo = geometry(raw['features'], raw['initial'], backbone.config)
    q, hit, w = raw['distance'], raw['hit'], raw['speed']
    if (hit.dtype != torch.bool or hit.ndim != 1 or not len(hit) or hit.shape != q.shape
            or w.shape != hit.shape or not torch.isfinite(q).all() or (q <= 0).any()
            or (q > geo['horizon']+1e-9).any()
            or not torch.isclose(q[~hit], geo['horizon'], rtol=1e-9, atol=1e-10).all()
            or not torch.isclose(raw['horizon'], geo['horizon'], rtol=1e-9, atol=1e-10)
            or not torch.isfinite(w[hit]).all() or (w[hit] <= 0).any()):
        raise ValueError('invalid hit-gradient or censoring labels')
    q = q.clamp_max(geo['horizon'])
    states, entry = entering_states(backbone, geo)
    cell, phase = state_indices(q, backbone.config.step, states.shape[1])
    b, db = local_profile(geo['features'], q)
    unique, inverse = torch.unique(q, sorted=True, return_inverse=True)
    from first_hit_se.causal_data import queries
    query = {k: v[..., inverse][None] for k, v in queries(geo, unique).items()}
    delta = backbone.correction(states, geo['left'])[None]
    H, rate = (v[0, 0] for v in branch_values(delta, {'mass': geo['mass'][None]}, query))
    distance_nll = H-torch.where(hit, rate, 0.)
    if not torch.isfinite(distance_nll).all():
        raise FloatingPointError('nonfinite frozen distance likelihood')
    return {'states': states[0], 'entry': entry[0], 'cell': cell, 'q': q, 'phase': phase,
            'b': b, 'db': db, 'hit': hit.clone(), 'w': w.double().clone(),
            'mode': raw['initial'][0].long(), 'profile': raw['profile'],
            'distance_nll_sum': distance_nll.sum()}


def points(profiles, indices=None):
    if not profiles or (indices is not None and len(indices) != len(profiles)):
        raise ValueError('invalid gradient profile batch')
    fields = {k: [] for k in ('z', 'entry', 'q', 'phase', 'b', 'db', 'w', 'mode')}
    samples = 0
    for i, p in enumerate(profiles):
        index = (torch.arange(len(p['hit']), device=p['hit'].device) if indices is None
                 else indices[i].to(p['hit'].device))
        samples += len(index)
        selected = index[p['hit'][index]]
        fields['z'].append(p['states'][p['cell'][selected]])
        fields['entry'].append(p['entry'][None].expand(len(selected), -1))
        fields['mode'].append(p['mode'].expand(len(selected)))
        for key in ('q', 'phase', 'b', 'db', 'w'):
            fields[key].append(p[key][selected])
    return {**{k: torch.cat(v) for k, v in fields.items()}, 'samples': samples}


def to_device(points, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in points.items()}


def loss(head, batch):
    # Censored samples contribute zero to the gradient term. Their distance
    # likelihood is constant in this second stage and is included in evaluation.
    if batch['samples'] <= 0:
        raise ValueError('empty training batch')
    if not len(batch['w']):
        return None
    return -log_prob('mixture', head(batch), batch['w']).sum()/batch['samples']


def load_joint(path, device='cpu'):
    saved = torch.load(path, map_location=device, weights_only=True)
    if (saved.get('format'), saved.get('version'), saved.get('conditioning'), saved.get('target')) != (
            FORMAT, VERSION, CONDITIONING, TARGET):
        raise ValueError('not an A/B residual GRU with a hit-gradient head')
    backbone = RenewalResidual(CausalConfig(**saved['backbone_config'])).to(device)
    backbone.load_state_dict(saved['backbone_state'])
    head = HitGradient(GradientConfig(**saved['head_config'])).to(device)
    head.load_state_dict(saved['head_state'])
    if head.config.hidden != backbone.config.hidden or not backbone.config.history:
        raise ValueError('gradient and distance representations disagree')
    return backbone.requires_grad_(False).eval(), head.eval(), saved
