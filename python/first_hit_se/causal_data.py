"""Fixed-q recurrent intervals and accurately integrated local probabilities.

Quadrature splits at every original cubic boundary and at a fine physical grid.
Partial integrals end at the actual query, never at the full ray's endpoint.
"""
import math

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from .hazard import local_log_hazard
from .model import validate_profile
from .spline import local_profile


def integrate(features, left, right, order):
    """Intervals must each lie inside one cubic; endpoints may coincide."""
    nodes, weights = (torch.as_tensor(a.copy(), dtype=features.dtype, device=features.device)
                      for a in np.polynomial.legendre.leggauss(order))
    width = right-left
    q = left[:, None]+width[:, None]*(nodes+1)/2
    b, db = local_profile(features, q.flatten())
    signs = features.new_tensor([1., -1.])[:, None, None]
    rates = local_log_hazard(signs*b.reshape(len(left), -1), signs*db.reshape(len(left), -1))
    return (rates.exp()*(weights/2)).sum(-1)*width


def geometry(features, config):
    validate_profile(features)
    features = features.double()
    Q = features[:, 4].exp().sum()
    count = max(1, math.ceil(float(Q)/config.step))
    left = torch.arange(count, dtype=features.dtype, device=features.device)*config.step
    right = (left+config.step).clamp_max(Q)
    points = torch.stack((left, (left+right)/2, right), -1)
    b, db = (v.reshape(count, 3) for v in local_profile(features, points.flatten()))
    signs = features.new_tensor([1., -1.])[:, None, None]
    sb, sd = (signs*b).asinh(), (signs*db).asinh()
    tokens = torch.cat((sb, sd, (right-left).log1p()[None, :, None].expand(2, -1, 1)), -1)
    local = torch.cat((sb[:, :, :1], sd[:, :, :1], left.log1p()[None, :, None].expand(2, -1, 1)), -1)
    fine = torch.arange(math.ceil(float(Q)/config.integration_step), dtype=features.dtype,
                        device=features.device)*config.integration_step
    knots = torch.unique(torch.cat((fine, left, features[:, 4].exp().cumsum(0).clamp_max(Q), Q[None])), sorted=True)
    interval_left, interval_right = knots[:-1], knots[1:]
    cell = (torch.searchsorted(left, (interval_left+interval_right)/2, right=True)-1).clamp(0, count-1)
    counts = torch.bincount(cell, minlength=count)
    offset = torch.arange(len(cell), device=cell.device)-(counts.cumsum(0)-counts)[cell]
    slots = int(counts.max())
    masses = integrate(features, interval_left, interval_right, config.quadrature_order)
    packed = masses.new_zeros((2, count*slots)).scatter(1, (cell*slots+offset)[None].expand(2, -1), masses)
    cumulative = torch.nn.functional.pad(packed.reshape(2, count, slots).cumsum(-1), (1, 0))
    prior = torch.special.log_ndtr(features[0, 0]*features.new_tensor([1., -1.]))
    prior -= torch.logsumexp(prior, 0)
    return {'features': features, 'tokens': tokens, 'left': local, 'entry': local[:, 0, :2],
        'widths': right-left, 'horizon': Q, 'mass': cumulative[:, :, -1], 'log_prior': prior,
        '_knots': knots, '_cell': cell, '_offset': offset, '_cumulative': cumulative,
        '_order': config.quadrature_order}


def queries(geo, q):
    if q.ndim != 1 or not torch.isfinite(q).all() or (q < 0).any() or (q > geo['horizon']+1e-9).any():
        raise ValueError('query outside the profile')
    q = q.clamp_max(geo['horizon'])
    i = (torch.searchsorted(geo['_knots'], q, right=True)-1).clamp(0, len(geo['_cell'])-1)
    cell, offset = geo['_cell'][i], geo['_offset'][i]
    partial = geo['_cumulative'][:, cell, offset]+integrate(geo['features'], geo['_knots'][i], q, geo['_order'])
    b, db = local_profile(geo['features'], q)
    signs = q.new_tensor([1., -1.])[:, None]
    return {'cell': cell, 'partial': partial, 'local_log_rate': local_log_hazard(signs*b, signs*db)}


def to_device(value, device):
    return {k: to_device(v, device) for k, v in value.items()} if isinstance(value, dict) else value.to(device)


def prepare_profile(raw, config, curve_points=513):
    geo = geometry(raw['features'], config)
    q = torch.linspace(0, 1, curve_points, dtype=torch.float64)*geo['horizon']
    combined = queries(geo, torch.cat((raw['distance'], q)))
    count = len(raw['hit'])
    row = {k: geo[k] for k in ('tokens', 'left', 'entry', 'widths', 'horizon', 'mass', 'log_prior')}
    row['events'] = {k: v[..., :count] for k, v in combined.items()}
    row['curve'] = {k: v[..., count:] for k, v in combined.items()}
    observed_distance = raw['distance'].clamp_max(geo['horizon'])
    row['reference_T'] = 1-torch.searchsorted(observed_distance[raw['hit']].sort().values, q, right=True).double()/count
    row.update(hit=raw['hit'], initial_sign=raw['initial_sign'])
    return row


def batch_profiles(profiles, indices=None):
    batch = {}
    for key in ('tokens', 'left', 'mass'):
        batch[key] = pad_sequence([p[key].transpose(0, 1) for p in profiles], batch_first=True).transpose(1, 2)
    batch['widths'] = pad_sequence([p['widths'] for p in profiles], batch_first=True)
    for key in ('entry', 'horizon', 'log_prior', 'reference_T'):
        batch[key] = torch.stack([p[key] for p in profiles])
    for key in ('hit', 'initial_sign'):
        batch[key] = torch.stack([p[key] if indices is None else p[key][indices[i]] for i, p in enumerate(profiles)])
    for kind in ('events', 'curve'):
        batch[kind] = {key: torch.stack([p[kind][key] if indices is None or kind == 'curve'
            else p[kind][key][..., indices[i]] for i, p in enumerate(profiles)]) for key in profiles[0][kind]}
    return batch


@torch.no_grad()
def query_model(model, features, q, config=None):
    from .causal_residual import branch_values
    geo = geometry(features, config or model.config)
    query = queries(geo, q)
    batch = {k: geo[k][None] for k in ('tokens', 'left', 'entry', 'mass', 'horizon')}
    delta = model(batch) if model is not None else torch.zeros_like(batch['mass'])
    H, rate = branch_values(delta, batch, {k: v[None] for k, v in query.items()})
    log_T = torch.logsumexp(geo['log_prior'][None, :, None]-H, 1)[0]
    return {'T': log_T.exp(), 'H': H[0], 'log_hazard': rate[0], 'delta': delta[0]}
