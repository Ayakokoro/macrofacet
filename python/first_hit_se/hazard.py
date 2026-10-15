"""Distance-only ablation: a local GP baseline, direct hazard, and residual hazard.

All distances are q=s/ell, starting at the medium entry. The two networks have
identical parameters and predict log rates on the same grid. The only difference
is the addition of the analytic local log hazard in the residual model.

The local baseline uses the original GP mean derivative, not the unit gradient
of the renderer's local_tangent material. It is a local-conditioning approximation
to the underlying continuous SE GP; the training teacher remains Tungsten-linear.
"""
from dataclasses import asdict, dataclass
from functools import lru_cache
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from renewal.model import _mlp
from .model import combine, validate_profile
from .spline import local_profile, resample_profile


def log_negative_part_mean(mean):
    """log E[max(-X,0)] for X ~ N(mean,1), including far positive tails."""
    mean = mean.double()
    log_phi = -.5*mean.square()-.5*math.log(2*math.pi)
    negative = torch.logaddexp(log_phi, (-mean).clamp_min(1e-300).log()
                               + torch.special.log_ndtr(-mean))
    x = mean.clamp(0, 26)
    # Mills ratio via erfcx avoids multiplying an underflow by an overflow.
    remainder = 1-x*math.sqrt(math.pi/2)*torch.special.erfcx(x/math.sqrt(2))
    positive = log_phi+remainder.log()
    # 1 - x*Phi(-x)/phi(x) = x^-2 (1 - 3/x^2 + 15/x^4 - ...).
    inv = mean.clamp_min(26).reciprocal().square()
    polynomial = 1+inv*(-3+inv*(15+inv*(-105+inv*(945+inv*(-10395+inv*135135)))))
    tail = log_phi+inv.log()+polynomial.log()
    return torch.where(mean <= 0, negative, torch.where(mean <= 26, positive, tail))


def local_log_hazard(b, db):
    """Positive-side local rate, per unit q; negate b and db for the other side."""
    b = b.double()
    return (-.5*b.square()-.5*math.log(2*math.pi)
            - torch.special.log_ndtr(b)+log_negative_part_mean(db))


@lru_cache(maxsize=16)
def _quadrature(order):
    return np.polynomial.legendre.leggauss(order)


def local_bin_log_rates(features, bins=256, quadrature_order=8):
    """Average the analytic local rate in each bin using Gauss-Legendre quadrature.

    Keeping the bin integral (rather than the midpoint rate) preserves optical
    depth. Integrals split at every cubic-profile boundary: mean derivatives may
    jump there, so a quadrature rule must not straddle those boundaries.
    """
    horizon = features[:, 4].exp().sum()
    nodes, weights = (torch.as_tensor(x.copy(), dtype=torch.float64, device=features.device)
                      for x in _quadrature(quadrature_order))
    regular = torch.linspace(0, 1, bins+1, dtype=torch.float64, device=features.device)*horizon
    profile_ends = features[:, 4].exp().cumsum(0)[:-1]
    edges = torch.unique(torch.cat((regular, profile_ends.clamp(0, horizon))), sorted=True)
    left, width = edges[:-1], edges[1:]-edges[:-1]
    index = ((left+width/2)/(horizon/bins)).floor().long().clamp_max(bins-1)
    q = left[:, None]+width[:, None]*(nodes+1)/2
    b, db = local_profile(features.double(), q.flatten())
    signs = features.new_tensor([1., -1.])[:, None, None]
    rates = local_log_hazard(signs*b.reshape(len(left), -1), signs*db.reshape(len(left), -1))
    log_mass = torch.logsumexp(rates+(weights/2).log(), -1)+(width/(horizon/bins)).log()
    index = index.expand(2, -1)
    peak = log_mass.new_full((2, bins), -math.inf).scatter_reduce(1, index, log_mass, reduce='amax')
    total = torch.zeros_like(peak).scatter_add(1, index, (log_mass-peak.gather(1, index)).exp())
    return peak+total.log()


def cumulative_hazard(log_rates, horizon):
    horizon = torch.as_tensor(horizon, dtype=log_rates.dtype, device=log_rates.device)
    return F.pad((log_rates.exp()*(horizon/log_rates.shape[-1])[..., None]).cumsum(-1), (1, 0))


def piecewise_distribution(log_rates, q, horizon, prefix=None):
    """Evaluate piecewise-constant hazards; batch dimensions broadcast, last is q."""
    horizon = torch.as_tensor(horizon, dtype=log_rates.dtype, device=log_rates.device)
    prefix = cumulative_hazard(log_rates, horizon) if prefix is None else prefix
    q, scale = torch.broadcast_tensors(q, horizon[..., None])
    shape = torch.broadcast_shapes(log_rates.shape[:-1], q.shape[:-1])
    q = q.expand(*shape, q.shape[-1])
    width = (scale/log_rates.shape[-1]).expand_as(q)
    index = (q/width).floor().long().clamp(0, log_rates.shape[-1]-1)
    log_rate = log_rates.expand(*shape, log_rates.shape[-1]).gather(-1, index)
    start = prefix.expand(*shape, prefix.shape[-1]).gather(-1, index)
    partial = (q-index*width).clamp_min(0)
    return start+partial*log_rate.exp(), log_rate


def inverse_piecewise(log_rates, horizon, uniforms, prefix=None):
    """Sample censored first-hit distances. Escape returns q=Q with hit=False."""
    horizon = torch.as_tensor(horizon, dtype=log_rates.dtype, device=log_rates.device)
    prefix = cumulative_hazard(log_rates, horizon) if prefix is None else prefix
    target = -torch.log1p(-uniforms)
    shape = torch.broadcast_shapes(log_rates.shape[:-1], target.shape[:-1])
    prefix = prefix.expand(*shape, prefix.shape[-1])
    log_rates = log_rates.expand(*shape, log_rates.shape[-1])
    target = target.expand(*shape, target.shape[-1])
    hit = target < prefix[..., -1:]
    index = torch.searchsorted(prefix[..., 1:].contiguous(), target.contiguous(), right=True)
    index = index.clamp_max(log_rates.shape[-1]-1)
    # An escape needs no division by a potentially underflowed rate.
    offset = torch.where(hit, target-prefix.gather(-1, index), 0.)
    width = horizon[..., None]/log_rates.shape[-1]
    partial = offset/log_rates.gather(-1, index).exp().clamp_min(1e-300)
    q = index*width+torch.minimum(partial.clamp_min(0), width)
    return torch.where(hit, q, horizon[..., None]), hit


@dataclass
class HazardConfig:
    mode: str = 'direct'
    profile_points: int = 256
    bins: int = 256
    controls: int = 64
    hidden: int = 32
    width: int = 64
    quadrature_order: int = 8

    def __post_init__(self):
        if self.mode not in ('direct', 'residual'):
            raise ValueError('hazard mode must be direct or residual')
        if min(self.profile_points, self.bins, self.controls, self.quadrature_order) < 2:
            raise ValueError('hazard grids and quadrature need at least two points')
        if min(self.hidden, self.width) < 1 or self.controls > self.bins:
            raise ValueError('invalid hazard network size')


class HazardNetwork(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or HazardConfig()
        c = self.config
        self.encoder = _mlp(3*c.profile_points+1, c.width, c.hidden)
        self.hazard_head = nn.Linear(c.hidden, c.controls)
        nn.init.zeros_(self.hazard_head.weight)
        nn.init.zeros_(self.hazard_head.bias)
        # Linear interpolation of control values at bin-center coordinates.
        at = ((torch.arange(c.bins, dtype=torch.float64)+.5)/c.bins)*(c.controls-1)
        left = at.floor().long().clamp_max(c.controls-2)
        self.register_buffer('_left', left, persistent=False)
        self.register_buffer('_fraction', at-left, persistent=False)

    def configuration(self):
        return asdict(self.config)

    def encode(self, fixed, sign):
        grid = fixed[..., :-1].reshape(*fixed.shape[:-1], self.config.profile_points, 3)
        signed = torch.cat((grid[..., :2]*sign[..., None, None], grid[..., 2:]), -1)
        inputs = torch.cat((signed.flatten(-2), fixed[..., -1:]), -1)
        z = self.encoder(inputs.to(self.hazard_head.weight.dtype))
        return self.hazard_head(z)

    def log_rates(self, fixed, signs, baseline=None):
        raw = self.encode(fixed, signs).double()
        interpolated = raw[..., self._left]*(1-self._fraction)+raw[..., self._left+1]*self._fraction
        if self.config.mode == 'residual':
            if baseline is None or baseline.shape != interpolated.shape:
                raise ValueError('residual hazard needs the matching local log rates')
            return baseline+interpolated
        return interpolated

    def query(self, features, queries):
        validate_profile(features)
        horizon = features[:, 4].exp().sum()
        if queries.ndim != 1 or not len(queries) or not torch.isfinite(queries).all():
            raise ValueError('queries must be a nonempty finite vector')
        if (queries < 0).any() or (queries > horizon+1e-10).any():
            raise ValueError('query outside profile')
        fixed = resample_profile(features, self.config.profile_points)
        signs = features.new_tensor([1., -1.])
        baseline = None
        if self.config.mode == 'residual':
            baseline = local_bin_log_rates(features, self.config.bins, self.config.quadrature_order)
        rates = self.log_rates(fixed.expand(2, -1), signs, baseline)
        H, log_rate = piecewise_distribution(rates, queries.clamp_max(horizon)[None], horizon)
        return combine(features, {'H': H, 'log_hazard': log_rate})


def batch_predictions(model, batch):
    """Both entry-sign branches share all parameters; the sign prior is analytic."""
    signs = batch['fixed'].new_tensor([1., -1.]).expand(len(batch['fixed']), -1)
    if model is None:
        log_rates = batch['baseline']
    else:
        log_rates = model.log_rates(batch['fixed'][:, None].expand(-1, 2, -1), signs, batch['baseline'])
    H, log_rate = piecewise_distribution(log_rates, batch['distance'][:, None], batch['horizon'][:, None])
    log_prior = torch.special.log_ndtr(signs*batch['b0'][:, None])
    log_prior -= torch.logsumexp(log_prior, -1, keepdim=True)
    log_mass = log_prior[:, :, None]-H
    log_T = torch.logsumexp(log_mass, 1)
    log_density = torch.logsumexp(log_mass+log_rate, 1)
    branch = (batch['initial_sign'] < 0).long()[:, None]
    selected_H = H.gather(1, branch).squeeze(1)
    selected_rate = log_rate.gather(1, branch).squeeze(1)
    selected_prior = log_prior.gather(1, branch.squeeze(1))
    hit = batch['hit']
    # Entry signs are observed in the existing data, including censored rays.
    conditional_nll = selected_H-torch.where(hit, selected_rate, 0.)
    return {'training_nll': conditional_nll, 'complete_distance_nll': conditional_nll-selected_prior,
            'distance_nll': torch.where(hit, -log_density, -log_T),
            'log_T': log_T, 'log_density': log_density, 'log_rates': log_rates}
