"""Positive conditional speed distributions with direct inverse sampling.

CDF = RQS(1-exp(-w/scale)); its derivative is trained by likelihood. The
exponential base retains unbounded support and a stable log-density in the tail.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from renewal.model import _mlp, mixture_cdf, mixture_log_prob
from .distance_probe import GradientHead
from .spline import SplineConfig, rational_quadratic, spline_parameters


class GradientCDF(nn.Module):
    def __init__(self, hidden, width=64, bins=8):
        super().__init__()
        self.config = SplineConfig(bins=bins)
        self.net = _mlp(hidden+4, width, 3*bins+2)
        # Uniform spline, with an exponential base at initialization.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        with torch.no_grad():
            self.net[-1].bias[1+2*bins:] = math.log(math.expm1(1-self.config.minimum_derivative))

    def forward(self, z, q, horizon, sign, b, db):
        local = torch.stack((q/horizon, q.log1p(), (sign*b).asinh(), (sign*db).asinh()), -1)
        raw = self.net(torch.cat((z, local.to(z.dtype)), -1))
        x, y, derivative, _, _ = spline_parameters(raw, self.config)
        scale = F.softplus(raw[..., 0].double()-sign*db)+.05
        return x, y, derivative, scale


def spline_log_prob(params, w):
    x, y, derivative, scale = params
    ratio = w.clamp_min(0)/scale
    u = -torch.expm1(-ratio)
    jacobian = rational_quadratic(u, x, y, derivative)[1]
    return torch.where(w >= 0, jacobian-scale.log()-ratio, -torch.inf)


def spline_cdf(params, w):
    x, y, derivative, scale = params
    u = -torch.expm1(-w.clamp_min(0)/scale)
    return rational_quadratic(u, x, y, derivative)[0]


def spline_quantile(params, uniform):
    x, y, derivative, scale = params
    u = rational_quadratic(uniform.clamp(0, 1), x, y, derivative, inverse=True)[0]
    largest = torch.nextafter(torch.ones_like(u), torch.zeros_like(u))
    w = -scale*torch.log1p(-torch.minimum(u, largest))
    return torch.where(uniform >= 1, torch.inf, w)


def inverse_log_ndtr(log_probability):
    """Normal quantile supporting log probabilities below floating-point underflow."""
    normal = torch.special.ndtri(log_probability.exp().clamp_min(1e-300))
    small = log_probability < math.log(1e-300)
    if small.any():
        target = log_probability[small]
        value = -torch.sqrt(-2*target)
        for _ in range(7):
            log_cdf = torch.special.log_ndtr(value)
            derivative = torch.exp(-.5*value.square()-.5*math.log(2*math.pi)-log_cdf)
            value = value-(log_cdf-target)/derivative
        normal = normal.clone()
        normal[small] = value
    return normal


def sample_mixture_components(params, component_uniform, quantile_uniform):
    """Select a truncated-normal component, then invert its conditional tail.

    This avoids comparing a direct spline inverse against iterative mixture-CDF
    bisection; both methods use direct sampling in the timing experiment.
    """
    weights, means, scales = normalized_mixture(params)
    cumulative = weights.exp().cumsum(-1)
    index = torch.searchsorted(cumulative.contiguous(), component_uniform[..., None].contiguous(), right=True)
    index = index.clamp_max(weights.shape[-1]-1)
    mean, scale = (value.gather(-1, index).squeeze(-1) for value in (means, scales))
    log_tail = torch.log1p(-quantile_uniform)+torch.special.log_ndtr(mean/scale)
    return (mean-scale*inverse_log_ndtr(log_tail)).clamp_min(0)


def make_head(kind, hidden, width, size, seed):
    torch.manual_seed(seed)
    if kind == 'mixture':
        return GradientHead(hidden, width, size)
    if kind == 'cdf':
        return GradientCDF(hidden, width, size)
    raise ValueError(kind)


def log_prob(kind, params, w):
    return spline_log_prob(params, w) if kind == 'cdf' else mixture_log_prob(*normalized_mixture(params), w)


def cdf(kind, params, w):
    return (spline_cdf(params, w) if kind == 'cdf'
            else mixture_cdf(*normalized_mixture(params), w)).clamp(0, 1)


def normalized_mixture(params):
    # The existing network emits float32 log weights. Normalize in float64 so
    # likelihood, CDF, and component sampling share exactly the same weights.
    weights, means, scales = (p.double() for p in params)
    return weights-torch.logsumexp(weights, -1, keepdim=True), means, scales


def sample(kind, params, uniforms):
    return (spline_quantile(params, uniforms[..., 0]) if kind == 'cdf'
            else sample_mixture_components(params, uniforms[..., 0], uniforms[..., 1]))
