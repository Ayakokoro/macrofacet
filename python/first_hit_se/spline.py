"""Fixed-grid, sign-symmetric first-hit distribution with an escape atom.

The bounded rational-quadratic CDF follows Durkan et al., Neural Spline Flows
(https://arxiv.org/abs/1906.04032). This is a direct implementation of the
equations, with a separately evaluated upper tail for survival probabilities.
Only the selected entry-sign branch is encoded when sampling a camera ray.
"""
from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from renewal.model import _mlp, mean_at
from .model import validate_profile, combine, losses_from_prediction


@dataclass
class SplineConfig:
    architecture: str = "spline"
    profile_points: int = 128
    bins: int = 32
    hidden: int = 32
    width: int = 64
    mixture_width: int = 64
    components: int = 8
    sigma_floor: float = .01
    minimum_bin_width: float = .001
    minimum_bin_height: float = .00001
    minimum_derivative: float = .0001
    maximum_step: float = 1/32  # Legacy query API; the spline has no ODE steps.

    def __post_init__(self):
        if self.architecture != "spline":
            raise ValueError("expected spline architecture")
        if not (2 <= self.profile_points <= 1024 and 2 <= self.bins <= 256):
            raise ValueError("invalid fixed grid or spline bin count")
        if min(self.hidden, self.width, self.mixture_width, self.components) < 1:
            raise ValueError("network dimensions must be positive")
        if not (0 < self.minimum_bin_width*self.bins < 1 and
                0 < self.minimum_bin_height*self.bins < 1 and
                0 < self.minimum_derivative < 1 and 0 < self.sigma_floor < 1):
            raise ValueError("invalid spline or mixture floors")


def local_profile(features, q):
    widths = features[:, 4].exp()
    ends = widths.cumsum(0)
    index = torch.searchsorted(ends.contiguous(), q.contiguous()).clamp_max(len(features)-1)
    u = ((q-(ends[index]-widths[index]))/widths[index]).clamp(0, 1)
    return mean_at(features[index], u)


def resample_profile(features, points):
    validate_profile(features)
    horizon = features[:, 4].exp().sum()
    u = torch.linspace(0, 1, points, dtype=features.dtype, device=features.device)
    b, db = local_profile(features, u*horizon)
    return torch.cat((torch.stack((b.asinh(), db.asinh(), u), -1).flatten(), horizon.log1p()[None]))


def spline_parameters(raw, config):
    """Network outputs stay float32; probability arithmetic is float64."""
    raw = raw.double()
    k = config.bins
    widths = config.minimum_bin_width+(1-k*config.minimum_bin_width)*raw[..., 1:1+k].softmax(-1)
    heights = config.minimum_bin_height+(1-k*config.minimum_bin_height)*raw[..., 1+k:1+2*k].softmax(-1)
    derivatives = config.minimum_derivative+F.softplus(raw[..., 1+2*k:])
    # Pin both ends, including their derivatives with respect to the logits.
    x = torch.cat((torch.zeros_like(widths[..., :1]), widths.cumsum(-1)[..., :-1], torch.ones_like(widths[..., :1])), -1)
    y = torch.cat((torch.zeros_like(heights[..., :1]), heights.cumsum(-1)[..., :-1], torch.ones_like(heights[..., :1])), -1)
    return x, y, derivatives, -F.softplus(-raw[..., 0]), -F.softplus(raw[..., 0])


def rational_quadratic(value, x, y, derivatives, inverse=False):
    """Return transform, log |Jacobian| and upper CDF tail (forward only).

    Leading dimensions are broadcastable. Bin lookup never builds a value-by-
    bin comparison tensor; parameters may be expanded views across queries.
    """
    shape = torch.broadcast_shapes(value.shape, x.shape[:-1])
    value = value.expand(shape).double()
    x, y, derivatives = (a.expand((*shape, a.shape[-1])) for a in (x, y, derivatives))
    knots = y if inverse else x
    index = torch.searchsorted(knots.contiguous(), value[..., None].contiguous(), right=True).squeeze(-1)
    index = (index-1).clamp(0, x.shape[-1]-2)
    def at(a, offset=0):
        return a.gather(-1, (index+offset)[..., None]).squeeze(-1)
    x0, y0, x1, y1 = at(x), at(y), at(x, 1), at(y, 1)
    w, h = x1-x0, y1-y0
    d0, d1 = at(derivatives), at(derivatives, 1)
    slope = h/w
    bend = d0+d1-2*slope
    if inverse:
        offset = (value-y0).clamp_min(0)
        a = offset*bend+h*(slope-d0)
        b = h*d0-offset*bend
        c = -slope*offset
        discriminant = (b*b-4*a*c).clamp_min(0)
        theta = (2*c/(-b-discriminant.sqrt())).clamp(0, 1)
    else:
        theta = ((value-x0)/w).clamp(0, 1)
    cross = theta*(1-theta)
    denominator = slope+bend*cross
    log_pdf = 2*slope.log()+(d1*theta.square()+2*slope*cross+d0*(1-theta).square()).log()-2*denominator.log()
    if inverse:
        return x0+w*theta, -log_pdf, None
    fraction = (slope*theta.square()+d0*cross)/denominator
    tail = (1-y1)+h*(slope*(1-theta).square()+d1*cross)/denominator
    return y0+h*fraction, log_pdf, tail.clamp_min(0)


class SplineFirstHitNetwork(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or SplineConfig()
        c = self.config
        self.encoder = _mlp(3*c.profile_points+1, c.width, c.hidden)
        self.cdf_head = nn.Linear(c.hidden, 3*c.bins+2)
        self.mixture_head = _mlp(c.hidden+4, c.mixture_width, 3*c.components)
        # Start at a uniform conditional CDF; learn an unrestricted escape logit.
        nn.init.zeros_(self.cdf_head.weight)
        nn.init.zeros_(self.cdf_head.bias)
        with torch.no_grad():
            self.cdf_head.bias[1+2*c.bins:] = math.log(math.expm1(1-c.minimum_derivative))

    def configuration(self):
        return asdict(self.config)

    def encode(self, fixed, sign):
        grid = fixed[..., :-1].reshape(*fixed.shape[:-1], self.config.profile_points, 3)
        signed = torch.cat((grid[..., :2]*sign[..., None, None], grid[..., 2:]), -1)
        inputs = torch.cat((signed.flatten(-2), fixed[..., -1:]), -1)
        z = self.encoder(inputs.to(next(self.parameters()).dtype))
        return z, self.cdf_head(z)

    def mixture(self, z, q, horizon, sign, b, db):
        local = torch.stack((q/horizon, q.log1p(), (sign*b).asinh(), (sign*db).asinh()), -1)
        raw = self.mixture_head(torch.cat((z, local.to(z.dtype)), -1))
        weights, delta, scale = raw.chunk(3, -1)
        return F.log_softmax(weights, -1), -sign[..., None]*db[..., None]+delta, F.softplus(scale)+self.config.sigma_floor

    def branch_distribution(self, raw, q, horizon):
        x, y, derivatives, log_escape, log_hit = spline_parameters(raw, self.config)
        cdf, log_pdf, tail = rational_quadratic((q/horizon).clamp(0, 1), x, y, derivatives)
        log_T = torch.logaddexp(log_escape, log_hit+tail.clamp_min(1e-300).log())
        log_density = log_hit+log_pdf-horizon.log()
        return -log_T, log_density-log_T, cdf

    def query(self, features, queries, *, with_mixture=True, maximum_step=None):
        validate_profile(features)
        if queries.ndim != 1 or not len(queries) or not torch.isfinite(queries).all():
            raise ValueError("queries must be a nonempty finite vector")
        horizon = features[:, 4].exp().sum()
        if (queries < 0).any() or (queries > horizon+1e-10).any():
            raise ValueError("query outside profile")
        q = queries.clamp_max(horizon)
        fixed = resample_profile(features, self.config.profile_points)
        signs = features.new_tensor([1., -1.])
        z, raw = self.encode(fixed.expand(2, -1), signs)
        H, log_rate, _ = self.branch_distribution(raw[:, None], q[None], horizon)
        result = {"H": H, "log_hazard": log_rate}
        if with_mixture:
            b, db = local_profile(features, q)
            mix = self.mixture(z[:, None].expand(-1, len(q), -1), q.expand(2, -1), horizon,
                               signs[:, None].expand(-1, len(q)), b.expand(2, -1), db.expand(2, -1))
            result.update(zip(("log_weights", "means", "scales"), mix))
        return combine(features, result)

    def losses(self, features, distance, hit, initial_sign, speed):
        return losses_from_prediction(self.query(features, distance), distance, hit, initial_sign, speed)
