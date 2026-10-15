"""Shared positive/negative branches with either continuous state or the original GRU.

Geometry is a continuous piecewise cubic. Only the GRU baseline encodes segments.
Distances are q=s/ell, means b=m/sigma, and W=-initial_sign*d(F/sigma)/dq > 0.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from renewal.model import (ModelConfig as GRUConfig, RenewalNetwork, _mlp,
                           mean_at, cumulative, log_hazard, mixture_log_prob, mixture_cdf)


@dataclass
class ModelConfig:
    architecture: str = "rno"
    hidden: int = 16
    width: int = 64
    mixture_width: int = 128
    components: int = 8
    sigma_floor: float = 0.01
    maximum_step: float = 1/32

    def __post_init__(self):
        if self.architecture not in ("rno", "gru"):
            raise ValueError("architecture must be rno or gru")
        if min(self.hidden, self.width, self.mixture_width, self.components) < 1:
            raise ValueError("network dimensions must be positive")
        if not (0 < self.sigma_floor < 1 and 0 < self.maximum_step <= 1):
            raise ValueError("invalid mixture floor or normalized integration step")


def validate_profile(features):
    if features.ndim != 2 or features.shape[1] != 5 or len(features) == 0:
        raise ValueError("a nonempty five-column cubic profile is required")
    if not torch.isfinite(features).all() or not torch.isfinite(features[:, 4].exp()).all() or not (features[:, 4].exp() > 0).all():
        raise ValueError("nonfinite or degenerate mean profile")
    if len(features) > 1 and not torch.allclose(features[:-1, 1], features[1:, 0], atol=1e-8, rtol=1e-8):
        raise ValueError("first-zero models require a continuous mean, not point_linear tangents")


def log_softplus(x):
    return torch.where(x < -20, x, F.softplus(x.clamp_min(-20)).log())


def combine(features, result):
    signs = features.new_tensor([1., -1.])
    log_prior = torch.special.log_ndtr(signs * features[0, 0])
    # Normalize roundoff without clipping rare branches.
    log_prior = log_prior - torch.logsumexp(log_prior, 0)
    log_mass = log_prior[:, None] - result["H"]
    log_T = torch.logsumexp(log_mass, 0)
    log_density = torch.logsumexp(log_mass + result["log_hazard"], 0)
    return {**result, "log_prior": log_prior, "log_T": log_T,
            "T": log_T.exp(), "log_density": log_density,
            "hit_branch_probability": torch.softmax(log_mass + result["log_hazard"], 0)}


class FirstHitNetwork(nn.Module):
    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config
        if c.architecture == "gru":
            self.gru_model = RenewalNetwork(GRUConfig(embedding=c.width, hidden=c.hidden,
                hazard_width=c.width, mixture_width=c.mixture_width,
                components=c.components, sigma_floor=c.sigma_floor))
        else:
            self.initial_encoder = _mlp(2, c.width, c.hidden)
            self.dynamics = _mlp(c.hidden+3, c.width, c.hidden)
            self.hazard_head = _mlp(c.hidden+3, c.width, 1)
            self.mixture_head = _mlp(c.hidden+3, c.mixture_width, 3*c.components)

    def configuration(self):
        return asdict(self.config)

    def expand_hidden_from(self, source):
        """Embed an RNO in a larger state with identical initial predictions.

        New coordinates retain their random encoder/dynamics outputs. Their
        connections into the old dynamics and both heads initially vanish, so
        they can receive gradients through those connections during fine-tuning.
        """
        old, new = source.configuration(), self.configuration()
        if old['architecture'] != 'rno' or new['architecture'] != 'rno':
            raise ValueError('hidden expansion requires two RNO models')
        if new['hidden'] <= old['hidden'] or any(old[k] != new[k] for k in old if k != 'hidden'):
            raise ValueError('hidden expansion may only increase the state dimension')
        before, after = old['hidden'], new['hidden']
        state = self.state_dict()
        for name, value in source.state_dict().items():
            target = state[name]
            if target.shape == value.shape:
                target.copy_(value)
            elif name in ('dynamics.0.weight', 'hazard_head.0.weight', 'mixture_head.0.weight'):
                target.zero_()
                target[:, :before].copy_(value[:, :before])
                target[:, after:].copy_(value[:, before:])
            elif name in ('initial_encoder.4.weight', 'initial_encoder.4.bias',
                          'dynamics.4.weight', 'dynamics.4.bias'):
                target[:before].copy_(value)
            else:
                raise ValueError(f'unsupported hidden expansion parameter: {name}')
        self.load_state_dict(state)

    def _inputs(self, h, segment, u, q):
        b, db = mean_at(segment, torch.as_tensor(u, dtype=segment.dtype, device=segment.device))
        signs = segment.new_tensor([1., -1.]).reshape((2,)+(1,)*b.ndim)
        age = torch.as_tensor(q, dtype=segment.dtype, device=segment.device).log1p()
        local = torch.stack((torch.asinh(signs*b), torch.asinh(signs*db),
                             age.expand_as(signs*b)), -1).to(h.dtype)
        return torch.cat((h, local), -1), signs*db

    def _rhs(self, h, segment, u, q):
        inputs, _ = self._inputs(h, segment, u, q)
        raw = self.hazard_head(inputs).squeeze(-1)
        return self.dynamics(inputs), F.softplus(raw).double()

    def _step(self, h, H, segment, left, width, q, dt):
        def rhs(state, x):
            u = (x-left)/width
            u = u.clamp(0, 1) if isinstance(u, torch.Tensor) else min(1., max(0., u))
            return self._rhs(state, segment, u, x)
        dh = dt.to(h.dtype)[None, :, None] if isinstance(dt, torch.Tensor) else dt
        dH = dt[None, :] if isinstance(dt, torch.Tensor) else dt
        k1, a1 = rhs(h, q)
        k2, a2 = rhs(h + (dh/2)*k1, q+dt/2)
        k3, a3 = rhs(h + (dh/2)*k2, q+dt/2)
        k4, a4 = rhs(h + dh*k3, q+dt)
        return h + (dh/6)*(k1+2*k2+2*k3+k4), H + (dH/6)*(a1+2*a2+2*a3+a4)

    def _rno(self, features, queries, with_mixture, maximum_step):
        c = self.config
        dtype = self.initial_encoder[0].weight.dtype
        b0, db0 = mean_at(features[0], features.new_tensor(0.))
        signs = features.new_tensor([1., -1.])
        h = self.initial_encoder(torch.asinh(torch.stack((signs*b0, signs*db0), -1)).to(dtype))
        H = features.new_zeros(2)
        widths = features[:, 4].exp().detach().cpu().tolist()
        left = q = 0.
        segment = 0
        states, hazards, indices, starts, lengths, positions = [], [], [], [], [], []
        for target in queries.detach().cpu().tolist():
            # Queries never split the persistent integration grid. Reuse the
            # left state and evaluate all final partial RK4 steps as one batch.
            while True:
                end = left+widths[segment]
                right = min(q+maximum_step, end)
                if target <= right or (segment+1 == len(features) and target-end < 1e-10 and right == end):
                    break
                h, H = self._step(h, H, features[segment], left, widths[segment], q, right-q)
                q = right
                if q == end and segment+1 < len(features):
                    left = end; segment += 1
            states.append(h); hazards.append(H); indices.append(segment)
            starts.append(left); lengths.append(widths[segment]); positions.append(q)
        selected = features[indices]
        starts, lengths, positions = (features.new_tensor(v) for v in (starts, lengths, positions))
        h, H = self._step(torch.stack(states, 1), torch.stack(hazards, 1), selected,
            starts, lengths, positions, (queries-positions).clamp_min(0))
        inputs, db = self._inputs(h, selected, ((queries-starts)/lengths).clamp(0, 1), queries)
        result = {"H": H, "log_hazard": log_softplus(self.hazard_head(inputs).squeeze(-1)).double()}
        if with_mixture:
            weights, delta, scale = self.mixture_head(inputs).chunk(3, -1)
            result.update(log_weights=F.log_softmax(weights, -1), means=-db[..., None]+delta,
                          scales=F.softplus(scale)+c.sigma_floor)
        return result

    def _gru(self, features, queries, with_mixture):
        signs = features.new_tensor([1., -1.])
        both = features[None].repeat(2, 1, 1)
        both = torch.cat((both[..., :4]*signs[:, None, None], both[..., 4:]), -1)
        initial = features.new_zeros((2, 4))
        initial[:, 1] = signs*features[0, 0]
        context, logits = self.gru_model.encode(both, initial, torch.tensor([len(features)]*2))
        widths = features[:, 4].exp()
        ends = widths.cumsum(0)
        index = torch.searchsorted(ends.contiguous(), queries.contiguous()).clamp_max(len(features)-1)
        left = ends[index]-widths[index]
        u = ((queries-left)/widths[index]).clamp(0, 1)
        totals = widths[None]*F.softplus(logits).sum(-1)/4
        prefix = F.pad(totals.cumsum(-1), (1, 0))[:, :-1]
        result = {"H": prefix[:, index]+cumulative(logits[:, index], widths[index], u),
                  "log_hazard": log_hazard(logits[:, index], u).double()}
        if with_mixture:
            mix = self.gru_model.mixture(context[:, index], both[:, index], u.expand(2, -1))
            result.update(zip(("log_weights", "means", "scales"), mix))
        return result

    def query(self, features, queries, *, with_mixture=True, maximum_step=None):
        validate_profile(features)
        if queries.ndim != 1 or not len(queries) or not torch.isfinite(queries).all():
            raise ValueError("queries must be a nonempty finite vector")
        horizon = features[:, 4].exp().sum()
        if (queries < 0).any() or (queries > horizon+1e-10).any():
            raise ValueError("query is outside the profile")
        order = queries.argsort(stable=True)
        inverse = order.argsort()
        step = self.config.maximum_step if maximum_step is None else maximum_step
        if not math.isfinite(step) or step <= 0:
            raise ValueError("invalid ODE integration step")
        ordered = queries[order].clamp_max(horizon)
        unique, restore = torch.unique_consecutive(ordered, return_inverse=True)
        result = (self._rno(features, unique, with_mixture, step) if self.config.architecture == "rno"
                  else self._gru(features, unique, with_mixture))
        return combine(features, {k: v[:, restore][:, inverse] for k, v in result.items()})

    def losses(self, features, distance, hit, initial_sign, speed):
        result = self.query(features, distance)
        return losses_from_prediction(result, distance, hit, initial_sign, speed)


def losses_from_prediction(result, distance, hit, initial_sign, speed):
    if not torch.all((initial_sign == 1) | (initial_sign == -1)):
        raise ValueError("initial signs must be +1 or -1")
    branch = (initial_sign < 0).long()
    rows = torch.arange(len(distance), device=distance.device)
    H, log_rate = result["H"][branch, rows], result["log_hazard"][branch, rows]
    log_q = torch.zeros_like(H)
    pit = torch.full_like(H, math.nan)
    if hit.any():
        mix = tuple(result[k][branch[hit], rows[hit]] for k in ("log_weights", "means", "scales"))
        log_q[hit] = mixture_log_prob(*mix, speed[hit])
        with torch.no_grad():
            pit[hit] = mixture_cdf(*mix, speed[hit])
    complete = -result["log_prior"][branch]+H-hit*log_rate-log_q
    # At a first zero, signed gradient identifies its starting branch.
    marginal_joint = torch.where(hit, complete, -result["log_T"])
    distance_nll = torch.where(hit, -result["log_density"], -result["log_T"])
    return {"complete_joint": complete, "joint": marginal_joint, "distance": distance_nll,
            "speed": -log_q, "pit": pit, "prediction": result}
