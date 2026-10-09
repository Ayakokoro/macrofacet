"""Causal GRU, integrated Bernstein hazard, and positive truncated Gaussian mixture."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


@dataclass
class ModelConfig:
    embedding: int = 64
    hidden: int = 64
    hazard_width: int = 64
    mixture_width: int = 128
    components: int = 8
    sigma_floor: float = 0.01

    def __post_init__(self):
        if min(self.embedding, self.hidden, self.hazard_width, self.mixture_width, self.components) < 1:
            raise ValueError("positive network widths are required")
        if not 0 < self.sigma_floor < 1:
            raise ValueError("sigma_floor must be in (0,1) in normalized speed units")


def _mlp(n_in, width, n_out):
    return nn.Sequential(nn.Linear(n_in, width), nn.SiLU(), nn.Linear(width, width),
                         nn.SiLU(), nn.Linear(width, n_out))


def bernstein(u: torch.Tensor, degree: int) -> torch.Tensor:
    return torch.stack([math.comb(degree, j) * u**j * (1-u)**(degree-j)
                        for j in range(degree+1)], dim=-1)


def cumulative(logits: torch.Tensor, dx: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    rates = F.softplus(logits)
    coefficients = F.pad(torch.cumsum(rates, dim=-1), (1, 0)) * dx.unsqueeze(-1) / 4
    return (coefficients * bernstein(u, 4)).sum(-1)


def log_hazard(logits: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    # No artificial hazard floor. The asymptote avoids log(softplus(-1000)).
    log_rates = torch.where(logits < -20, logits, torch.log(F.softplus(logits.clamp_min(-20))))
    return torch.logsumexp(log_rates + torch.log(bernstein(u, 3)), dim=-1)


def mean_at(features: torch.Tensor, u: torch.Tensor):
    b0, b1, d0, d1, log_dx = features.unbind(-1)
    delta = b1-b0
    a = -2*delta + d0 + d1
    b = 3*delta - 2*d0 - d1
    value = ((a*u+b)*u+d0)*u+b0
    derivative = ((3*a*u+2*b)*u+d0) * torch.exp(-log_dx)
    return value, derivative


def mixture_log_prob(log_weights, means, scales, w):
    # Double precision keeps tail normalization and its gradients reliable.
    weights, mu, sigma, w = log_weights.double(), means.double(), scales.double(), w.double()
    a = mu / sigma
    t = w.unsqueeze(-1) / sigma
    regular = -0.5*(t-a)**2 - sigma.log() - 0.5*math.log(2*math.pi) - torch.special.log_ndtr(a)
    # Avoid subtracting two O(a^2) terms for strongly negative component means.
    tail = (-0.5*t*t + t*a - sigma.log() + 0.5*math.log(2/math.pi)
            - torch.special.erfcx((-a).clamp_min(0) / math.sqrt(2)).log())
    components = torch.where(a < -5, tail, regular)
    result = torch.logsumexp(weights + components, dim=-1)
    return torch.where(w > 0, result, torch.full_like(result, -torch.inf))


def mixture_cdf(log_weights, means, scales, w):
    a = means.double() / scales.double()
    shifted = (means.double() - w.double().unsqueeze(-1)) / scales.double()
    log_survival = (torch.special.log_ndtr(shifted) - torch.special.log_ndtr(a)).clamp_max(0)
    cdf = (log_weights.double().exp() * (-torch.expm1(log_survival))).sum(-1)
    return torch.where(w > 0, cdf, torch.zeros_like(cdf))


class RenewalNetwork(nn.Module):
    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config
        self.segment_encoder = nn.Sequential(nn.Linear(5, c.embedding), nn.SiLU(),
                                              nn.Linear(c.embedding, c.embedding), nn.SiLU())
        self.initial_encoder = _mlp(4, c.hidden, c.hidden)
        self.gru = nn.GRU(c.embedding, c.hidden, batch_first=True)
        self.hazard_head = _mlp(c.hidden+c.embedding, c.hazard_width, 4)
        self.mixture_head = _mlp(c.hidden+c.embedding+3, c.mixture_width, 3*c.components)

    def encode(self, features, initial, lengths):
        # A's hidden random Z0,D0 must not enter the network, even if a caller
        # accidentally populates those slots. B supplies known residuals.
        known = torch.cat((initial[:, :2], initial[:, 2:] * initial[:, :1]), dim=-1)
        dtype = self.segment_encoder[0].weight.dtype
        start = self.initial_encoder(torch.asinh(known).to(dtype))
        embedded = self.segment_encoder(torch.cat((torch.asinh(features[..., :4]),
                                                   features[..., 4:]), dim=-1).to(dtype))
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        propagated, _ = self.gru(packed, start.unsqueeze(0))
        propagated, _ = pad_packed_sequence(propagated, batch_first=True, total_length=features.shape[1])
        entering = torch.cat((start.unsqueeze(1), propagated[:, :-1]), dim=1)
        context = torch.cat((entering, embedded), dim=-1)
        return context, self.hazard_head(context)

    def mixture(self, context, features, u):
        b, db = mean_at(features, u)
        query = torch.stack((u, torch.asinh(b), torch.asinh(db)), dim=-1)
        parameters = self.mixture_head(torch.cat((context, query.to(context.dtype)), dim=-1))
        weights, delta, scale = parameters.chunk(3, dim=-1)
        return F.log_softmax(weights, dim=-1), -db.unsqueeze(-1)+delta, F.softplus(scale)+self.config.sigma_floor

    def losses(self, batch):
        features, lengths = batch["features"], batch["lengths"]
        context, logits = self.encode(features, batch["initial"], lengths)
        last = lengths.to(features.device)-1
        rows = torch.arange(len(last), device=features.device)
        dx = features[..., 4].exp()
        totals = dx * F.softplus(logits).sum(-1) / 4
        before = torch.arange(features.shape[1], device=features.device)[None, :] < last[:, None]
        H = (totals * before).sum(-1) + cumulative(logits[rows, last], dx[rows, last], batch["u"])
        hit = batch["hit"]
        log_h = torch.zeros_like(H)
        log_q = torch.zeros_like(H)
        pit = torch.full_like(H, math.nan)
        if hit.any():
            log_h[hit] = log_hazard(logits[rows[hit], last[hit]], batch["u"][hit]).to(H.dtype)
            mixture = self.mixture(context[rows[hit], last[hit]], features[rows[hit], last[hit]], batch["u"][hit])
            log_q[hit] = mixture_log_prob(*mixture, batch["speed"][hit]).to(H.dtype)
            with torch.no_grad():
                pit[hit] = mixture_cdf(*mixture, batch["speed"][hit]).to(H.dtype)
        return {"joint": H-log_h-log_q, "distance": H-log_h, "speed": -log_q,
                "pit": pit, "H": H}

    def configuration(self):
        return asdict(self.config)
