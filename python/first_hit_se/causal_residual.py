"""A causal, fixed-physical-step correction to the analytic local hazard."""
from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass
class CausalConfig:
    hidden: int = 16
    embedding: int = 16
    width: int = 32
    history: bool = True
    step: float = .125
    integration_step: float = .015625
    quadrature_order: int = 8

    def __post_init__(self):
        if min(self.hidden, self.embedding, self.width) < 1 or not 0 < self.step <= 1:
            raise ValueError('invalid causal network size or physical step')
        if not 0 < self.integration_step <= self.step or self.quadrature_order < 2:
            raise ValueError('invalid integration settings')


class CausalResidual(nn.Module):
    def __init__(self, config=None, *, entry_size=2):
        super().__init__()
        self.config = config or CausalConfig()
        c = self.config
        self.initial = nn.Sequential(nn.Linear(entry_size, c.hidden), nn.SiLU(), nn.Linear(c.hidden, c.hidden))
        self.segment = nn.Sequential(nn.Linear(7, c.embedding), nn.SiLU(),
                                     nn.Linear(c.embedding, c.embedding), nn.SiLU())
        self.gru = nn.GRU(c.embedding, c.hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(c.hidden+3, c.width), nn.SiLU(), nn.Linear(c.width, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def configuration(self):
        return asdict(self.config)

    def initial_state(self, entry):
        return self.initial(entry.to(self.head[0].weight.dtype))

    def correction(self, state, left):
        return self.head(torch.cat((state, left.to(state.dtype)), -1)).squeeze(-1).double()

    def advance(self, state, completed_segment):
        """Call only after the current interval has been consumed."""
        embedded = self.segment(completed_segment.to(state.dtype))
        return self.gru(embedded[:, None], state[None])[1][0]

    def forward(self, batch):
        # Padded suffixes cannot affect any entering state of a valid interval.
        B, K, L, _ = batch['tokens'].shape
        start = self.initial_state(batch['entry']).reshape(B*K, -1)
        if self.config.history:
            embedded = self.segment(batch['tokens'].to(start.dtype)).reshape(B*K, L, -1)
            propagated, _ = self.gru(embedded, start[None])
            entering = torch.cat((start[:, None], propagated[:, :-1]), 1)
        else:
            entering = start[:, None].expand(-1, L, -1)
        return self.correction(entering.reshape(B, K, L, -1), batch['left'])

    def active_parameters(self):
        blocks = (self.initial, self.segment, self.gru, self.head) if self.config.history else (self.initial, self.head)
        return sum(p.numel() for block in blocks for p in block.parameters())


def branch_values(delta, batch, query, prefix=None):
    factor = delta.exp()
    if prefix is None:
        prefix = torch.nn.functional.pad((batch['mass']*factor).cumsum(-1), (1, 0))
    index = query['cell'][:, None].expand(-1, delta.shape[1], -1)
    H = prefix.gather(-1, index)+query['partial']*factor.gather(-1, index)
    rate = query['local_log_rate']+delta.gather(-1, index)
    return H, rate


def prediction(model, batch):
    delta = model(batch) if model is not None else torch.zeros_like(batch['mass'])
    prefix = torch.nn.functional.pad((batch['mass']*delta.exp()).cumsum(-1), (1, 0))
    H, log_rate = branch_values(delta, batch, batch['events'], prefix)
    log_prior = batch['log_prior']
    branch = (batch['initial_sign'] < 0).long()[:, None]
    training_nll = H.gather(1, branch).squeeze(1)-torch.where(
        batch['hit'], log_rate.gather(1, branch).squeeze(1), 0.)
    log_T = torch.logsumexp(log_prior[:, :, None]-H, 1)
    log_pdf = torch.logsumexp(log_prior[:, :, None]-H+log_rate, 1)
    curve_H = branch_values(delta, batch, batch['curve'], prefix)[0]
    T = torch.logsumexp(log_prior[:, :, None]-curve_H, 1).exp()
    return {'delta': delta, 'training_nll': training_nll,
            'distance_nll': torch.where(batch['hit'], -log_pdf, -log_T), 'T': T}


def objective(model, batch, weights):
    p = prediction(model, batch)
    nll = p['training_nll'].mean()
    # Width weighting integrates delta^2 over q/Q, including a short last cell.
    energy = (p['delta'].square()*batch['widths'][:, None]).sum(-1)/batch['horizon'][:, None]
    regularizer = (energy*batch['log_prior'].exp()).sum(-1).mean()
    curve = (p['T']-batch['reference_T']).square().mean()
    loss = nll+weights['residual_l2_weight']*regularizer+weights['transmittance_weight']*curve
    return {'loss': loss, 'nll': nll, 'residual_l2': regularizer, 'transmittance_mse': curve}
