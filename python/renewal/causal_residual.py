"""SE residual hazard with observed A/B start conditions, without a sign mixture.

A: f(0)>0, with the value and derivative unobserved.
B: f(0)=0 and a known outward derivative g0>0; the event is the next downcrossing.
The positive-side local hazard is a reference in both modes. The GRU learns
its correction conditional on the start and the completed mean-profile prefix.
"""
import torch

from first_hit_se.causal_residual import CausalConfig, CausalResidual, branch_values


FORMAT = 'renewal_causal_residual'
VERSION = 1
CONDITIONING = {
    'modes': ['positive_exterior', 'collision_state'],
    'initial': ['surface_mode', 'b0', 'known_value_residual', 'known_derivative_residual'],
    'derivative_residual': 'ell*g0/sigma - b_prime(0)',
    'encoding': 'asinh; unknown A residuals masked',
    'event': 'first_downcrossing_after_start',
    'baseline': 'positive_side_local_hazard',
}


class RenewalResidual(CausalResidual):
    def __init__(self, config=None):
        super().__init__(config, entry_size=4)

    def initial_state(self, entry):
        # A's latent starting value/derivative must never become observations.
        residuals = torch.where(entry[..., :1] == 1, entry[..., 2:], 0.)
        known = torch.cat((entry[..., :2], residuals), -1)
        return super().initial_state(known.asinh())


def prediction(model, batch):
    delta = model(batch) if model is not None else torch.zeros_like(batch['mass'])
    if delta.shape[1] != 1:
        raise ValueError('A/B conditions require one conditional distribution per profile')
    prefix = torch.nn.functional.pad((batch['mass']*delta.exp()).cumsum(-1), (1, 0))
    H, rate = (v[:, 0] for v in branch_values(delta, batch, batch['events'], prefix))
    nll = H-torch.where(batch['hit'], rate, 0.)
    curve_H = branch_values(delta, batch, batch['curve'], prefix)[0][:, 0]
    return {'delta': delta, 'distance_nll': nll, 'H': H, 'log_hazard': rate,
            'T': (-curve_H).exp()}


def objective(model, batch, weights):
    p = prediction(model, batch)
    # Each profile gets equal weight, even when label counts differ.
    nll = (p['distance_nll'].masked_fill(~batch['valid'], 0.).sum(-1)
           / batch['valid'].sum(-1)).mean()
    energy = (p['delta'][:, 0].square()*batch['widths']).sum(-1)/batch['horizon']
    regularizer = energy.mean()
    curve = (p['T']-batch['reference_T']).square().mean()
    loss = nll+weights['residual_l2_weight']*regularizer+weights['transmittance_weight']*curve
    return {'loss': loss, 'nll': nll, 'residual_l2': regularizer, 'transmittance_mse': curve}


def load_model(path, device='cpu'):
    saved = torch.load(path, map_location=device, weights_only=True)
    if (saved.get('format'), saved.get('version'), saved.get('conditioning')) != (FORMAT, VERSION, CONDITIONING):
        raise ValueError('not an A/B conditional residual GRU checkpoint')
    model = RenewalResidual(CausalConfig(**saved['model_config'])).to(device)
    model.load_state_dict(saved['model_state'])
    model.eval()
    return model, saved
