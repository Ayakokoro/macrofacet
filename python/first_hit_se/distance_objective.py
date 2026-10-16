"""Training-only residual regularization and empirical survival supervision."""
import math

import torch

from .distance_models import transmittance


def objective_weights(settings=None, mode='residual'):
    settings = settings or {}
    names = ('residual_l2_weight', 'transmittance_weight')
    if set(settings)-set(names):
        raise ValueError('unknown distance objective setting')
    result = {name: float(settings.get(name, 0.)) for name in names}
    if any(not math.isfinite(value) or value < 0 for value in result.values()):
        raise ValueError('distance objective weights must be finite and nonnegative')
    if result['residual_l2_weight'] and mode != 'residual':
        raise ValueError('residual output regularization requires a residual hazard model')
    return result


def distance_objective(model, batch, prediction, weights):
    """Reuse the likelihood's encoded rates; no second network forward pass.

    The residual penalty averages squared log-rate corrections over normalized
    distance, weights the two signs by their analytic entrance probabilities,
    then averages rays equally. It is not a weight-decay penalty.

    Curve targets use all training realizations of each ray, including escapes,
    independent of the realization subsample used for this batch's likelihood.
    """
    nll = prediction['training_nll'].mean()
    zero = nll.new_zeros(())
    residual_l2, curve_mse = zero, zero
    if weights['residual_l2_weight']:
        if model.config.mode != 'residual':
            raise ValueError('residual output regularization requires residual log rates')
        delta = prediction['encoded']-batch['baseline'].detach()
        branch_energy = delta.square().mean(-1)
        residual_l2 = (branch_energy*prediction['log_prior'].detach().exp()).sum(-1).mean()
    if weights['transmittance_weight']:
        target = batch['reference_T'].detach()
        unit = torch.linspace(0, 1, target.shape[-1], dtype=target.dtype, device=target.device)
        q = unit[None]*batch['horizon'][:, None]
        T = transmittance(model, prediction['encoded'], q, batch['horizon'], prediction['log_prior'])
        curve_mse = (T-target).square().mean()
    # Keep the disabled path identical to the original objective.
    loss = nll
    if weights['residual_l2_weight']:
        loss = loss+weights['residual_l2_weight']*residual_l2
    if weights['transmittance_weight']:
        loss = loss+weights['transmittance_weight']*curve_mse
    return {'loss': loss, 'nll': nll, 'residual_l2': residual_l2, 'transmittance_mse': curve_mse}
