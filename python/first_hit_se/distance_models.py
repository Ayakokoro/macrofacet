"""Common probability contract for the distance-only representation experiment."""
import torch

from .hazard import HazardConfig, HazardNetwork, inverse_piecewise, piecewise_distribution
from .spline import SplineConfig, SplineFirstHitNetwork, rational_quadratic, spline_parameters


MODES = ('direct', 'residual', 'cdf')


class DistanceSpline(SplineFirstHitNetwork):
    """Reuse the tested spline transform, without unused gradient parameters."""
    def __init__(self, config):
        super().__init__(config)
        del self.mixture_head

    def query(self, features, queries):
        return super().query(features, queries, with_mixture=False)


def make_model(config, mode, seed):
    torch.manual_seed(seed)
    c = config['model']
    if mode == 'cdf':
        return DistanceSpline(SplineConfig(
            profile_points=c['profile_points'], bins=config['spline_bins'],
            hidden=c['hidden'], width=c['width']))
    return HazardNetwork(HazardConfig(mode=mode, **c))


def latent(model, fixed, sign):
    grid = fixed[..., :-1].reshape(*fixed.shape[:-1], model.config.profile_points, 3)
    signed = torch.cat((grid[..., :2]*sign[..., None, None], grid[..., 2:]), -1)
    inputs = torch.cat((signed.flatten(-2), fixed[..., -1:]), -1)
    return model.encoder(inputs.to(next(model.parameters()).dtype))


def parameters(model, batch):
    signs = batch['fixed'].new_tensor([1., -1.]).expand(len(batch['fixed']), -1)
    fixed = batch['fixed'][:, None].expand(-1, 2, -1)
    if model is None:
        return batch['baseline']
    if isinstance(model, DistanceSpline):
        return model.encode(fixed, signs)[1]
    return model.log_rates(fixed, signs, batch['baseline'])


def branches(model, encoded, q, horizon):
    if isinstance(model, DistanceSpline):
        H, log_rate, _ = model.branch_distribution(encoded[:, :, None], q[:, None],
                                                   horizon[:, None, None])
        return H, log_rate
    return piecewise_distribution(encoded, q[:, None], horizon[:, None])


def predictions(model, batch):
    encoded = parameters(model, batch)
    H, log_rate = branches(model, encoded, batch['distance'], batch['horizon'])
    signs = batch['b0'].new_tensor([1., -1.])
    log_prior = torch.special.log_ndtr(batch['b0'][:, None]*signs)
    log_prior -= torch.logsumexp(log_prior, -1, keepdim=True)
    log_T = torch.logsumexp(log_prior[:, :, None]-H, 1)
    log_density = torch.logsumexp(log_prior[:, :, None]-H+log_rate, 1)
    index = (batch['initial_sign'] < 0).long()[:, None]
    conditional = H.gather(1, index).squeeze(1)-torch.where(
        batch['hit'], log_rate.gather(1, index).squeeze(1), 0.)
    return {'training_nll': conditional, 'distance_nll': torch.where(
        batch['hit'], -log_density, -log_T),
        'complete_distance_nll': conditional-log_prior.gather(1, index.squeeze(1)),
        'encoded': encoded, 'log_prior': log_prior, 'log_T': log_T,
        'log_density': log_density}


def transmittance(model, encoded, q, horizon, log_prior):
    H, _ = branches(model, encoded, q, horizon)
    return torch.logsumexp(log_prior[:, :, None]-H, 1).exp()


def sample_distance(model, fixed, sign, baseline, horizon, uniforms):
    """Selected sign branches; the final axis of uniforms indexes samples."""
    if isinstance(model, DistanceSpline):
        raw = model.encode(fixed, sign)[1]
        x, y, d, _, log_hit = spline_parameters(raw, model.config)
        probability = log_hit.exp()[:, None]
        hit = uniforms < probability
        u = (uniforms/probability.clamp_min(1e-300)).clamp(0, 1)
        q = rational_quadratic(u, x[:, None], y[:, None], d[:, None], inverse=True)[0]
        return torch.where(hit, q*horizon[:, None], horizon[:, None]), hit
    rates = baseline if model is None else model.log_rates(fixed, sign, baseline)
    return inverse_piecewise(rates, horizon, uniforms)
