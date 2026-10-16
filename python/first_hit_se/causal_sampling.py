"""Deployment contract and independent inverse sampler for the frozen causal GRU.

The recurrent grid is part of the model, not a renderer accuracy setting.
This module deliberately leaves all hash-locked training modules unchanged.
"""
import argparse
import json
from pathlib import Path

import torch

from renewal.collect import digest, write_json
from .causal_data import geometry, queries
from .causal_gradient import query_points
from .causal_gradient_study import load_head
from .causal_study import load_run
from .gradient_cdf import sample


def export_bundle(backbone_path, head_path, destination):
    backbone_path, head_path = Path(backbone_path).resolve(), Path(head_path).resolve()
    backbone, distance_saved = load_run(backbone_path)
    head, gradient_saved = load_head(head_path)
    protocol_path = head_path.parents[2]/'protocol.json'
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    if digest(protocol_path) != gradient_saved['identity']['protocol_sha256']:
        raise ValueError('gradient protocol has changed')
    # Verify the actual paired backbone, not just matching dimensions or seed.
    hashes = protocol.get('identity', protocol)['checkpoint_sha256']
    if digest(backbone_path) not in hashes.values():
        raise ValueError('backbone was not used to train this gradient head')
    if distance_saved['identity']['seed'] != gradient_saved['identity']['seed']:
        raise ValueError('backbone and gradient seeds do not match')
    if not backbone.config.history or not head.config.history or head.config.kind != 'mixture':
        raise ValueError('deployment currently supports the history GRU mixture head')
    if backbone.config.hidden != head.config.hidden or backbone.config.quadrature_order != 8:
        raise ValueError('incompatible hidden size or quadrature order')
    def weights(module):
        return {k: {'shape': list(v.shape), 'values': v.detach().cpu().flatten().tolist()}
                for k, v in module.state_dict().items()}
    bundle = {'format': 'macrofacet.causal_first_hit', 'version': 1,
        'event': 'unconditioned_first_zero', 'profile_mode': 'cubic',
        'kernel': 'squared_exponential', 'activation_dtype': 'float32',
        'distance_convention': 'q=distance_from_entry/ell', 'mean_convention': 'b=m/sigma',
        'gradient_convention': 'signed_derivative=-initial_sign*sigma/ell*W',
        'state_convention': 'entering_fixed_q_interval_left_limit_gradient',
        'model_config': backbone.configuration(), 'gradient_config': head.configuration(),
        'backbone': weights(backbone), 'gradient': weights(head),
        'provenance': {'backbone': str(backbone_path), 'gradient': str(head_path),
            'backbone_sha256': digest(backbone_path), 'gradient_sha256': digest(head_path),
            'gradient_protocol_sha256': digest(protocol_path),
            'label': 'finite-history GPIS first-zero and Tungsten-linear secant speed'}}
    write_json(Path(destination), bundle)
    return bundle


@torch.no_grad()
def sample_profile(backbone, head, features, uniforms, tolerance=1e-9):
    """Four open uniforms per sample: entry sign, survival, component, quantile.

    Vectorized reference inversion of the *trained* integrated hazard. Used to
    check the online C++ sampler, whose root solve only evaluates local hazard.
    """
    uniforms = uniforms.double()
    if uniforms.ndim != 2 or uniforms.shape[1] != 4 or not ((uniforms > 0) & (uniforms < 1)).all():
        raise ValueError('expected N x 4 finite open uniforms')
    if not tolerance > 0:
        raise ValueError('invalid root tolerance')
    geo = geometry(features, backbone.config)
    batch = {k: geo[k][None] for k in ('tokens', 'left', 'entry', 'mass')}
    delta = backbone(batch)[0]
    mass = geo['mass']*delta.exp()
    prefix = torch.nn.functional.pad(mass.cumsum(-1), (1, 0))
    branch = (uniforms[:, 0].log() >= geo['log_prior'][0]).long()
    sign = 1-2*branch
    target = -uniforms[:, 1].log()
    hit = target < prefix[branch, -1]
    lo = torch.zeros_like(target)
    hi = torch.full_like(target, geo['horizon'])
    for _ in range(80):
        if float((hi-lo).max()) <= tolerance:
            break
        mid = (lo+hi)/2
        query = queries(geo, mid)
        cell = query['cell']
        H = prefix[branch, cell]+delta[branch, cell].exp()*query['partial'][branch, torch.arange(len(mid))]
        lo = torch.where(H < target, mid, lo)
        hi = torch.where(H >= target, mid, hi)
    q = torch.where(hit, (lo+hi)/2, geo['horizon'])
    speed = torch.zeros_like(q)
    if hit.any():
        params = head(query_points(backbone, features, q[hit], sign[hit]))
        speed[hit] = sample(head.config.kind, params, uniforms[hit, 2:])
    return {'q': q, 'hit': hit, 'initial_sign': sign, 'speed': speed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backbone', required=True)
    parser.add_argument('--gradient', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    export_bundle(args.backbone, args.gradient, args.output)
    print(args.output)


if __name__ == '__main__':
    main()
