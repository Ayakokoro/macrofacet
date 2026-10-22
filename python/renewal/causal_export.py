"""Export a self-contained A/B joint checkpoint for the C++ scalar renderer."""
from pathlib import Path

from .causal_gradient import load_joint, TARGET
from .causal_residual import CONDITIONING
from .collect import digest, write_json


def export_joint(checkpoint: Path, output: Path):
    backbone, head, saved = load_joint(checkpoint, 'cpu')
    if backbone.config.quadrature_order != 8:
        raise ValueError('C++ joint inference requires eight-point quadrature')

    def weights(module):
        result = {}
        for name, tensor in module.state_dict().items():
            if not tensor.is_floating_point() or not tensor.isfinite().all():
                raise ValueError(f'invalid joint tensor: {name}')
            value = tensor.detach().cpu().float().contiguous()
            result[name] = {'shape': list(value.shape), 'values': value.flatten().tolist()}
        return result

    bundle = {
        'format': 'macrofacet.causal_renewal', 'version': 1,
        'kernel': 'squared_exponential', 'profile_mode': 'cubic', 'activation_dtype': 'float32',
        'conditioning': CONDITIONING, 'target': TARGET,
        'distance_convention': 'q=distance_from_birth/ell', 'mean_convention': 'b=m/sigma',
        'state_convention': 'entering_fixed_q_interval_left_limit_gradient',
        'gru_convention': 'pytorch_rzn_reset_after',
        'model_config': backbone.configuration(), 'gradient_config': head.configuration(),
        'backbone': weights(backbone), 'gradient': weights(head),
        'checkpoint_sha256': digest(checkpoint), 'selected_epoch': saved.get('epoch'),
    }
    write_json(output, bundle)
    return {'bundle': str(output), 'checkpoint_sha256': bundle['checkpoint_sha256'],
            'parameters': sum(len(t['values']) for group in ('backbone', 'gradient')
                              for t in bundle[group].values())}
