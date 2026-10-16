"""Complete the causal study, including its matched-capacity control."""
import copy

from renewal.collect import resolve, write_json
from .data import read_json
from .causal_study import run


def run_all(config_path):
    spec = read_json(resolve(config_path))
    if {v['name'] for v in spec['variants']} == {'local_matched_T'}:
        run(config_path, 'train')
        return run(config_path, 'evaluate')
    if {v['name'] for v in spec['variants']} != {'local_head_T', 'gru_nll', 'gru_T'}:
        raise ValueError('the full report requires the registered causal-study variants')
    capacity = copy.deepcopy(spec)
    capacity['output_directory'] = spec['output_directory'].rstrip('/\\')+'/capacity_control'
    capacity['model']['width'] = 128
    capacity['variants'] = [{'name': 'local_matched_T', 'history': False,
                             'residual_l2_weight': .01, 'transmittance_weight': 100.}]
    path = resolve(spec['output_directory'])/'capacity_config.json'
    if path.exists() and read_json(path) != capacity:
        raise ValueError('capacity-control configuration changed')
    write_json(path, capacity)
    for config, phase in ((config_path, 'train'), (path, 'train'),
                          (config_path, 'evaluate'), (path, 'evaluate'), (config_path, 'diagnose')):
        run(config, phase)
    return run(config_path, 'report')
