"""Instrument copies of the causal renderer without changing production code.

Host times include copies/waits. No additional CUDA synchronization is inserted.
"""
from pathlib import Path
import hashlib
import json
import sys

root, output = map(Path, sys.argv[1:])
output.mkdir(parents=True, exist_ok=True)
hashes = {}


def load(name):
    data = (root/name).read_bytes()
    hashes[name] = hashlib.sha256(data).hexdigest()
    return '#include "RenewalProfile.h"\n'+data.decode('utf-8-sig').replace('\r\n', '\n')


def replace(source, old, new):
    if source.count(old) != 1:
        raise ValueError((old, source.count(old)))
    return source.replace(old, new)


def wrap(source, expression, name):
    return replace(source, expression, f'renewal_profile::measure("{name}",[&]() {{ return {expression}; }})')


def write(name, source):
    path = output/name
    if not path.exists() or path.read_text(encoding='utf-8') != source:
        path.write_text(source, encoding='utf-8')


s = load('src/learned/CausalFirstHitModel.cpp')
s = wrap(s, 'makeCausalTorchBackend(model.impl_->weights,backend=="torch_cuda")', 'setup.device_backend')
s = wrap(s, 'backend.initialize(entries)', 'sample.initialize')
s = wrap(s, 'backend.correction(active,left)', 'sample.correction')
s = wrap(s, 'backend.advance(survivors,tokens)', 'sample.advance')
s = wrap(s, 'backend.mixture(hits,local)', 'sample.mixture')
s = replace(s, '        std::vector<float> left;\n        for (int slot:active)',
    '        renewal_profile::Scope stepScope("sample.step");\n'
    '        renewal_profile::count("active_rows",active.size());\n'
    '        if (active.size()<512) renewal_profile::count("small_steps",1);\n'
    '        std::vector<float> left;\n        for (int slot:active)')
begin = '        if (impl_->executor && active.size()>=RenewalCpuExecutor::parallelThreshold)'
s = replace(s, begin, '        { renewal_profile::Scope integrateScope("sample.local_integrate_and_root");\n'+begin)
s = replace(s, '        else integrateRows(0,active.size());', '        else integrateRows(0,active.size());\n        }')
s = replace(s, '        active=std::move(survivors);',
    '        renewal_profile::count("survivor_rows",survivors.size());\n        active=std::move(survivors);')
write('CausalFirstHitModelProfile.cpp', s)

s = load('src/learned/CausalFirstHitTorchBackend.cpp')
s = wrap(s, 'at::from_blob(const_cast<float*>(values.data()),{static_cast<int64_t>(values.size()/columns),columns},at::kFloat).to(device_)',
    'torch.feature_upload')
s = wrap(s, 'at::from_blob(const_cast<int*>(slots.data()),{static_cast<int64_t>(slots.size())},at::kInt).to(device_,at::kLong)',
    'torch.index_upload')
s = wrap(s, 'tensor.to(at::kCPU).contiguous()', 'torch.download_and_wait')
write('CausalFirstHitTorchProfile.cpp', s)

s = load('src/integrator/MacrofacetPathTracer.cpp')
s = wrap(s, 'session.sample(profiles,uniforms,tolerance)', 'render.first_hit_sampling')
s = wrap(s, 'std::make_unique<HybridFirstHitTransport>(prepared)', 'render.transport_setup')
first = '        executor.forRanges(count,[&](std::size_t a,std::size_t b) {'
if s.count(first) != 2:
    raise ValueError('expected profile and continuation CPU batches')
before, middle, after = s.split(first)
middle = middle.replace('        });', '        }); }', 1)
after = after.replace('        });', '        }); }', 1)
s = before+'        { renewal_profile::Scope scope("render.profiles");\n'+first+middle+\
    '        { renewal_profile::Scope scope("render.normals_and_continuation");\n'+first+after
write('CausalRenderProfile.cpp', s)
write('source_hashes.json', json.dumps(hashes, indent=2))
