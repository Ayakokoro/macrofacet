"""Generate instrumented copies, asserting every edit against current sources.

The diagnostic links these objects before the static libraries, so the original
wavefront/backend objects are not extracted. Production sources remain untouched.
No CUDA synchronization is added. NVTX ranges attribute actual CUDA kernels in
Nsight; host timers on async calls alone are NOT GPU execution measurements.
"""
from pathlib import Path
import hashlib
import json
import sys

root, output = map(Path, sys.argv[1:])
output.mkdir(parents=True, exist_ok=True)
hashes = {}

def write(name, content):
    path = output / name
    if not path.exists() or path.read_text(encoding='utf-8') != content:
        path.write_text(content, encoding='utf-8')

def load(relative):
    data = (root / relative).read_bytes()
    hashes[relative] = hashlib.sha256(data).hexdigest()
    return '#include "RenewalProfile.h"\n' + data.decode('utf-8-sig')

def replace(source, old, new):
    assert source.count(old) == 1, (old, source.count(old))
    return source.replace(old, new)

def wrap(source, expression, label):
    return replace(source, expression,
                   f'renewal_profile::measure("{label}",[&]() {{ return {expression}; }})')

s = load('src/integrator/RenewalWavefront.cpp')
s = replace(s, '    for (;;) {', '    for (;;) {\n        { renewal_profile::Scope schedule("cpu.schedule");')
s = wrap(s, 'medium.beginFlight(slot.ray,slot.gradient)', 'cpu.mean_profile')
s = replace(s, '        slot.flight.reset(); slot.newSample = true;',
            '        if (slot.flight) renewal_profile::count("constructed_segments",slot.flight->mean.constructedSegmentCount());\n'
            '        slot.flight.reset(); slot.newSample = true;')
s = replace(s, '                slot.flight.reset();',
            '                renewal_profile::count("constructed_segments",slot.flight->mean.constructedSegmentCount());\n'
            '                slot.flight.reset();')
s = replace(s, '        if (activeIds.empty()) break;', '        }\n        if (activeIds.empty()) break;')
s = wrap(s, 'network.initialize(initializeIds,firstSegments,starts)', 'batch.initialize')
s = wrap(s, 'network.evaluate(activeIds,activeSegments)', 'batch.segment')
s = replace(s, '        for (std::size_t i = 0; i < activeIds.size(); ++i) {',
            '        { renewal_profile::Scope cpu("cpu.hazard_and_decision");\n'
            '        for (std::size_t i = 0; i < activeIds.size(); ++i) {')
s = replace(s, '        if (!hitIds.empty()) {', '        }\n        if (!hitIds.empty()) {')
s = wrap(s, 'network.mixture(hitIds,hitCoordinates)', 'batch.mixture')
s = replace(s, '            for (std::size_t i = 0; i < hitIds.size(); ++i) {',
            '            renewal_profile::Scope scatter("cpu.scatter");\n'
            '            for (std::size_t i = 0; i < hitIds.size(); ++i) {')
write('RenewalWavefrontProfile.cpp', s)

s = load('src/learned/RenewalTorchBackend.cpp')
s = replace(s, '        const auto ids = indices(slots);',
            '        renewal_profile::Scope backend("segment.backend");\n'
            '        const auto ids = renewal_profile::measure("segment.upload_ids",[&]() { return indices(slots); });')
s = wrap(s, 'states_.index_select(0,ids)', 'segment.gather_state')
s = wrap(s, 'input(features,5)', 'segment.upload_features')
s = wrap(s, 'transformInput(featuresDevice,0,4)', 'segment.input_transform')
s = wrap(s, 'at::silu(encoder2_(at::silu(encoder0_(networkFeatures))))', 'segment.encoder')
s = wrap(s, 'at::cat({old,embedded},1)', 'segment.context')
s = wrap(s, 'hazard_(context)', 'segment.hazard_mlp')
s = wrap(s, 'at::softplus(logits,1,20)', 'segment.output_transform')
s = wrap(s, 'at::gru_cell(embedded,old,gruInput_.weight,gruHidden_.weight,\n                                      gruInput_.bias,gruHidden_.bias)', 'segment.gru')
s = wrap(s, 'enteringContext_.index_copy_(0,ids,context)', 'segment.store_context')
s = wrap(s, 'states_.index_copy_(0,ids,next)', 'segment.store_state')
s = wrap(s, 'at::cat({rates,at::isfinite(next).all(1,true).to(at::kFloat)},1)', 'segment.validate_pack')
s = wrap(s, 'download(packedDevice)', 'segment.download_wait')
write('RenewalTorchProfile.cpp', s)

s = load('src/learned/RenewalBatchSession.cpp')
start = s.index('std::vector<std::array<double,4>> RenewalBatchSession::evaluate(')
end = s.index('std::vector<RenewalSpeedMixture> RenewalBatchSession::mixture(', start)
body = s[start:end]
body = replace(body, '    auto& input = impl_->input;\n', '')
body = replace(body, '    impl_->validateSlots(slots, segments.size(), 1);',
               '    auto& input = impl_->input;\n'
               '    { renewal_profile::Scope preprocessing("segment.cpu_preprocess");\n'
               '    impl_->validateSlots(slots, segments.size(), 1);')
body = replace(body, '    const auto raw = impl_->engine->evaluate(slots, input);',
               '    }\n    const auto raw = impl_->engine->evaluate(slots, input);\n'
               '    renewal_profile::Scope postprocessing("segment.cpu_postprocess");')
s = s[:start] + body + s[end:]
s = wrap(s, 'impl_->engine->initialize(slots, input)', 'initialize.backend')
s = wrap(s, 'impl_->engine->mixture(slots, input)', 'mixture.backend')
write('RenewalBatchSessionProfile.cpp', s)

s = load('src/gpss/RayMeanProfile.cpp')
s = replace(s, 's.mean->queryRayPoint(s.origin,s.direction,begin,maximumEnd)',
            'renewal_profile::measurePointQuery([&]() { return '
            's.mean->queryRayPoint(s.origin,s.direction,begin,maximumEnd); })')
s = wrap(s, 'mean.requireFullRayCoverage(origin,direction,maximumDistance)', 'profile.coverage')
s = wrap(s, 'mean.appendRayBreakpoints(origin,direction,0.0,maximumDistance,knots)', 'profile.breakpoints')
s = wrap(s, 'partition(std::move(knots),maximumStep)', 'profile.partition')
s = replace(s, '    for (std::size_t i=1; i<knots.size(); ++i) {\n        const double lo=knots[i-1], hi=knots[i], width=hi-lo;',
            '    { renewal_profile::Scope sampling("profile.sample_and_fit");\n'
            '    for (std::size_t i=1; i<knots.size(); ++i) {\n        const double lo=knots[i-1], hi=knots[i], width=hi-lo;')
s = replace(s, '    return RayMeanProfile(std::move(segments));\n}',
            '    }\n    return RayMeanProfile(std::move(segments));\n}')
write('RayMeanProfileProfile.cpp', s)
(output / 'source_hashes.json').write_text(json.dumps(hashes, indent=2), encoding='utf-8')
