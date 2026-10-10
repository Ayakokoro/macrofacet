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
    return '#include "RenewalProfile.h"\n' + data.decode('utf-8-sig').replace('\r\n','\n')

def replace(source, old, new):
    assert source.count(old) == 1, (old, source.count(old))
    return source.replace(old, new)

def wrap(source, expression, label):
    return replace(source, expression,
                   f'renewal_profile::measure("{label}",[&]() {{ return {expression}; }})')

s = load('src/integrator/RenewalWavefront.cpp')
for marker,label in [
        ('    const auto drainIncoming = [&]() {','cpu.ready_drain'),
        ('    const auto prepareFeatures = [&](const std::vector<int>& ready) {','cpu.prepare_features'),
        ('    const auto prepareBatch = [&](RenewalBatchRequests& requests, std::size_t activeCpu) {','cpu.schedule'),
        ('    const auto collectBatch = [&](const std::shared_ptr<BatchWork>& batch) {','cpu.dispatch_completed'),
        ('        enqueueRanges(ns,[&,batch](std::size_t begin,std::size_t end,std::vector<int>& ready) {','cpu.advance_work'),
        ('        enqueueRanges(batch->requests.mixtureSlots.size(),[&,batch](std::size_t begin,std::size_t end,std::vector<int>& ready) {','cpu.scatter_work'),
        ('        enqueueRanges(static_cast<std::size_t>(capacity),[&](std::size_t begin,std::size_t end,std::vector<int>& ready) {','cpu.initialize_work')]:
    s = replace(s,marker,marker+f'\n        renewal_profile::Scope scope("{label}");')
s = wrap(s, 'medium.beginFlight(slot.ray,slot.gradient)', 'cpu.mean_profile')
s = replace(s, '        slot.flight.reset();',
            '        if (slot.flight) renewal_profile::count("constructed_segments",slot.flight->mean.constructedSegmentCount());\n        slot.flight.reset();')
s = wrap(s, 'network.submitWavefrontAsync(batch->requests,preparedFeatures)', 'batch.enqueue')
s = wrap(s, 'network.collectInto(batch->ticket,batch->output)', 'batch.collect')
s = replace(s, '                            function(begin,end,ready);',
            '                            renewal_profile::Scope taskScope("cpu.worker_task");\n'
            '                            function(begin,end,ready);')
s = wrap(s, 'readyCondition.wait(lock,ready)', 'cpu.wait_ready')
s = wrap(s, 'inferenceCondition.wait(lock,ready)', 'inference.wait_ready')
write('RenewalWavefrontProfile.cpp', s)

s = load('src/learned/RenewalTorchBackend.cpp')
s = wrap(s, 'deviceInput.copy_(frame.hostInput.narrow(0,0,count),true)', 'batch.upload_enqueue')
s = wrap(s, 'initial_(transformInput(block.narrow(1,1,4),0,4))', 'initialize.network')
s = wrap(s, 'states_.index_select(0,ids)', 'segment.gather_state')
s = wrap(s, 'transformInput(block.narrow(1,1,5),0,4)', 'segment.input_transform')
s = wrap(s, 'at::silu(encoder2_(at::silu(encoder0_(networkFeatures))))', 'segment.encoder')
s = wrap(s, 'at::cat({old,embedded},1)', 'segment.context')
s = wrap(s, 'hazard_(context)', 'segment.hazard_mlp')
s = wrap(s, 'at::softplus(logits,1,20)', 'segment.output_transform')
s = wrap(s, 'at::gru_cell(embedded,old,gruInput_.weight,gruHidden_.weight,\n                                          gruInput_.bias,gruHidden_.bias)', 'segment.gru')
s = wrap(s, 'enteringContext_.index_copy_(0,ids,context)', 'segment.store_context')
s = wrap(s, 'states_.index_copy_(0,ids,next)', 'segment.store_state')
s = wrap(s, 'mixture_(at::cat({context,transformInput(block.narrow(1,1,3),1,2)},1))', 'mixture.network')
s = wrap(s, 'frame.hostOutput.narrow(0,0,frame.outputCount).copy_(frame.deviceOutput,true)', 'batch.download_enqueue')
s = wrap(s, 'frame.completed.synchronize()', 'batch.event_wait')
write('RenewalTorchProfile.cpp', s)

s = load('src/learned/RenewalBatchSession.cpp')
s = replace(s, '    const RenewalBatchRequests& r, const WavefrontFeatures* features) {',
            '    const RenewalBatchRequests& r, const WavefrontFeatures* features) {\n'
            '    renewal_profile::Scope scope("batch.session_enqueue");')
s = replace(s, 'void RenewalBatchSession::collectInto(Ticket ticket, RenewalBatchResults& result) {',
            'void RenewalBatchSession::collectInto(Ticket ticket, RenewalBatchResults& result) {\n'
            '    renewal_profile::Scope scope("batch.session_collect");')
s = wrap(s, 'impl_->engine->prepareInput(batch,5*ni+6*ns+4*nm)', 'batch.prepare_input')
s = wrap(s, 'impl_->engine->submit(batch,ni,ns,nm)', 'batch.backend_enqueue')
s = wrap(s, 'impl_->engine->collect(batch)', 'batch.backend_collect')
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
