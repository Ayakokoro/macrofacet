#include "macrofacet/integrator/CausalRenewalCuda.h"
#include "macrofacet/fields/NanoVdbMean.h"
#include "learned/CausalFirstHitBackend.h"
#include "cuda/CausalRenewalDevice.h"
#include <nanovdb/io/IO.h>
#include <nanovdb/HostBuffer.h>
#include <nlohmann/json.hpp>
#include <fstream>
#include <memory>

namespace mf {
namespace {
using namespace cuda_renewal;
V3 vec(const Vector3& v) { return {v.x(),v.y(),v.z()}; }
Vector3 vec(V3 v) { return {v.x,v.y,v.z}; }
std::unique_ptr<Context> context(const ExperimentConfig& config) {
    if (!config.renewal.causalModel) throw std::invalid_argument("prepare the joint model before CUDA rendering");
    const auto* mean=dynamic_cast<const NanoVdbMean*>(config.field.mean.get());
    if (!mean) throw std::invalid_argument("cuda_wavefront requires a NanoVDB mean");
    std::ifstream metadata(mean->gridFile().string()+".json");
    nlohmann::json sidecar; if (metadata) metadata>>sidecar;
    if (sidecar.value("coverage","")!="full_domain")
        throw std::invalid_argument("cuda_wavefront requires NanoVDB coverage=full_domain");
    const Matrix3 metric=config.field.kernel.metric();
    const double inverseEllSquared=metric.trace()/3;
    if (!(inverseEllSquared>0) || (metric-Matrix3::Identity()*inverseEllSquared).cwiseAbs().maxCoeff()>1e-12*inverseEllSquared)
        throw std::invalid_argument("cuda_wavefront currently supports isotropic SE kernels only");
    const auto& model=*config.renewal.causalModel;
    const auto& w=model.networkWeights();
    if (w.hidden>32 || w.embedding>32 || w.head0.output>64 || w.mixture0.output>64 || w.components>16)
        throw std::invalid_argument("cuda_wavefront supports hidden/embedding <=32, head widths <=64, mixture components <=16");
    std::vector<float> weights;
    const auto dense=[&](const RenewalDenseWeights& layer) {
        Dense result{layer.input,layer.output,static_cast<int>(weights.size()),0};
        for (int i=0;i<layer.input;++i) for (int j=0;j<layer.output;++j)
            weights.push_back(layer.weight[j*layer.input+i]);
        result.bias=static_cast<int>(weights.size());
        weights.insert(weights.end(),layer.bias.begin(),layer.bias.end()); return result;
    };
    Network n{};
    n.initial0=dense(w.initial0); n.initial2=dense(w.initial2);
    n.segment0=dense(w.segment0); n.segment2=dense(w.segment2);
    n.head0=dense(w.head0); n.head2=dense(w.head2);
    n.gruInput=dense(w.gruInput); n.gruHidden=dense(w.gruHidden);
    n.mixture0=dense(w.mixture0); n.mixture2=dense(w.mixture2); n.mixture4=dense(w.mixture4);
    n.hidden=w.hidden; n.embedding=w.embedding; n.components=w.components;
    n.step=model.step(); n.fine=model.integrationStep();
    const auto& r=config.render;
    const Vector3 forward=normalizedOrThrow(r.cameraTarget-r.cameraPosition);
    Vector3 right,up; orthonormalComplement(forward,right,up);
    Scene s{}; s.minimum=vec(config.field.activeDomain.minimum); s.maximum=vec(config.field.activeDomain.maximum);
    s.camera=vec(r.cameraPosition); s.forward=vec(forward); s.right=vec(right); s.up=vec(up);
    s.eta=vec(config.material.conductor.eta); s.k=vec(config.material.conductor.k);
    s.sigma=config.field.kernel.sigma(); s.ell=1/std::sqrt(inverseEllSquared);
    s.profileStep=config.renewal.profileMaximumStep; s.tolerance=config.numeric.distanceAbsoluteTolerance;
    s.aspect=double(r.width)/r.height; s.cameraScale=std::tan(r.verticalFovDegrees*kPi/360);
    s.width=r.width; s.height=r.height; s.spp=r.samplesPerPixel;
    s.rouletteDepth=r.rouletteStartDepth; s.safetyDepth=r.safetyDepthCap;
    s.white=r.environment=="unit_white"; s.unitFresnel=config.material.conductor.forceUnitFresnel || s.white;
    s.seed=config.seed;
    s.deltaTracking=config.renewal.sampler=="segmented_delta_tracking";
    if (r.environment!="unit_white" && r.environment!="directional_gradient")
        throw std::invalid_argument("cuda_wavefront supports unit_white and directional_gradient environments");
    auto grid=nanovdb::io::readGrid<nanovdb::HostBuffer>(mean->gridFile().string(),"sdf");
    if (!grid.grid<float>()) throw std::invalid_argument("CUDA requires a float SDF grid");
    return std::make_unique<Context>(grid.data(),grid.bufferSize(),n,weights,s);
}
RenderedImage render(const ExperimentConfig& config,std::atomic<std::uint64_t>* progress) {
    auto gpu=context(config);
    const int capacity=config.renewal.rayPoolSize ? config.renewal.rayPoolSize : 8192;
    Statistics counters{}; std::uint64_t last=0;
    const auto pixels=gpu->render(capacity,counters,[&](std::uint64_t count) {
        if (progress) progress->fetch_add(count-last,std::memory_order_relaxed); last=count;
    });
    RenderedImage image; image.width=config.render.width; image.height=config.render.height;
    image.pixels.reserve(pixels.size()); for (auto p:pixels) image.pixels.push_back(vec(p));
    auto& s=image.statistics;
    s.paths=counters.completed; s.realCollisions=counters.collisions; s.escapedPaths=counters.escaped;
    s.rouletteTerminations=counters.roulette; s.safetyCapTerminations=counters.capped;
    s.numericalFailures=counters.errors; s.accumulatedPathDepth=double(counters.depth);
    s.renewal.backend="cuda_wavefront"; s.renewal.flights=counters.flights;
    s.renewal.segments=counters.intervals; s.renewal.mixtureQueries=counters.collisions;
    s.renewal.submissions=counters.kernels; s.renewal.readbacks=counters.kernels/8+1;
    s.renewal.rayPoolSize=capacity; s.renewal.maximumBatchSize=capacity; s.renewal.cpuWorkers=0;
    s.tracking.candidates=counters.candidates; s.tracking.nullCollisions=counters.nulls;
    s.tracking.boundIntervals=counters.bounds; s.tracking.trackingSegments=counters.bounds;
    return image;
}
} // namespace
void registerCausalRenewalCudaBackend() { setCausalRenewalCudaRenderer(render); }
std::vector<CausalCudaFlightOutput> sampleCausalRenewalCuda(const ExperimentConfig& config,
    const std::vector<CausalCudaFlightInput>& input) {
    auto gpu=context(config); std::vector<cuda_renewal::Input> in; in.reserve(input.size());
    for (const auto& x:input) {
        if (!config.field.activeDomain.contains(x.ray.origin) || !x.ray.origin.allFinite())
            throw std::invalid_argument("CUDA query origin must be inside the active domain");
        cuda_renewal::Input r{};
        r.origin=vec(x.ray.origin); r.direction=vec(normalizedOrThrow(x.ray.direction));
        r.surface=x.birthGradient.has_value(); r.birthGradient=vec(x.birthGradient.value_or(Vector3::Zero()));
        for (int i=0;i<3;++i) {
            if (!(x.uniforms[i]>0 && x.uniforms[i]<1)) throw std::invalid_argument("CUDA uniforms must be open");
            r.uniforms[i]=x.uniforms[i];
        }
        for (int i=0;i<2;++i) {
            if (!std::isfinite(x.normals[i])) throw std::invalid_argument("CUDA normal variates must be finite");
            r.normals[i]=x.normals[i];
        }
        r.queryOnly=x.queryOnly; r.trackingSeed=x.trackingSeed; in.push_back(r);
    }
    const auto raw=gpu->sample(in); std::vector<CausalCudaFlightOutput> out; out.reserve(raw.size());
    for (std::size_t i=0;i<raw.size();++i) {
        const auto& x=raw[i];
        if (x.error) throw std::runtime_error("CUDA flight "+std::to_string(i)+" failed, code "+std::to_string(x.error));
        out.push_back({x.hit!=0,x.q,x.speed,x.hazard,vec(x.gradient),x.intervals,x.candidates,x.nulls,x.bounds});
    }
    return out;
}
} // namespace mf
