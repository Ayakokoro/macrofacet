#pragma once
#include "macrofacet/integrator/MacrofacetPathTracer.h"
#include <array>

namespace mf {
// Optional field-library backend. Registration keeps CUDA/NanoVDB out of the
// core library's dependencies, including in CPU-only builds.
using CausalCudaRenderFunction = RenderedImage (*)(const ExperimentConfig&, std::atomic<std::uint64_t>*);
void setCausalRenewalCudaRenderer(CausalCudaRenderFunction renderer);
bool causalRenewalCudaRegistered();
RenderedImage renderCausalRenewalCuda(const ExperimentConfig&, std::atomic<std::uint64_t>*);

struct CausalCudaFlightInput {
    Ray ray;
    std::optional<Vector3> birthGradient;
    std::array<double,3> uniforms;
    std::array<double,2> normals;
    bool queryOnly = false; // integrate the entire flight, without sampling
    std::uint64_t trackingSeed = 0; // independent Philox stream for variable-count delta candidates
};
struct CausalCudaFlightOutput {
    bool hit = false;
    double q = 0, speed = 0, cumulativeHazard = 0;
    Vector3 gradient = Vector3::Zero();
    std::uint64_t intervals = 0;
    std::uint64_t candidates = 0, nullCollisions = 0, boundIntervals = 0;
};
// Available when MACROFACET_ENABLE_CAUSAL_CUDA is ON; inputs deliberately
// include all inversion variates, or a seed for delta tracking's extra draws.
// Delta samples do not integrate hazard (cumulativeHazard is NaN); queryOnly
// always runs the reference quadrature and returns the full-flight hazard.
std::vector<CausalCudaFlightOutput> sampleCausalRenewalCuda(
    const ExperimentConfig&, const std::vector<CausalCudaFlightInput>&);
void registerCausalRenewalCudaBackend();
} // namespace mf
