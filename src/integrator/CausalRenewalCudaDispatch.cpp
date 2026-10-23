#include "macrofacet/integrator/CausalRenewalCuda.h"
#include <stdexcept>

namespace mf {
namespace { CausalCudaRenderFunction renderer = nullptr; }
void setCausalRenewalCudaRenderer(CausalCudaRenderFunction function) { renderer = function; }
bool causalRenewalCudaRegistered() { return renderer != nullptr; }
RenderedImage renderCausalRenewalCuda(const ExperimentConfig& config, std::atomic<std::uint64_t>* progress) {
    if (!renderer) throw std::runtime_error("cuda_wavefront requires a build with MACROFACET_ENABLE_CAUSAL_CUDA=ON and registered NanoVDB fields");
    return renderer(config,progress);
}
} // namespace mf
