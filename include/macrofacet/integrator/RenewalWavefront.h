#pragma once
#include "macrofacet/integrator/MacrofacetPathTracer.h"

namespace mf {
RenderedImage renderRenewalWavefront(const ExperimentConfig& prepared,
    std::atomic<std::uint64_t>* completedCameraRays);
}
