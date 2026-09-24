#pragma once

#include "macrofacet/core/Random.h"
#include "macrofacet/experiments/ExperimentConfig.h"
#include "macrofacet/transport/OpticalDepthSampler.h"
#include <vector>

namespace mf {

struct RenderStatistics {
    std::uint64_t paths = 0;
    std::uint64_t realCollisions = 0;
    std::uint64_t escapedPaths = 0;
    std::uint64_t rouletteTerminations = 0;
    std::uint64_t safetyCapTerminations = 0;
    std::uint64_t numericalFailures = 0;
    double accumulatedPathDepth = 0.0;
    TrackingDiagnostics tracking;
};

inline void mergeInto(RenderStatistics& destination, const RenderStatistics& source) {
    destination.paths += source.paths;
    destination.realCollisions += source.realCollisions;
    destination.escapedPaths += source.escapedPaths;
    destination.rouletteTerminations += source.rouletteTerminations;
    destination.safetyCapTerminations += source.safetyCapTerminations;
    destination.numericalFailures += source.numericalFailures;
    destination.accumulatedPathDepth += source.accumulatedPathDepth;
    destination.tracking.hazardEvaluations += source.tracking.hazardEvaluations;
    destination.tracking.quadratureIntervals += source.tracking.quadratureIntervals;
    destination.tracking.newtonIterations += source.tracking.newtonIterations;
    destination.tracking.bisectionSteps += source.tracking.bisectionSteps;
    destination.tracking.maximumResidual = std::max(destination.tracking.maximumResidual, source.tracking.maximumResidual);
    destination.tracking.maximumIntegrationError = std::max(destination.tracking.maximumIntegrationError,
                                                           source.tracking.maximumIntegrationError);
}

struct RenderedImage {
    int width = 0;
    int height = 0;
    std::vector<Spectrum> pixels;
    RenderStatistics statistics;
};

Spectrum traceCameraPath(const Ray& initialRay, ModelMode mode,
                         const ExperimentConfig& config, Random& rng,
                         RenderStatistics& statistics);
RenderedImage renderAnalyticScene(ModelMode mode, const ExperimentConfig& config);
void writePfm(const std::filesystem::path& path, const RenderedImage& image);
void writeBmpPreview(const std::filesystem::path& path, const RenderedImage& image);

} // namespace mf
