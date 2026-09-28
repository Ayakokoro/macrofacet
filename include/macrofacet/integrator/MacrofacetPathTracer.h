#pragma once

#include "macrofacet/core/Random.h"
#include "macrofacet/experiments/ExperimentConfig.h"
#include "macrofacet/transport/ClassicNullTracking.h"
#include <atomic>
#include <vector>

namespace mf {

class NarrowBandMedium;

struct RenderStatistics {
    std::uint64_t paths = 0;
    std::uint64_t realCollisions = 0;
    std::uint64_t escapedPaths = 0;
    std::uint64_t rouletteTerminations = 0;
    std::uint64_t safetyCapTerminations = 0;
    std::uint64_t numericalFailures = 0;
    double accumulatedPathDepth = 0.0;
    DdaTrackingDiagnostics tracking;
};

inline void mergeInto(RenderStatistics& destination, const RenderStatistics& source) {
    destination.paths += source.paths;
    destination.realCollisions += source.realCollisions;
    destination.escapedPaths += source.escapedPaths;
    destination.rouletteTerminations += source.rouletteTerminations;
    destination.safetyCapTerminations += source.safetyCapTerminations;
    destination.numericalFailures += source.numericalFailures;
    destination.accumulatedPathDepth += source.accumulatedPathDepth;
    destination.tracking.candidates += source.tracking.candidates;
    destination.tracking.nullCollisions += source.tracking.nullCollisions;
    destination.tracking.boundIntervals += source.tracking.boundIntervals;
    destination.tracking.nearCandidates += source.tracking.nearCandidates;
    destination.tracking.farCandidates += source.tracking.farCandidates;
    destination.tracking.roundedCandidateSteps += source.tracking.roundedCandidateSteps;
    destination.tracking.adaptiveMajorantFlights += source.tracking.adaptiveMajorantFlights;
}

struct RenderedImage {
    int width = 0;
    int height = 0;
    std::vector<Spectrum> pixels;
    RenderStatistics statistics;
};

Spectrum traceCameraPath(const Ray& initialRay,
                         const ExperimentConfig& config, Random& rng,
                         RenderStatistics& statistics);
Spectrum traceCameraPath(const Ray& initialRay,
                         const ExperimentConfig& config, const NarrowBandMedium& medium,
                         Random& rng, RenderStatistics& statistics);
// When supplied, increment completedCameraRays after each finished pixel sample.
// Multiple renders may share the counter to report their combined progress.
RenderedImage renderAnalyticScene(const ExperimentConfig& config,
    std::atomic<std::uint64_t>* completedCameraRays = nullptr);
void writePfm(const std::filesystem::path& path, const RenderedImage& image);
void writeBmpPreview(const std::filesystem::path& path, const RenderedImage& image);

} // namespace mf
