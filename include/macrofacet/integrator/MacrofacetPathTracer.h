#pragma once

#include "macrofacet/core/Random.h"
#include "macrofacet/experiments/ExperimentConfig.h"
#include "macrofacet/transport/ClassicNullTracking.h"
#include "macrofacet/transport/RenewalMedium.h"
#include <algorithm>
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
    RenewalTrackingDiagnostics renewal;
};

inline void mergeInto(RenderStatistics& destination, const RenderStatistics& source) {
    destination.renewal.flights += source.renewal.flights;
    destination.renewal.segments += source.renewal.segments;
    destination.renewal.mixtureQueries += source.renewal.mixtureQueries;
    destination.renewal.initializationBatches += source.renewal.initializationBatches;
    destination.renewal.segmentBatches += source.renewal.segmentBatches;
    destination.renewal.mixtureBatches += source.renewal.mixtureBatches;
    destination.renewal.maximumBatchSize = std::max(destination.renewal.maximumBatchSize,source.renewal.maximumBatchSize);
    destination.renewal.backend = source.renewal.backend;
    destination.renewal.cpuWorkers = std::max(destination.renewal.cpuWorkers,source.renewal.cpuWorkers);
    destination.renewal.parallelAdvanceBatches += source.renewal.parallelAdvanceBatches;
    destination.renewal.serialAdvanceBatches += source.renewal.serialAdvanceBatches;
    destination.renewal.submissions += source.renewal.submissions;
    destination.renewal.readbacks += source.renewal.readbacks;
    destination.renewal.maximumInFlightBatches = std::max(destination.renewal.maximumInFlightBatches,source.renewal.maximumInFlightBatches);
    destination.renewal.blockingCollects += source.renewal.blockingCollects;
    destination.renewal.cpuBatchesWithGpuPending += source.renewal.cpuBatchesWithGpuPending;
    destination.renewal.combinedSubmissions += source.renewal.combinedSubmissions;
    destination.renewal.maximumInitializationBatch = std::max(destination.renewal.maximumInitializationBatch,source.renewal.maximumInitializationBatch);
    destination.renewal.maximumMixtureBatch = std::max(destination.renewal.maximumMixtureBatch,source.renewal.maximumMixtureBatch);
    destination.renewal.maximumInitializationWait = std::max(destination.renewal.maximumInitializationWait,source.renewal.maximumInitializationWait);
    destination.renewal.maximumMixtureWait = std::max(destination.renewal.maximumMixtureWait,source.renewal.maximumMixtureWait);
    destination.renewal.rayPoolSize = std::max(destination.renewal.rayPoolSize,source.renewal.rayPoolSize);
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
    destination.tracking.trackingSegments += source.tracking.trackingSegments;
    destination.tracking.maximumSegmentNullCollisions = std::max(
        destination.tracking.maximumSegmentNullCollisions,
        source.tracking.maximumSegmentNullCollisions);
    destination.tracking.maximumMajorant = std::max(destination.tracking.maximumMajorant,
                                                   source.tracking.maximumMajorant);
    destination.tracking.maximumCandidateMajorant = std::max(
        destination.tracking.maximumCandidateMajorant, source.tracking.maximumCandidateMajorant);
    destination.tracking.maximumEvaluatedHazard = std::max(
        destination.tracking.maximumEvaluatedHazard, source.tracking.maximumEvaluatedHazard);
    destination.tracking.maximumSegmentOpticalDepth = std::max(
        destination.tracking.maximumSegmentOpticalDepth, source.tracking.maximumSegmentOpticalDepth);
    destination.tracking.maximumConditionalIntervalMajorant = std::max(
        destination.tracking.maximumConditionalIntervalMajorant,
        source.tracking.maximumConditionalIntervalMajorant);
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
