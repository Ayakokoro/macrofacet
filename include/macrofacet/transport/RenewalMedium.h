#pragma once

#include "macrofacet/gpss/GPSSField.h"
#include "macrofacet/transport/RenewalRayDistribution.h"

namespace mf {

struct RenewalTrackingDiagnostics {
    std::uint64_t flights = 0, segments = 0, mixtureQueries = 0;
    std::uint64_t initializationBatches = 0, segmentBatches = 0, mixtureBatches = 0, maximumBatchSize = 0;
    std::string backend = "scalar";
    int cpuWorkers = 1;
    std::uint64_t parallelAdvanceBatches = 0, serialAdvanceBatches = 0;
    std::uint64_t submissions = 0, readbacks = 0, combinedSubmissions = 0;
    std::uint64_t maximumInitializationBatch = 0, maximumMixtureBatch = 0;
    std::uint64_t maximumInitializationWait = 0, maximumMixtureWait = 0;
    int rayPoolSize = 0;
    std::uint64_t maximumInFlightBatches = 0, blockingCollects = 0, cpuBatchesWithGpuPending = 0;
};

struct RenewalFlight {
    Ray ray;
    std::optional<Vector3> birthGradient;
    RayMeanProfile mean;
    RayStartCondition start;
    double ell, sigma;
    Vector3 birthMeanGradient;
    double maximumDistance() const { return mean.maximumX()*ell; }
};

struct RenewalSurfaceSample {
    bool hit = false;
    double distance = 0, speed = 0; // physical distance, standardized W > 0
    std::size_t segment = 0;
    Point3 position = Point3::Zero();
    Vector3 gradient = Vector3::Zero(), normal = Vector3::Zero();
};

// Reconstruct the full gradient given physical directional derivative V < 0.
// Supports isotropic and fixed positive-definite elliptical Mat32 and SE kernels.
Vector3 sampleRenewalGradient(const CovarianceKernel& kernel, const Vector3& direction,
    double normalizedDistance, double derivative, const Vector3& meanGradient,
    const Vector3& birthMeanGradient, const std::optional<Vector3>& birthGradient, Random& rng);

class RenewalMedium {
public:
    RenewalMedium(const GPSSField& field, const RenewalHazardModel& model, double maximumStep = 0.25,
                  const std::string& profileMode = "cubic");
    // Origin must already be in the active domain. A null flight exits immediately.
    // cubic builds to the domain exit; point_linear builds only visited segments.
    std::optional<RenewalFlight> beginFlight(const Ray& ray,
        std::optional<Vector3> birthGradient = std::nullopt) const;
    RenewalSurfaceSample sample(const RenewalFlight& flight, Random& rng,
        double physicalTolerance = 1e-9, RenewalTrackingDiagnostics* diagnostics = nullptr) const;
    // Shared collision reconstruction for scalar and batched network queries.
    RenewalSurfaceSample sampleHit(const RenewalFlight& flight, std::size_t segment,
        double u, const RenewalSpeedMixture& mixture, Random& rng) const;
    double transmittance(const RenewalFlight& flight, double maximumDistance) const;
    // Shadow direction gets a fresh B initialization with the SAME full gradient.
    double surfaceTransmittance(const Ray& ray, const Vector3& birthGradient,
        double maximumDistance) const;
private:
    const GPSSField& field_;
    const RenewalHazardModel& model_;
    double maximumStep_;
    bool pointLinear_ = false;
};

} // namespace mf
