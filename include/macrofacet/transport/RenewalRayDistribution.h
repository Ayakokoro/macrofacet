#pragma once

#include "macrofacet/learned/RenewalHazardModel.h"
#include <cstddef>
#include <vector>

namespace mf {

// Unit-distance hazard coefficients on the ORIGINAL deterministic mean segment.
// Never rebuild this segment at a visibility cutoff: doing so changes the GRU input.
class RenewalHazardSegment {
public:
    RenewalHazardSegment(double begin, double end, std::array<double, 4> rates);
    double begin() const { return begin_; }
    double end() const { return end_; }
    const std::array<double, 4>& rates() const { return rates_; }
    double hazard(double u) const;
    double cumulative(double u) const;
    double integral(double uBegin, double uEnd) const;
    double inverse(double opticalDepth, double uBegin, double uEnd, double normalizedTolerance) const;
private:
    double begin_, end_;
    std::array<double, 4> rates_;
};

struct RenewalDistanceSample {
    bool hit = false;
    double distance = 0.0; // physical absolute ray distance; cutoff for a miss
    std::size_t segment = 0; // segment count for a miss
};

// Cache per-ray coefficients for repeated sampling/visibility queries. Sharing the
// immutable model across rays/threads is safe; history belongs to each ray.
class RenewalRayDistribution {
public:
    RenewalRayDistribution(std::vector<RenewalHazardSegment> segments, double ell);
    static RenewalRayDistribution fromModel(const RenewalHazardModel& model,
        const RayMeanProfile& mean, const RayStartCondition& start, double ell);
    double maximumDistance() const { return segments_.back().end()*ell_; }
    double lengthScale() const { return ell_; }
    const std::vector<RenewalHazardSegment>& segments() const { return segments_; }
    double cumulativeHazard(double physicalDistance) const;
    double hazard(double physicalDistance) const; // h_s = h_x / ell
    double transmittance(double physicalDistance) const;
    // Conditional on survival from the ORIGINAL birth to from, not a fresh start.
    double transmittance(double from, double to) const;
    RenewalDistanceSample sample(Random& rng) const;
    RenewalDistanceSample sample(Random& rng, double from, double to,
        double physicalTolerance = 1e-9) const;
    RenewalDistanceSample sampleOpticalDepth(double opticalDepth, double from, double to,
        double physicalTolerance = 1e-9) const;
private:
    double normalized(double physicalDistance) const;
    std::size_t containing(double x) const;
    std::vector<RenewalHazardSegment> segments_;
    std::vector<double> prefix_;
    double ell_;
};

// Single-query variants evaluate the causal prefix and stop at the hit/cutoff.
RenewalDistanceSample sampleRenewalDistance(const RenewalHazardModel& model,
    const RayMeanProfile& mean, const RayStartCondition& start, double ell,
    Random& rng, double from, double to, double physicalTolerance = 1e-9);
double renewalTransmittance(const RenewalHazardModel& model, const RayMeanProfile& mean,
    const RayStartCondition& start, double ell, double from, double to);

} // namespace mf
