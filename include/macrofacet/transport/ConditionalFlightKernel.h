#pragma once

#include "macrofacet/transport/FlightKernel.h"
#include "macrofacet/core/Random.h"

namespace mf {

FlightState startConditionalExterior(const GPSSField& field, const Point3& entry,
                                     const Vector3& direction, Random& rng);

struct ConditionalMajorantInterval {
    double begin, end, maximum;
};

struct ConditionalMajorants {
    double split = 0.0;
    double nearMaximum = 0.0;
    double farMaximum = 0.0;
    std::uint64_t boundIntervals = 0;
    std::vector<ConditionalMajorantInterval> farIntervals;
};

struct ConditionalBirthMajorant {
    double split = 0.0;
    double maximum = 0.0;
    double smoothEnd = 0.0;
    double slope = 0.0;
    double curvature = 0.0;
};

struct ConditionalScalarStatistics {
    double meanF, varianceF, meanK, meanAtZero, varianceAtZero;
};

// SE GP conditioned on the complete (value, gradient) observation at birth.
// Surface flights have birthValue=0; exterior flights start from F>0.
class ConditionalFlightKernel final : public FlightKernel {
public:
    ConditionalFlightKernel(const GPSSField& field, const FlightState& state);
    HazardEvaluation evaluate(double age) const override;
    HitStatistics hitStatistics(double age) const override;
    ConditionalScalarStatistics scalarStatistics(double age) const;
    ConditionalBirthMajorant birthMajorant(double end) const;
    double birthAdjacentMajorant(double begin, double end,
                                const ConditionalBirthMajorant& birth) const;
    ConditionalMajorants twoSegmentMajorants(double end) const;
    double intervalMajorant(double begin, double end) const;
private:
    MeanJet birthMean_;
    Vector3 deltaGradient_, precisionDirection_;
    double deltaValue_, deltaSlope_, inverseLength2_, sigma2_;
};

} // namespace mf
