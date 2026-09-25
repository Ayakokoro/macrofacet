#pragma once

#include "macrofacet/core/Random.h"
#include "macrofacet/transport/FlightKernel.h"
#include <cstdint>
#include <vector>

namespace mf {

struct FlightSample {
    bool collided = false;
    double age = 0.0;
    std::optional<double> logSurvival;
    std::optional<double> logDistancePdf;
    std::optional<double> escapeMass;
};

struct TrackingDiagnostics {
    std::uint64_t hazardEvaluations = 0;
    std::uint64_t quadratureIntervals = 0;
    std::uint64_t newtonIterations = 0;
    std::uint64_t bisectionSteps = 0;
    double maximumResidual = 0.0;
    double maximumIntegrationError = 0.0;
};

// Per-flight cache; no RNG ownership or changes to the statistical birth state.
class OpticalDepthSampler {
public:
    OpticalDepthSampler(const FlightKernel& kernel, const NumericPolicy& policy,
                        double maximumAge = -1.0, int initialCells = 16,
                        TrackingDiagnostics* diagnostics = nullptr);
    FlightSample sample(Random& rng);
    FlightSample invert(double exponentialTarget);
private:
    struct Segment { double lo, hi, prefix, prefixError, depth, error; };
    IntegralResult integrate(double a, double b) const;
    const FlightKernel& kernel_;
    NumericPolicy policy_;
    double begin_, end_;
    std::vector<double> knots_;
    std::vector<Segment> segments_;
    TrackingDiagnostics* diagnostics_;
};

PositiveResult integrateHazard(const FlightKernel& kernel, double a, double b,
                               const NumericPolicy& policy = defaultNumericPolicy());
FlightSample sampleFlight(const FlightKernel& kernel, Random& rng,
                          const NumericPolicy& policy = defaultNumericPolicy(),
                          TrackingDiagnostics* diagnostics = nullptr, int initialCells = 16,
                          double maximumAge = -1.0);

} // namespace mf
