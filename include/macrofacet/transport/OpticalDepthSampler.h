#pragma once

#include "macrofacet/core/Random.h"
#include "macrofacet/transport/FlightKernel.h"

namespace mf {

struct FlightSample {
    bool collided = false;
    double age = 0.0;
    std::optional<double> logSurvival;
    std::optional<double> logDistancePdf;
    std::optional<double> escapeMass;
};

PositiveResult integrateHazard(const FlightKernel& kernel, double a, double b,
                               const NumericPolicy& policy = defaultNumericPolicy());
FlightSample sampleFlight(const FlightKernel& kernel, Random& rng,
                          const NumericPolicy& policy = defaultNumericPolicy());

} // namespace mf

