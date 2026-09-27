#pragma once

#include "macrofacet/transport/ClassicFlightKernel.h"
#include "macrofacet/transport/DensityMajorantGrid.h"
#include "macrofacet/core/Random.h"
#include <cstdint>

namespace mf {

struct FlightSample {
    bool collided = false;
    double age = 0.0;
};

struct DdaTrackingDiagnostics {
    std::uint64_t candidates = 0;
    std::uint64_t nullCollisions = 0;
};

FlightSample sampleClassicDdaTracking(const ClassicFlightKernel& kernel,
                                     const DensityMajorantGrid& densityMajorant,
                                     double areaMajorant, Random& rng,
                                     double maximumAge = -1.0,
                                     DdaTrackingDiagnostics* diagnostics = nullptr);

} // namespace mf
