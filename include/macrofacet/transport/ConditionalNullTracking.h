#pragma once

#include "macrofacet/transport/ConditionalFlightKernel.h"
#include "macrofacet/transport/ClassicNullTracking.h"

namespace mf {

// Two bounds: analytic birth envelope and a conditional-moment bound. Expensive
// far segments use certified interval thinning to skip redundant null events.
// The original birth observation and absolute age survive the segment switch.
FlightSample sampleConditionalDeltaTracking(const ConditionalFlightKernel& kernel,
    Random& rng, DdaTrackingDiagnostics* diagnostics = nullptr,
    double maximumAge = -1.0);

} // namespace mf
