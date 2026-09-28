#include "macrofacet/transport/ConditionalNullTracking.h"
#include "macrofacet/transport/ConditionalMedium.h"

namespace mf {

FlightSample sampleConditionalDeltaTracking(const ConditionalFlightKernel& kernel,
    Random& rng, DdaTrackingDiagnostics* diagnostics, double maximumAge) {
    return ConditionalMedium(kernel.field()).sample(kernel,rng,diagnostics,maximumAge);
}

} // namespace mf
