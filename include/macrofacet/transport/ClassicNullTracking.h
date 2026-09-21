#pragma once

#include "macrofacet/transport/ClassicFlightKernel.h"
#include "macrofacet/transport/OpticalDepthSampler.h"

namespace mf {

FlightSample sampleClassicNullTracking(const ClassicFlightKernel& kernel,
                                       double certifiedMajorant, Random& rng);

} // namespace mf

