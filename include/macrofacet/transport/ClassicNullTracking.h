#pragma once

#include "macrofacet/transport/ClassicFlightKernel.h"
#include "macrofacet/transport/OpticalDepthSampler.h"
#include "macrofacet/transport/DensityMajorantGrid.h"

namespace mf {

FlightSample sampleClassicNullTracking(const ClassicFlightKernel& kernel,
                                       double certifiedMajorant, Random& rng,
                                       double maximumAge = -1.0);
FlightSample sampleClassicDdaTracking(const ClassicFlightKernel& kernel,
                                     const DensityMajorantGrid& densityMajorant,
                                     double areaMajorant, Random& rng,
                                     double maximumAge = -1.0);

} // namespace mf
