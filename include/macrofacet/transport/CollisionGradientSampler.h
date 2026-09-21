#pragma once

#include "macrofacet/core/Random.h"
#include "macrofacet/mathutility/SmallGaussian.h"

namespace mf {

class FlightKernel;

Vector3 sampleFluxWeightedGradient(const Gaussian<3>& gradient, const Vector3& w, Random& rng,
                                   const NumericPolicy& policy = defaultNumericPolicy());
Vector3 sampleCollisionGradient(const FlightKernel& kernel, double age, Random& rng,
                                const NumericPolicy& policy = defaultNumericPolicy());

} // namespace mf

