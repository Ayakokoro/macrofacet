#pragma once

#include "macrofacet/core/Random.h"
#include "macrofacet/mathutility/NumericPolicy.h"

namespace mf {

PositiveResult positiveRawMoment(int order, double mean, double stddev,
                                 const NumericPolicy& policy = defaultNumericPolicy());
PositiveResult negativePartMean(double mean, double stddev,
                                const NumericPolicy& policy = defaultNumericPolicy());
double negativeFluxNormalLogPdf(double k, double mean, double stddev,
                                const NumericPolicy& policy = defaultNumericPolicy());
PositiveResult negativeFluxNormalCdf(double k, double mean, double stddev,
                                     const NumericPolicy& policy = defaultNumericPolicy());
double sampleNegativeFluxNormal(double mean, double stddev, Random& rng,
                                const NumericPolicy& policy = defaultNumericPolicy());

} // namespace mf

