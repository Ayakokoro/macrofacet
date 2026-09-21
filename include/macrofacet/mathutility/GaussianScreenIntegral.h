#pragma once

#include "macrofacet/mathutility/NumericPolicy.h"

namespace mf {

PositiveResult negativeFluxTimesCdf(double meanK, double stddevK, double a, double b,
                                    const NumericPolicy& policy = defaultNumericPolicy());
PositiveResult negativeFluxTimesAffineIndicator(
    double meanK, double stddevK, double intercept, double slope,
    const NumericPolicy& policy = defaultNumericPolicy());

} // namespace mf

