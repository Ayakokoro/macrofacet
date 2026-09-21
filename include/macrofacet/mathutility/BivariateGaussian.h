#pragma once

#include "macrofacet/core/Types.h"
#include "macrofacet/mathutility/NumericPolicy.h"

namespace mf {

PositiveResult standardBivariateNormalCdf(double a, double b, double correlation,
                                          const NumericPolicy& policy = defaultNumericPolicy());
PositiveResult positiveOrthant2(const Vector2& mean, const Matrix2& covariance,
                                const NumericPolicy& policy = defaultNumericPolicy());

} // namespace mf

