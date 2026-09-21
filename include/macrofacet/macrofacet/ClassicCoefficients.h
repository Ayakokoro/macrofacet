#pragma once

#include "macrofacet/gpss/GPSSField.h"
#include "macrofacet/mathutility/NumericPolicy.h"

namespace mf {

struct ClassicEvaluation {
    PositiveResult density;
    PositiveResult projectedArea;
    PositiveResult extinction;
};

ClassicEvaluation evaluateClassic(const GPSSField& field, const Point3& x, const Vector3& w,
                                  const NumericPolicy& policy = defaultNumericPolicy());
double classicMajorant(const GPSSField& field, const Bounds3& domain, const Vector3& w,
                       const NumericPolicy& policy = defaultNumericPolicy());

} // namespace mf

