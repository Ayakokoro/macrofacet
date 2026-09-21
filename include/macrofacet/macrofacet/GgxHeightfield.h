#pragma once

#include "macrofacet/core/Types.h"
#include "macrofacet/mathutility/NumericPolicy.h"

namespace mf {

class GgxHeightfield {
public:
    GgxHeightfield(double alphaX, double alphaY);
    PositiveResult evaluateD(const Vector3& unitNormal) const;
    PositiveResult projectedArea(const Vector3& travelDirection) const;
    double visibleNormalPdf(const Vector3& unitNormal, const Vector3& travelDirection) const;
private:
    double alphaX_;
    double alphaY_;
};

} // namespace mf

