#pragma once

#include "macrofacet/core/Types.h"
#include <cmath>

namespace mf {

// Columns are the tangent axes and the mean-surface normal. +Z maps to the
// world frame, preserving the meaning of anisotropic GGX on a flat XY plane.
inline Matrix3 tangentFrame(const Vector3& meanGradient) {
    Vector3 n = Vector3::UnitZ();
    if (meanGradient.squaredNorm() > 0.0) n = normalizedOrThrow(meanGradient);
    const double sign = std::copysign(1.0, n.z());
    const double a = -1.0 / (sign + n.z());
    const double b = n.x() * n.y() * a;
    Matrix3 frame;
    frame.col(0) = Vector3(1.0 + sign * n.x() * n.x() * a,
                           sign * b, -sign * n.x());
    frame.col(1) = Vector3(b, sign + n.y() * n.y() * a, -n.y());
    frame.col(2) = n;
    return frame;
}

} // namespace mf
