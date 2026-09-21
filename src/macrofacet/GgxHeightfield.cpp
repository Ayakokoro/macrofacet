#include "macrofacet/macrofacet/GgxHeightfield.h"
#include <cmath>
#include <stdexcept>

namespace mf {

GgxHeightfield::GgxHeightfield(double alphaX, double alphaY)
    : alphaX_(alphaX), alphaY_(alphaY) {
    if (!(alphaX > 0.0) || !(alphaY > 0.0)) {
        throw std::invalid_argument("GGX roughness must be positive");
    }
}

PositiveResult GgxHeightfield::evaluateD(const Vector3& unitNormal) const {
    if (std::abs(unitNormal.norm() - 1.0) > 1e-9) {
        throw std::invalid_argument("GGX NDF direction must be unit length");
    }
    if (!(unitNormal.z() > 0.0)) return exactZero();
    const double denominator = std::pow(unitNormal.x() / alphaX_, 2) +
                               std::pow(unitNormal.y() / alphaY_, 2) +
                               unitNormal.z() * unitNormal.z();
    const double value = 1.0 / (kPi * alphaX_ * alphaY_ * denominator * denominator);
    return {value, std::log(value), 0.0, NumericStatus::Ok};
}

PositiveResult GgxHeightfield::projectedArea(const Vector3& travelDirection) const {
    const Vector3 w = normalizedOrThrow(travelDirection);
    const double q = alphaX_ * alphaX_ * w.x() * w.x() +
                     alphaY_ * alphaY_ * w.y() * w.y();
    const double root = std::sqrt(w.z() * w.z() + q);
    const double value = w.z() > 0.0 ? q / (2.0 * (root + w.z()))
                                     : 0.5 * (root - w.z());
    return value > 0.0 ? PositiveResult{value, std::log(value), 0.0, NumericStatus::Ok}
                       : exactZero();
}

double GgxHeightfield::visibleNormalPdf(const Vector3& unitNormal,
                                        const Vector3& travelDirection) const {
    const double cosine = -normalizedOrThrow(travelDirection).dot(unitNormal);
    if (!(cosine > 0.0)) return 0.0;
    const PositiveResult area = projectedArea(travelDirection);
    if (area.status == NumericStatus::ExactZero) return 0.0;
    return cosine * evaluateD(unitNormal).value / area.value;
}

} // namespace mf

