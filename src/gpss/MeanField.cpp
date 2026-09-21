#include "macrofacet/gpss/MeanField.h"
#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>

namespace mf {

PlaneMean::PlaneMean(Vector3 normal, double offset)
    : normal_(normalizedOrThrow(normal)), offset_(offset) {
    if (!std::isfinite(offset)) throw std::invalid_argument("plane offset must be finite");
}

MeanJet PlaneMean::evaluate(const Point3& x) const {
    return {normal_.dot(x) - offset_, normal_};
}

BoundsSummary PlaneMean::bounds(const Bounds3& domain) const {
    Point3 minimizing;
    for (int i = 0; i < 3; ++i) {
        minimizing[i] = normal_[i] >= 0.0 ? domain.minimum[i] : domain.maximum[i];
    }
    return {normal_.dot(minimizing) - offset_, 1.0, true};
}

SphereMean::SphereMean(Point3 center, double radius) : center_(std::move(center)), radius_(radius) {
    if (!(radius > 0.0) || !center_.allFinite()) {
        throw std::invalid_argument("invalid sphere mean");
    }
}

MeanJet SphereMean::evaluate(const Point3& x) const {
    const Vector3 displacement = x - center_;
    const double length = displacement.norm();
    if (length == 0.0) {
        throw std::domain_error("sphere mean is nondifferentiable at its center");
    }
    return {length - radius_, displacement / length};
}

BoundsSummary SphereMean::bounds(const Bounds3& domain) const {
    Point3 closest = center_.cwiseMax(domain.minimum).cwiseMin(domain.maximum);
    return {(closest - center_).norm() - radius_, 1.0, true};
}

} // namespace mf

