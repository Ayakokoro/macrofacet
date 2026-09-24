#include "macrofacet/gpss/ShaderBallMean.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace mf {

ShaderBallMean::ShaderBallMean(Point3 center, double radius, Vector3 grooveAxis,
                             double grooveRadius)
    : center_(std::move(center)), radius_(radius), grooveAxis_(std::move(grooveAxis)),
      grooveRadius_(grooveRadius) {
    if (!center_.allFinite() || !std::isfinite(radius_) || !(radius_ > 0.0) ||
        !std::isfinite(grooveRadius_) || !(grooveRadius_ > 0.0) ||
        grooveRadius_ > 0.15 * radius_ || !grooveAxis_.allFinite() ||
        !(grooveAxis_.cwiseAbs().maxCoeff() > 0.0)) {
        throw std::invalid_argument(
            "shader ball requires finite center, positive radius, 0 < groove radius <= 0.15 * radius, "
            "and finite nonzero groove axis");
    }
    // Rescale first so finite axes with very large or very small components
    // can be normalized without overflow or underflow in their squared norm.
    grooveAxis_ /= grooveAxis_.cwiseAbs().maxCoeff();
    grooveAxis_.normalize();
    Vector3 unused;
    orthonormalComplement(grooveAxis_, radialFallback_, unused);

    const double theta = std::asin(0.3);
    const double delta = 2.0 * std::asin(0.5 * (grooveRadius_ / radius_));
    const double alpha = std::acos(-0.5 * (grooveRadius_ / radius_));
    grooveMajorRadius_ = std::sqrt(1.0 - 0.3 * 0.3) * radius_;

    const auto arc = [](const Vector2& center2, double arcRadius, double begin,
                        double end, double normalSign) {
        Arc result;
        result.center = center2;
        result.radius = arcRadius;
        result.normalSign = normalSign;
        result.firstDirection = Vector2(std::cos(begin), std::sin(begin));
        result.lastDirection = Vector2(std::cos(end), std::sin(end));
        result.first = center2 + arcRadius * result.firstDirection;
        result.last = center2 + arcRadius * result.lastDirection;
        return result;
    };
    // Only exposed meridian arcs are boundaries. In particular, the rotation
    // axis is not a boundary. The small groove radius keeps both grooves apart
    // and away from that axis.
    arcs_[0] = arc(Vector2::Zero(), radius_, -0.5 * kPi, -theta - delta, 1.0);
    arcs_[1] = arc(Vector2::Zero(), radius_, -theta + delta, theta - delta, 1.0);
    arcs_[2] = arc(Vector2::Zero(), radius_, theta + delta, 0.5 * kPi, 1.0);
    arcs_[3] = arc(Vector2(grooveMajorRadius_, -0.3 * radius_), grooveRadius_,
                   -theta + alpha, -theta + 2.0 * kPi - alpha, -1.0);
    arcs_[4] = arc(Vector2(grooveMajorRadius_, 0.3 * radius_), grooveRadius_,
                   theta + alpha, theta + 2.0 * kPi - alpha, -1.0);
}

MeanJet ShaderBallMean::ballJet(const Vector3& p) const {
    const double axial = p.dot(grooveAxis_);
    const Vector3 radialVector = p - axial * grooveAxis_;
    const double radial = radialVector.norm();
    const double radialTolerance = 32.0 * std::numeric_limits<double>::epsilon() *
        std::max(radius_, std::abs(axial));
    // Near the rotation axis, cancellation in the projection can leave a
    // residual that is not perpendicular to the axis. Choose the known
    // orthogonal radial direction there, without changing the queried radius.
    const Vector3 radialDirection = radial > radialTolerance ?
        Vector3(radialVector / radial) : radialFallback_;
    const Vector2 meridian(radial, axial);

    double closestSquaredDistance = std::numeric_limits<double>::infinity();
    Vector2 closestOffset = Vector2::Zero();
    Vector2 boundaryNormal(0.0, 1.0);
    const auto consider = [&](const Arc& arc, const Vector2& point) {
        const Vector2 offset = meridian - point;
        const double squaredDistance = offset.squaredNorm();
        if (squaredDistance < closestSquaredDistance) {
            closestSquaredDistance = squaredDistance;
            closestOffset = offset;
            boundaryNormal = arc.normalSign * (point - arc.center) / arc.radius;
        }
    };
    for (const Arc& arc : arcs_) {
        const Vector2 relative = meridian - arc.center;
        // Each arc spans less than pi. The two cross products locate its
        // angular sector without trigonometric functions in the query path.
        const double fromFirst = arc.firstDirection.x() * relative.y() -
            arc.firstDirection.y() * relative.x();
        const double toLast = relative.x() * arc.lastDirection.y() -
            relative.y() * arc.lastDirection.x();
        const double length = relative.norm();
        if (length > 0.0 && fromFirst >= 0.0 && toLast >= 0.0) {
            consider(arc, arc.center + arc.radius * (relative / length));
        } else {
            consider(arc, arc.first);
            consider(arc, arc.last);
        }
    }

    const double closestDistance = std::sqrt(closestSquaredDistance);
    const double meridianLength = meridian.norm();
    const bool inside = meridianLength <= radius_ &&
        (meridian - arcs_[3].center).norm() >= grooveRadius_ &&
        (meridian - arcs_[4].center).norm() >= grooveRadius_;
    const double sign = inside ? -1.0 : 1.0;
    const double normalTolerance = 32.0 * std::numeric_limits<double>::epsilon() *
        std::max(radius_, meridianLength);
    const Vector2 normal = closestDistance <= normalTolerance ? boundaryNormal :
        Vector2(sign * closestOffset / closestDistance);
    return {sign * closestDistance, normal.x() * radialDirection + normal.y() * grooveAxis_};
}

MeanJet ShaderBallMean::pedestalJet(const Vector3& p) const {
    // Rounded cylinder: total radius 1.08R, height 0.20R, bevel radius 0.04R.
    // Its top touches the original sphere at z = -R, so the interiors remain
    // disjoint even when the groove axis tilts.
    const double radial = std::hypot(p.x(), p.y());
    const double height = p.z() + 1.1 * radius_;
    const Vector2 q(radial - 1.04 * radius_, std::abs(height) - 0.06 * radius_);
    const Vector2 outside = q.cwiseMax(0.0);
    const double outsideLength = outside.norm();
    const double value = outsideLength + std::min(std::max(q.x(), q.y()), 0.0) - 0.04 * radius_;

    Vector2 normal;
    if (outsideLength > 0.0) normal = outside / outsideLength;
    else normal = q.x() >= q.y() ? Vector2(1.0, 0.0) : Vector2(0.0, 1.0);
    const Vector3 radialDirection = radial > 0.0 ?
        Vector3(p.x() / radial, p.y() / radial, 0.0) : Vector3::UnitX();
    return {value, normal.x() * radialDirection +
        normal.y() * (height >= 0.0 ? 1.0 : -1.0) * Vector3::UnitZ()};
}

MeanJet ShaderBallMean::evaluate(const Point3& x) const {
    const Vector3 p = x - center_;
    const MeanJet ball = ballJet(p);
    const MeanJet pedestal = pedestalJet(p);
    // The ball and pedestal intersect in at most one tangent point, so every
    // component boundary remains exposed. Their union therefore has the exact
    // SDF given by this minimum (disjoint interiors alone would not suffice).
    return ball.value <= pedestal.value ? ball : pedestal;
}

BoundsSummary ShaderBallMean::bounds(const Bounds3& domain) const {
    const Point3 midpoint = 0.5 * domain.minimum + 0.5 * domain.maximum;
    const double halfDiagonal = (0.5 * (domain.maximum - domain.minimum)).norm();
    // Exact signed distance is globally 1-Lipschitz, including across seams.
    return {evaluate(midpoint).value - halfDiagonal, 1.0, true};
}

} // namespace mf
