#include "TestHarness.h"
#include "macrofacet/gpss/ShaderBallMean.h"
#include <array>
#include <limits>

void testShaderBallMean(TestContext& context) {
    using namespace mf;
    constexpr double grooveRadius = 0.07;
    const ShaderBallMean mean(Point3::Zero(), 1.0, Vector3::UnitZ(), grooveRadius);
    const Vector3 ringDirection(std::sqrt(0.91), 0.0, 0.3);

    struct DistanceCase {
        Point3 point;
        double expected;
        const char* name;
    };
    const std::array<DistanceCase, 12> cases{{
        {Point3(1.0, 0.0, 0.0), 0.0, "uncut equator"},
        {Point3(1.2, 0.0, 0.0), 0.2, "outside uncut equator"},
        {Point3(0.0, 0.0, 1.0), 0.0, "uncut north pole"},
        {ringDirection, grooveRadius, "removed upper groove center"},
        {Point3(ringDirection.x(), 0.0, -0.3), grooveRadius, "removed lower groove center"},
        {(1.0 - grooveRadius) * ringDirection, 0.0, "groove bottom"},
        {(1.0 - grooveRadius - 0.02) * ringDirection, -0.02, "solid behind groove bottom"},
        {Point3(0.6, 0.0, -1.0), 0.0, "pedestal top"},
        {Point3(0.0, 0.0, -1.2), 0.0, "pedestal bottom"},
        {Point3(1.08, 0.0, -1.1), 0.0, "pedestal side"},
        {Point3(0.0, 0.0, -1.1), -0.1, "inside pedestal"},
        {Point3(0.0, 0.0, -1.3), 0.1, "below pedestal"}
    }};
    for (const auto& sample : cases) {
        const MeanJet jet = mean.evaluate(sample.point);
        context.near(jet.value, sample.expected, 3e-12,
                     std::string("shader ball distance: ") + sample.name);
        context.require(jet.gradient.allFinite() && jet.gradient.norm() <= 1.0 + 1e-11,
                        std::string("shader ball finite bounded gradient: ") + sample.name);
    }

    const double bevelOffset = 0.04 / std::sqrt(2.0);
    const Point3 bevelPoint(1.04 + bevelOffset, 0.0, -1.04 + bevelOffset);
    const MeanJet bevel = mean.evaluate(bevelPoint);
    context.near(bevel.value, 0.0, 3e-12, "shader ball rounded pedestal corner is on surface");
    context.require((bevel.gradient - Vector3(1.0, 0.0, 1.0).normalized()).norm() < 1e-10,
                    "shader ball rounded pedestal corner has diagonal outward normal");
    context.require((mean.evaluate((1.0 - grooveRadius) * ringDirection).gradient -
                     ringDirection).norm() < 1e-10,
                    "shader ball groove bottom normal points into the removed material");
    const MeanJet join = mean.evaluate(Point3(0.0, 0.0, -1.0));
    context.near(join.value, 0.0, 3e-12, "shader ball touches the pedestal");
    context.require(join.gradient.allFinite() && join.gradient.norm() <= 1.0 + 1e-11,
                    "shader ball tangent join returns a finite bounded gradient");

    // Check derivatives in smooth regions of the ball, groove and pedestal.
    const std::array<Point3, 8> smoothPoints{{
        Point3(1.2, 0.1, 0.0), Point3(0.1, 0.1, 1.1),
        (1.0 - grooveRadius - 0.02) * ringDirection,
        (1.0 - grooveRadius + 0.02) * ringDirection,
        Point3(0.6, 0.1, -0.97), Point3(0.2, 0.1, -1.3),
        Point3(1.13, 0.03, -1.1), bevelPoint + Vector3(0.01, 0.0, 0.01)
    }};
    constexpr double step = 1e-6;
    for (const Point3& point : smoothPoints) {
        const MeanJet jet = mean.evaluate(point);
        Vector3 difference;
        for (int axis = 0; axis < 3; ++axis) {
            const Vector3 offset = step * Vector3::Unit(axis);
            difference[axis] = (mean.evaluate(point + offset).value -
                                mean.evaluate(point - offset).value) / (2.0 * step);
        }
        context.require((jet.gradient - difference).norm() < 3e-8,
                        "shader ball gradient matches finite differences in smooth regions");
        context.near(jet.gradient.norm(), 1.0, 2e-12,
                     "shader ball exact distance has unit gradient at smooth points");
    }

    const Point3 center(2.1, -3.2, 0.7);
    const ShaderBallMean translated(center, 1.0, Vector3::UnitZ(), grooveRadius);
    const ShaderBallMean scaled(Point3::Zero(), 2.5, Vector3::UnitZ(), 2.5 * grooveRadius);
    for (const Point3& point : smoothPoints) {
        const MeanJet original = mean.evaluate(point);
        const MeanJet shifted = translated.evaluate(point + center);
        const MeanJet enlarged = scaled.evaluate(2.5 * point);
        context.near(shifted.value, original.value, 2e-12, "shader ball translates with center");
        context.require((shifted.gradient - original.gradient).norm() < 2e-11,
                        "shader ball gradient translates with center");
        context.near(enlarged.value, 2.5 * original.value, 3e-12,
                     "shader ball and pedestal distances scale with radius");
        context.require((enlarged.gradient - original.gradient).norm() < 2e-11,
                        "shader ball distance gradient is scale invariant");
    }

    // Rotate the groove axis around Y. These points stay far from the fixed pedestal.
    constexpr double angle = 0.4;
    auto rotate = [angle](const Vector3& value) -> Vector3 {
        return Vector3(std::cos(angle) * value.x() + std::sin(angle) * value.z(),
                       value.y(),
                       -std::sin(angle) * value.x() + std::cos(angle) * value.z());
    };
    const ShaderBallMean tilted(Point3::Zero(), 1.0, 3.0 * rotate(Vector3::UnitZ()), grooveRadius);
    context.near(tilted.grooveAxis().norm(), 1.0, 2e-13,
                 "shader ball normalizes the supplied groove axis");
    for (const Point3& point : std::array<Point3, 3>{{
             (1.0 - grooveRadius - 0.02) * ringDirection,
             (1.0 - grooveRadius + 0.02) * ringDirection,
             Point3(0.1, 0.1, 1.1)}}) {
        const MeanJet original = mean.evaluate(point);
        const MeanJet rotated = tilted.evaluate(rotate(point));
        context.near(rotated.value, original.value, 3e-12,
                     "shader ball groove distance follows axis orientation");
        context.require((rotated.gradient - rotate(original.gradient)).norm() < 2e-11,
                        "shader ball groove gradient follows axis orientation");
    }
    const Vector3 arbitraryAxis = Vector3(1.0, 2.0, 3.0).normalized();
    const ShaderBallMean arbitraryTilt(Point3::Zero(), 1.0, arbitraryAxis, grooveRadius);
    for (double position : std::array<double, 8>{{-1.7, -1.0, -0.3, 0.0, 0.2, 0.93, 1.0, 1.7}}) {
        const MeanJet jet = arbitraryTilt.evaluate(position * arbitraryAxis);
        context.require(std::isfinite(jet.value) && jet.gradient.allFinite() &&
                            jet.gradient.norm() <= 1.0 + 1e-12,
                        "shader ball arbitrary rotation axis has finite bounded gradients");
    }

    const std::array<Bounds3, 3> domains{{
        {Point3(-1.5, -1.5, -1.5), Point3(1.5, 1.5, 1.5)},
        {Point3(0.8, -0.1, 0.1), Point3(1.2, 0.1, 0.5)},
        {Point3(0.6, 0.0, -1.4), Point3(1.2, 0.3, -0.8)}
    }};
    for (const Bounds3& domain : domains) {
        const BoundsSummary bound = tilted.bounds(domain);
        context.require(bound.certified && std::isfinite(bound.minimumValue) &&
                            std::isfinite(bound.maximumGradientNorm),
                        "shader ball supplies finite certified bounds");
        for (int ix = 0; ix <= 5; ++ix) {
            for (int iy = 0; iy <= 5; ++iy) {
                for (int iz = 0; iz <= 5; ++iz) {
                    const Point3 fraction(ix / 5.0, iy / 5.0, iz / 5.0);
                    const Point3 point = domain.minimum +
                        (domain.maximum - domain.minimum).cwiseProduct(fraction);
                    const MeanJet jet = tilted.evaluate(point);
                    context.require(jet.value + 1e-12 >= bound.minimumValue &&
                                        jet.gradient.allFinite() &&
                                        jet.gradient.norm() <= bound.maximumGradientNorm + 1e-12,
                                    "shader ball bounds enclose sampled distance and gradient");
                }
            }
        }
    }
    context.require(!mean.affineGradient().has_value(), "shader ball mean is non-affine");

    auto rejects = [&](const Point3& invalidCenter, double radius,
                       const Vector3& axis, double groove) {
        bool rejected = false;
        try {
            const ShaderBallMean invalid(invalidCenter, radius, axis, groove);
            (void)invalid;
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        context.require(rejected, "shader ball rejects invalid center, size or groove axis");
    };
    const double infinity = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    rejects(Point3::Zero(), 0.0, Vector3::UnitZ(), grooveRadius);
    rejects(Point3::Zero(), -1.0, Vector3::UnitZ(), grooveRadius);
    rejects(Point3::Zero(), infinity, Vector3::UnitZ(), grooveRadius);
    rejects(Point3::Zero(), nan, Vector3::UnitZ(), grooveRadius);
    rejects(Point3(infinity, 0.0, 0.0), 1.0, Vector3::UnitZ(), grooveRadius);
    rejects(Point3::Zero(), 1.0, Vector3::Zero(), grooveRadius);
    rejects(Point3::Zero(), 1.0, Vector3(nan, 0.0, 1.0), grooveRadius);
    rejects(Point3::Zero(), 1.0, Vector3::UnitZ(), 0.0);
    rejects(Point3::Zero(), 1.0, Vector3::UnitZ(), -0.01);
    rejects(Point3::Zero(), 1.0, Vector3::UnitZ(), 0.16);
    rejects(Point3::Zero(), 1.0, Vector3::UnitZ(), nan);
}
