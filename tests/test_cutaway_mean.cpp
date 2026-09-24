#include "TestHarness.h"
#include "macrofacet/gpss/MeanField.h"
#include <array>
#include <limits>

void testCutawayMean(TestContext& context) {
    using namespace mf;
    constexpr double innerRadius = 0.4;
    constexpr double outerRadius = 1.0;
    const CutawaySphereMean mean(Point3::Zero(), innerRadius, outerRadius);

    struct DistanceCase {
        Point3 point;
        double expected;
        const char* name;
    };
    const std::array<DistanceCase, 15> cases{{
        {Point3::Zero(), -innerRadius, "center"},
        {Point3(0.2, 0.2, 0.0), std::sqrt(0.08) - innerRadius, "inside exposed small sphere"},
        {Point3(0.4, 0.4, 0.0), std::sqrt(0.32) - innerRadius, "outside exposed small sphere"},
        {Point3(-0.7, -0.2, 0.1), std::sqrt(0.54) - outerRadius, "inside retained large sphere"},
        {Point3(-1.3, 0.2, 0.1), std::sqrt(1.74) - outerRadius, "outside retained large sphere"},
        {Point3(0.15, 0.7, 0.1), 0.15, "outside x cut face"},
        {Point3(-0.15, 0.7, 0.1), -0.15, "inside x cut face"},
        {Point3(0.7, 0.15, 0.1), 0.15, "outside y cut face"},
        {Point3(0.7, -0.15, 0.1), -0.15, "inside y cut face"},
        {Point3(0.0, 0.7, 0.0), 0.0, "x cut face boundary"},
        {Point3(0.7, 0.0, 0.0), 0.0, "y cut face boundary"},
        {Point3(-0.05, -0.07, 0.7), -std::sqrt(0.0074), "concave edge of cut faces"},
        {Point3(-0.1, -0.2, 0.15), -std::sqrt(0.1125), "nearest inner pole from retained solid"},
        {Point3(0.05, 0.07, 1.1),
         std::hypot(0.05, std::sqrt(1.2149) - outerRadius), "clipped outer sphere rim"},
        {Point3(2.0, 2.0, 0.0), std::sqrt(5.0), "far outside clipped outer sphere"}
    }};
    for (const auto& sample : cases) {
        const MeanJet jet = mean.evaluate(sample.point);
        context.near(jet.value, sample.expected, 2e-13,
                     std::string("cutaway exact distance: ") + sample.name);
        context.require(jet.gradient.allFinite() && jet.gradient.norm() <= 1.0 + 1e-12,
                        std::string("cutaway finite bounded gradient: ") + sample.name);
    }

    // Compare derivatives only where the nearest boundary point is unique.
    const std::array<Point3, 10> smoothPoints{{
        Point3(0.2, 0.2, 0.0), Point3(0.4, 0.4, 0.0),
        Point3(-0.7, -0.2, 0.1), Point3(-1.3, 0.2, 0.1),
        Point3(0.15, 0.7, 0.1), Point3(-0.15, 0.7, 0.1),
        Point3(0.7, 0.15, 0.1), Point3(-0.05, -0.07, 0.7),
        Point3(-0.1, -0.2, 0.15), Point3(0.05, 0.07, 1.1)
    }};
    constexpr double step = 1e-6;
    for (const Point3& point : smoothPoints) {
        const MeanJet jet = mean.evaluate(point);
        Vector3 finiteDifference;
        for (int axis = 0; axis < 3; ++axis) {
            const Vector3 offset = step * Vector3::Unit(axis);
            finiteDifference[axis] =
                (mean.evaluate(point + offset).value - mean.evaluate(point - offset).value) /
                (2.0 * step);
        }
        context.require((jet.gradient - finiteDifference).norm() < 2e-8,
                        "cutaway gradient matches the derivative of exact signed distance");
        context.near(jet.gradient.norm(), 1.0, 2e-13,
                     "cutaway smooth distance gradient is unit length");
    }

    // Surface jets must remain usable by the transport code, including at sharp edges.
    const double diagonal = innerRadius / std::sqrt(2.0);
    const std::array<Point3, 6> surfacePoints{{
        Point3(diagonal, diagonal, 0.0), Point3(-outerRadius, 0.0, 0.0),
        Point3(0.0, 0.7, 0.0), Point3(0.7, 0.0, 0.0),
        Point3(0.0, innerRadius, 0.0), Point3(0.0, outerRadius, 0.0)
    }};
    for (const Point3& point : surfacePoints) {
        const MeanJet jet = mean.evaluate(point);
        context.near(jet.value, 0.0, 2e-13, "cutaway boundary has zero mean");
        context.require(jet.gradient.allFinite() && jet.gradient.norm() <= 1.0 + 1e-12,
                        "cutaway boundary gradient is finite and bounded");
    }
    context.require((mean.evaluate(Point3(diagonal, diagonal, 0.0)).gradient -
                     Vector3(1.0, 1.0, 0.0).normalized()).norm() < 1e-12,
                    "cutaway exposed small sphere surface has outward normal");
    context.require((mean.evaluate(Point3(-outerRadius, 0.0, 0.0)).gradient +
                     Vector3::UnitX()).norm() < 1e-12,
                    "cutaway retained outer sphere surface has outward normal");
    context.require((mean.evaluate(Point3(0.0, 0.7, 0.0)).gradient -
                     Vector3::UnitX()).norm() < 1e-12,
                    "cutaway x face surface has outward normal");

    const Point3 center(2.1, -3.2, 0.7);
    const CutawaySphereMean translated(center, innerRadius, outerRadius);
    for (const Point3& point : smoothPoints) {
        const MeanJet original = mean.evaluate(point);
        const MeanJet shifted = translated.evaluate(point + center);
        context.near(shifted.value, original.value, 2e-13, "cutaway mean translates with center");
        context.require((shifted.gradient - original.gradient).norm() < 2e-12,
                        "cutaway gradient translates with center");
    }

    const std::array<Bounds3, 3> domains{{
        {Point3::Constant(-1.4), Point3::Constant(1.4)},
        {Point3(0.1, 0.2, -0.2), Point3(0.9, 0.8, 0.3)},
        {Point3(-0.7, -0.6, 0.2), Point3(-0.1, -0.1, 0.8)}
    }};
    for (const Bounds3& domain : domains) {
        const BoundsSummary bound = mean.bounds(domain);
        context.require(bound.certified && std::isfinite(bound.minimumValue) &&
                            std::isfinite(bound.maximumGradientNorm),
                        "cutaway supplies finite certified bounds");
        for (int ix = 0; ix <= 6; ++ix) {
            for (int iy = 0; iy <= 6; ++iy) {
                for (int iz = 0; iz <= 6; ++iz) {
                    const Point3 fraction(ix / 6.0, iy / 6.0, iz / 6.0);
                    const Point3 point = domain.minimum +
                        (domain.maximum - domain.minimum).cwiseProduct(fraction);
                    const MeanJet jet = mean.evaluate(point);
                    context.require(jet.value + 1e-12 >= bound.minimumValue &&
                                        jet.gradient.norm() <= bound.maximumGradientNorm + 1e-12,
                                    "cutaway bounds enclose sampled distance and gradient");
                }
            }
        }
    }
    context.require(!mean.affineGradient().has_value(), "cutaway mean is non-affine");

    auto rejects = [&](const Point3& invalidCenter, double inner, double outer) {
        bool rejected = false;
        try {
            const CutawaySphereMean invalid(invalidCenter, inner, outer);
            (void)invalid;
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        context.require(rejected, "cutaway rejects invalid center or radii");
    };
    rejects(Point3::Zero(), 0.0, 1.0);
    rejects(Point3::Zero(), -0.1, 1.0);
    rejects(Point3::Zero(), 0.4, 0.4);
    rejects(Point3::Zero(), 0.5, 0.4);
    rejects(Point3::Zero(), 0.4, std::numeric_limits<double>::infinity());
    rejects(Point3::Zero(), std::numeric_limits<double>::quiet_NaN(), 1.0);
    rejects(Point3(std::numeric_limits<double>::infinity(), 0.0, 0.0), 0.4, 1.0);
}
