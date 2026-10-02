#include "TestHarness.h"
#include "macrofacet/experiments/ExperimentConfig.h"
#include <array>

void testGpssStatistics(TestContext& context) {
    using namespace mf;
    const Point3 x(0.1, -0.2, 0.4);
    const Point3 y(-0.3, 0.5, 0.2);
    for (const CovarianceKernelType type : std::array<CovarianceKernelType, 3>{
             CovarianceKernelType::SquaredExponential,
             CovarianceKernelType::Matern32,
             CovarianceKernelType::Matern52}) {
        const auto kernel = CovarianceKernel::fromCorrelationLengths(
            type, 0.3, Vector3(0.4, 0.7, 1.1));
        const KernelJet xy = kernel.evaluate(x, y);
        const KernelJet yx = kernel.evaluate(y, x);
        context.near(xy.valueValue, yx.valueValue, 1e-14, "kernel symmetry");
        context.require(
            (xy.gradientXGradientY - yx.gradientXGradientY.transpose()).norm() < 1e-13,
            "kernel derivative block transpose symmetry");
        const double epsilon = 1e-6;
        for (int axis = 0; axis < 3; ++axis) {
            Point3 plus = x;
            Point3 minus = x;
            plus[axis] += epsilon;
            minus[axis] -= epsilon;
            const double derivative = (kernel.evaluate(plus, y).valueValue -
                                       kernel.evaluate(minus, y).valueValue) /
                                      (2.0 * epsilon);
            context.relative(derivative, xy.gradientXValueY[axis], 2e-7,
                             "kernel analytic first derivative");
            const Vector3 mixed =
                (kernel.evaluate(x, y + epsilon * Vector3::Unit(axis)).gradientXValueY -
                 kernel.evaluate(x, y - epsilon * Vector3::Unit(axis)).gradientXValueY) /
                (2.0 * epsilon);
            context.require((mixed - xy.gradientXGradientY.col(axis)).norm() < 2e-7,
                            "kernel analytic mixed derivative");
        }
        const KernelJet same = kernel.evaluate(x, x);
        context.require(same.gradientXValueY.norm() == 0.0,
                        "same-point value-gradient covariance is zero");
        context.require((same.gradientXGradientY -
                         kernel.gradientCovarianceAtZero()).norm() < 1e-14,
                        "same-point KernelJet exposes the canonical gradient covariance");
    }
}

