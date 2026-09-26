#include "TestHarness.h"
#include "macrofacet/experiments/ExperimentConfig.h"

void testGpssStatistics(TestContext& context) {
    using namespace mf;
    const auto kernel = SquaredExponentialKernel::fromCorrelationLengths(
        0.3, Vector3(0.4, 0.7, 1.1));
    const Point3 x(0.1, -0.2, 0.4);
    const Point3 y(-0.3, 0.5, 0.2);
    const KernelJet xy = kernel.evaluate(x, y);
    const KernelJet yx = kernel.evaluate(y, x);
    context.near(xy.valueValue, yx.valueValue, 1e-14, "kernel symmetry");
    context.require((xy.gradientXGradientY - yx.gradientXGradientY.transpose()).norm() < 1e-13,
                    "kernel derivative block transpose symmetry");
    const double epsilon = 1e-6;
    for (int axis = 0; axis < 3; ++axis) {
        Point3 plus = x;
        Point3 minus = x;
        plus[axis] += epsilon;
        minus[axis] -= epsilon;
        const double derivative = (kernel.evaluate(plus, y).valueValue -
                                   kernel.evaluate(minus, y).valueValue) / (2.0 * epsilon);
        context.relative(derivative, xy.gradientXValueY[axis], 2e-7,
                         "kernel analytic derivative");
    }
    context.require(kernel.evaluate(x, x).gradientXValueY.norm() == 0.0,
                    "same-point value-gradient covariance is zero");

}

