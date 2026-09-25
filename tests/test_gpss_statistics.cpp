#include "TestHarness.h"
#include "macrofacet/experiments/ExperimentConfig.h"
#include "macrofacet/gpss/ConditionedRay.h"
#include <Eigen/Eigenvalues>

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

    GPSSField field = buildDefaultField();
    const Vector3 direction = normalizedOrThrow(Vector3(0.8, 0.0, 0.6));
    ConditionedRay ray(field, Point3::Zero(), Vector3::UnitZ(), direction);
    const Gaussian<2> origin = ray.endpointValueSlope(0.0);
    context.near(origin.mean[0], 0.0, 1e-15, "conditioned origin value");
    context.near(origin.mean[1], direction.z(), 1e-14, "conditioned origin slope");
    context.near(origin.covariance.norm(), 0.0, 1e-15, "conditioned origin covariance");
    for (double age : {1e-8, 1e-6, 1e-4, 0.02, 1.0}) {
        const Gaussian<2> yfk = ray.endpointValueSlope(age);
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eig(yfk.covariance);
        context.require(eig.eigenvalues().minCoeff() >= -1e-12,
                        "near-origin conditional covariance remains PSD");
        context.require(yfk.mean.allFinite() && yfk.covariance.allFinite(),
                        "conditioned ray statistics stay finite");
    }
    const Gaussian<2> far = ray.endpointValueSlope(2.0);
    const PointPrior prior = field.pointPrior(2.0 * direction);
    context.relative(far.covariance(0, 0), prior.varianceF, 2e-5,
                     "far conditional variance recovers prior");
}

