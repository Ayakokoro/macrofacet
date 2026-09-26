#include "TestHarness.h"
#include "macrofacet/mathutility/Gaussian1D.h"
#include "macrofacet/mathutility/GaussianMoments1D.h"
#include "macrofacet/mathutility/Quadrature.h"
#include "macrofacet/mathutility/SmallGaussian.h"

void testGaussianMath(TestContext& context) {
    using namespace mf;
    context.near(normalCdf(0.0), 0.5, 1e-15, "Phi(0)");
    context.near(normalPdf(0.0), kInvSqrtTwoPi, 1e-15, "phi(0)");
    for (double z : {-40.0, -20.0, -8.0, -2.0, 0.0, 2.0, 8.0, 20.0, 40.0}) {
        context.require(std::isfinite(normalLogCdf(z)), "normal log CDF remains finite");
        context.near(normalCdf(z) + normalCdf(-z), 1.0, 2e-15, "normal CDF symmetry");
    }
    context.relative(normalPdfOverCdf(-40.0), 40.0249688, 2e-7, "deep-tail inverse Mills ratio");
    for (double probability : {1e-12, 1e-6, 0.01, 0.5, 0.99, 1.0 - 1e-12}) {
        const double z = normalQuantile(probability);
        context.relative(normalCdf(z), probability, 2e-10, "normal quantile round trip");
    }

    NumericPolicy policy;
    policy.relativeTolerance = 1e-9;
    policy.absoluteTolerance = 1e-12;
    for (int order = 0; order <= 3; ++order) {
        const double mean = -1.7;
        const double stddev = 0.8;
        auto integrand = [=](double x) {
            const double z = (x - mean) / stddev;
            return std::pow(x, order) * normalPdf(z) / stddev;
        };
        const IntegralResult reference = integrateSemiInfinite(integrand, 0.0, policy);
        const PositiveResult moment = positiveRawMoment(order, mean, stddev, policy);
        context.relative(moment.value, reference.value, 2e-7, "truncated Gaussian moment");
    }

    Gaussian<1> target;
    target.mean[0] = 2.0;
    target.covariance(0, 0) = 4.0;
    Gaussian<1> observation;
    observation.mean[0] = -1.0;
    observation.covariance(0, 0) = 9.0;
    Eigen::Matrix<double, 1, 1> cross;
    cross(0, 0) = 3.0;
    Eigen::Matrix<double, 1, 1> observed;
    observed[0] = 2.0;
    const Gaussian<1> conditioned = conditionGaussian(target, observation, cross, observed);
    context.near(conditioned.mean[0], 3.0, 1e-12, "Gaussian conditional mean");
    context.near(conditioned.covariance(0, 0), 3.0, 1e-12, "Gaussian conditional variance");

}

