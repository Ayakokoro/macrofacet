#include "macrofacet/transport/CollisionGradientSampler.h"
#include "macrofacet/mathutility/Gaussian1D.h"
#include "macrofacet/mathutility/GaussianMoments1D.h"
#include "macrofacet/transport/FlightKernel.h"
#include <cmath>

namespace mf {

Vector3 sampleFluxWeightedGradient(const Gaussian<3>& gradient, const Vector3& wInput,
                                   Random& rng, const NumericPolicy& policy) {
    const Vector3 w = normalizedOrThrow(wInput);
    const double meanK = w.dot(gradient.mean);
    const Vector3 cw = gradient.covariance * w;
    const double varianceK = validateNonnegative(w.dot(cw), gradient.covariance.norm(), policy);
    Vector3 result;
    if (varianceK > 0.0) {
        const double k = sampleNegativeFluxNormal(meanK, std::sqrt(varianceK), rng, policy);
        const Vector3 conditionalMean = gradient.mean + cw * ((k - meanK) / varianceK);
        Matrix3 conditionalCovariance = gradient.covariance - cw * cw.transpose() / varianceK;
        conditionalCovariance = 0.5 * (conditionalCovariance + conditionalCovariance.transpose());
        Vector3 u, v;
        orthonormalComplement(w, u, v);
        Eigen::Matrix<double, 3, 2> basis;
        basis.col(0) = u;
        basis.col(1) = v;
        Gaussian<2> transverse;
        transverse.mean = basis.transpose() * conditionalMean;
        transverse.covariance = basis.transpose() * conditionalCovariance * basis;
        const Vector2 z = sampleGaussianPSD(transverse, rng, policy);
        result = k * w + basis * z;
    } else {
        if (!(meanK < 0.0)) {
            throw NumericError(NumericStatus::InvalidInput, "gradient has no crossing flux");
        }
        result = sampleGaussianPSD(gradient, rng, policy);
    }
    if (!(w.dot(result) < 0.0) || !(result.norm() > 0.0)) {
        throw NumericError(NumericStatus::NeedHigherPrecision,
                           "sampled gradient violated the crossing support");
    }
    return result;
}

static Vector3 sampleMidpointGradient(const Gaussian<4>& joint, const Vector3& w,
                                      Random& rng, const NumericPolicy& policy) {
    Gaussian<3> base;
    base.mean = joint.mean.template segment<3>(1);
    base.covariance = joint.covariance.template block<3, 3>(1, 1);
    for (int attempt = 0; attempt < 1000000; ++attempt) {
        const Vector3 gradient = sampleFluxWeightedGradient(base, w, rng, policy);
        Gaussian<1> target;
        target.mean[0] = joint.mean[0];
        target.covariance(0, 0) = joint.covariance(0, 0);
        Gaussian<3> observation = base;
        Eigen::Matrix<double, 1, 3> cross = joint.covariance.template block<1, 3>(0, 1);
        const Gaussian<1> yGivenG = conditionGaussian(target, observation, cross, gradient, policy);
        double probability;
        if (yGivenG.covariance(0, 0) > 0.0) {
            probability = normalCdf(yGivenG.mean[0] / std::sqrt(yGivenG.covariance(0, 0)));
        } else {
            probability = yGivenG.mean[0] > 0.0 ? 1.0 : 0.0;
        }
        if (rng.openUniform01() < probability) return gradient;
    }
    throw NumericError(NumericStatus::SamplingNotConverged,
                       "midpoint gradient rejection sampler exhausted its budget");
}

Vector3 sampleCollisionGradient(const FlightKernel& kernel, double age, Random& rng,
                                const NumericPolicy& policy) {
    const HitStatistics statistics = kernel.hitStatistics(age);
    if (kernel.mode() == ModelMode::Midpoint &&
        statistics.midpointAndGradientGivenEndpointZero.has_value()) {
        return sampleMidpointGradient(*statistics.midpointAndGradientGivenEndpointZero,
                                      kernel.state().direction, rng, policy);
    }
    return sampleFluxWeightedGradient(statistics.gradientGivenEndpointZero,
                                      kernel.state().direction, rng, policy);
}

} // namespace mf

