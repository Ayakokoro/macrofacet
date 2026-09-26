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

Vector3 sampleCollisionGradient(const FlightKernel& kernel, double age, Random& rng,
                                const NumericPolicy& policy) {
    const HitStatistics statistics = kernel.hitStatistics(age);
    return sampleFluxWeightedGradient(statistics.collisionGradient,
                                      kernel.state().direction, rng, policy);
}

} // namespace mf

