#pragma once

#include "macrofacet/macrofacet/ConductorPhase.h"
#include "macrofacet/macrofacet/GaussianNdf.h"
#include "macrofacet/mathutility/SmallGaussian.h"

namespace mf {

// Marginal direction evaluation for a fixed flight endpoint, before its full
// gradient has been sampled. Continuation still samples and retains that gradient.
class ConditionalConductorPhase {
public:
    ConditionalConductorPhase(const Gaussian<3>& endpointGradient,
                              const ConductorParameters& material,
                              const NumericPolicy& policy = defaultNumericPolicy())
        : ndf_(endpointGradient.mean, endpointGradient.covariance, policy), material_(material) {}

    double evaluateSamplingPdf(const Vector3& travel, const Vector3& outgoing) const {
        const Vector3 d = normalizedOrThrow(travel), v = normalizedOrThrow(outgoing);
        if ((v - d).squaredNorm() == 0.0) return 0.0;
        const Vector3 n = normalizedOrThrow(v - d);
        const auto area = ndf_.projectedArea(d);
        if (area.status == NumericStatus::ExactZero) return 0.0;
        return std::exp(ndf_.evaluateD(n).logValue - area.logValue - std::log(4.0));
    }

    Spectrum evaluateEnergy(const Vector3& travel, const Vector3& outgoing) const {
        const Vector3 d = normalizedOrThrow(travel), v = normalizedOrThrow(outgoing);
        if ((v - d).squaredNorm() == 0.0) return Spectrum::Zero();
        const Vector3 n = normalizedOrThrow(v - d);
        return conductorFresnel(-d.dot(n), material_) * evaluateSamplingPdf(d, v);
    }

private:
    GaussianNdf ndf_;
    ConductorParameters material_;
};

} // namespace mf
