#include "macrofacet/macrofacet/ConductorPhase.h"
#include "macrofacet/macrofacet/BeckmannVisibleSampler.h"
#include "macrofacet/macrofacet/GaussianNdf.h"
#include "macrofacet/macrofacet/GgxHeightfield.h"
#include "macrofacet/transport/CollisionGradientSampler.h"
#include <algorithm>
#include <complex>
#include <stdexcept>

namespace mf {

Spectrum conductorFresnel(double cosTheta, const ConductorParameters& material) {
    if (material.forceUnitFresnel) return Spectrum::Ones();
    const double c = std::clamp(std::abs(cosTheta), 0.0, 1.0);
    Spectrum result;
    for (int channel = 0; channel < 3; ++channel) {
        const std::complex<double> index(material.eta[channel], material.k[channel]);
        if (c == 0.0) { result[channel] = 1.0; continue; }
        const std::complex<double> gamma = std::sqrt(index * index - (1.0 - c * c));
        const std::complex<double> rs = (c - gamma) / (c + gamma);
        const std::complex<double> rp = (index * index * c - gamma) /
                                        (index * index * c + gamma);
        result[channel] = 0.5 * (std::norm(rs) + std::norm(rp));
    }
    return result;
}

ConductorPhase::ConductorPhase(const GPSSField& field, const MaterialConfig& material, Point3 position,
                               double beckmannMixtureWeight, bool useTargetVndf)
    : material_(material), position_(std::move(position)), mixtureWeight_(beckmannMixtureWeight),
      useTargetVndf_(useTargetVndf), targetPrior_(material.materialNdf(field, position_)),
      targetAlpha_(material.ggxAlphaAt(position_)) {
    if (!(mixtureWeight_ >= 0.0 && mixtureWeight_ < 1.0)) {
        throw std::invalid_argument("Beckmann mixture weight must lie in [0,1)");
    }
    if (!useTargetVndf_ && mixtureWeight_ > 0.0) {
        beckmannProposal_.emplace(targetPrior_.meanG, targetPrior_.covarianceG);
    }
}

double ConductorPhase::targetD(const Vector3& n) const {
    if (material_.ndfFamily == NdfFamily::GGXBaseline) {
        return GgxHeightfield(targetAlpha_.x(), targetAlpha_.y()).evaluateD(n).value;
    }
    return GaussianNdf(targetPrior_.meanG, targetPrior_.covarianceG).evaluateD(n).value;
}

double ConductorPhase::targetArea(const Vector3& w) const {
    if (material_.ndfFamily == NdfFamily::GGXBaseline) {
        return GgxHeightfield(targetAlpha_.x(), targetAlpha_.y()).projectedArea(w).value;
    }
    return GaussianNdf(targetPrior_.meanG, targetPrior_.covarianceG).projectedArea(w).value;
}

double ConductorPhase::targetVisiblePdf(const Vector3& n, const Vector3& w) const {
    const double cosine = std::max(-normalizedOrThrow(w).dot(n), 0.0);
    const double area = targetArea(w);
    return area > 0.0 ? cosine * targetD(n) / area : 0.0;
}

Vector3 ConductorPhase::sampleUniformFacing(const Vector3& w, Random& rng) const {
    const Vector3 axis = -normalizedOrThrow(w);
    Vector3 u, v;
    orthonormalComplement(axis, u, v);
    const double z = rng.openUniform01();
    const double phi = 2.0 * kPi * rng.openUniform01();
    const double radius = std::sqrt(std::max(0.0, 1.0 - z * z));
    return normalizedOrThrow(z * axis + radius * (std::cos(phi) * u + std::sin(phi) * v));
}

Vector3 ConductorPhase::sampleBeckmann(const Vector3& w, Random& rng) const {
    return beckmannProposal_->sample(w, rng);
}

Vector3 ConductorPhase::sampleTargetVisible(const Vector3& w, Random& rng) const {
    if (material_.ndfFamily == NdfFamily::GGXBaseline) {
        throw NumericError(NumericStatus::UnsupportedDegenerateNdf,
                           "target VNDF proposal requires a Gaussian NDF");
    }
    // The target VNDF is the material NDF, so this proposal is exact and the
    // MIS weight collapses; targetPrior_ is the same distribution targetD and
    // targetArea use.
    Gaussian<3> gradient;
    gradient.mean = targetPrior_.meanG;
    gradient.covariance = targetPrior_.covarianceG;
    return normalizedOrThrow(sampleFluxWeightedGradient(gradient, w, rng));
}

double ConductorPhase::proposalNormalPdf(const Vector3& n, const Vector3& w,
                                         double mixtureWeight) const {
    if (useTargetVndf_) return targetVisiblePdf(n, w);
    const double uniform = -normalizedOrThrow(w).dot(n) > 0.0 ? 1.0 / (2.0 * kPi) : 0.0;
    if (mixtureWeight == 0.0) return uniform;
    const double beckmannPdf = beckmannProposal_->pdf(n, w);
    return mixtureWeight * beckmannPdf + (1.0 - mixtureWeight) * uniform;
}

Spectrum ConductorPhase::evaluateEnergy(const Vector3& w, const Vector3& wNew) const {
    Vector3 difference = wNew - w;
    if (!(difference.norm() > 0.0)) return Spectrum::Zero();
    Vector3 n = difference.normalized();
    if (w.dot(n) >= 0.0) n = -n;
    const double cosine = std::abs(w.dot(n));
    const double area = targetArea(w);
    if (!(cosine > 0.0) || !(area > 0.0)) return Spectrum::Zero();
    return conductorFresnel(cosine, material_.conductor) * (targetD(n) / (4.0 * area));
}

double ConductorPhase::evaluateSamplingPdf(const Vector3& w, const Vector3& wNew) const {
    Vector3 difference = wNew - w;
    if (!(difference.norm() > 0.0)) return 0.0;
    Vector3 n = difference.normalized();
    if (w.dot(n) >= 0.0) n = -n;
    const double cosine = std::abs(w.dot(n));
    if (!(cosine > 0.0)) return 0.0;
    return proposalNormalPdf(n, w, mixtureWeight_) / (4.0 * cosine);
}

PhaseSample ConductorPhase::samplePhase(const Vector3& w, Random& rng) const {
    const double effectiveMixture = mixtureWeight_;
    Vector3 n = useTargetVndf_ ? sampleTargetVisible(w, rng)
        : (rng.openUniform01() < effectiveMixture ? sampleBeckmann(w, rng)
                                                  : sampleUniformFacing(w, rng));
    const double target = targetVisiblePdf(n, w);
    const double qn = useTargetVndf_ ? target : proposalNormalPdf(n, w, effectiveMixture);
    if (!(qn > 0.0) || !(target > 0.0)) {
        return {reflectTravelDirection(w, n), Spectrum::Zero(),
                qn > 0.0 ? qn / (4.0 * std::abs(w.dot(n))) : 0.0,
                Spectrum::Zero(), n};
    }
    const double cosine = std::abs(w.dot(n));
    const Vector3 outgoing = reflectTravelDirection(w, n);
    const Spectrum fresnel = conductorFresnel(cosine, material_.conductor);
    return {outgoing, fresnel * (target / (4.0 * cosine)), qn / (4.0 * cosine),
            fresnel * (target / qn), n};
}

} // namespace mf
