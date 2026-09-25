#pragma once

#include "macrofacet/core/Random.h"
#include "macrofacet/gpss/GPSSField.h"
#include "macrofacet/macrofacet/BeckmannVisibleSampler.h"
#include "macrofacet/macrofacet/MaterialConfig.h"
#include <optional>

namespace mf {

Spectrum conductorFresnel(double cosTheta, const ConductorParameters& material);

struct PhaseSample {
    Vector3 direction = Vector3::Zero();
    Spectrum energyValue = Spectrum::Zero();
    double samplingPdf = 0.0;
    Spectrum throughputWeight = Spectrum::Zero();
    Vector3 sampledNormal = Vector3::Zero();
};

class ConductorPhase {
public:
    ConductorPhase(const GPSSField& field, const MaterialConfig& material, Point3 position,
                   double beckmannMixtureWeight = 0.0, bool useTargetVndf = false);
    Spectrum evaluateEnergy(const Vector3& w, const Vector3& wNew) const;
    double evaluateSamplingPdf(const Vector3& w, const Vector3& wNew) const;
    PhaseSample samplePhase(const Vector3& w, Random& rng) const;
private:
    double targetD(const Vector3& n) const;
    double targetArea(const Vector3& w) const;
    double targetVisiblePdf(const Vector3& n, const Vector3& w) const;
    double proposalNormalPdf(const Vector3& n, const Vector3& w, double mixtureWeight) const;
    Vector3 sampleUniformFacing(const Vector3& w, Random& rng) const;
    Vector3 sampleBeckmann(const Vector3& w, Random& rng) const;
    Vector3 sampleTargetVisible(const Vector3& w, Random& rng) const;

    MaterialConfig material_;
    Point3 position_;
    double mixtureWeight_;
    bool useTargetVndf_;
    // The target NDF, evaluated once: it is fixed at position_ and targetD /
    // targetArea are each read several times per bounce. materialNdf() reads the
    // alpha grid, so re-evaluating it per call would be a grid fetch in an inner
    // loop. targetAlpha_ is the GGX family's equivalent -- the GGX branch has no
    // covarianceG to read, so it takes alpha directly.
    PointPrior targetPrior_;
    Vector2 targetAlpha_;
    std::optional<LocalBeckmannVisibleSampler> beckmannProposal_;
};

} // namespace mf
