#pragma once

#include "macrofacet/core/Random.h"

namespace mf {

// PBRT v3 visible-slope Beckmann proposal. The input is a travel direction;
// PBRT's corresponding view direction is its negation.
Vector3 samplePbrtBeckmannVisible(const Vector3& travelDirection, double alphaX,
                                 double alphaY, Random& rng);
double pbrtBeckmannVisiblePdf(const Vector3& normal, const Vector3& travelDirection,
                             double alphaX, double alphaY);

// The PBRT sampler works around +Z. This proposal constructs that frame from
// the mean gradient and diagonalizes the covariance within its tangent plane.
class LocalBeckmannVisibleSampler {
public:
    LocalBeckmannVisibleSampler(const Vector3& meanGradient,
                               const Matrix3& covarianceGradient);
    Vector3 sample(const Vector3& travelDirection, Random& rng) const;
    double pdf(const Vector3& normal, const Vector3& travelDirection) const;

private:
    Vector3 toLocal(const Vector3& vector) const;
    Vector3 toWorld(const Vector3& vector) const;

    Vector3 tangentX_ = Vector3::UnitX();
    Vector3 tangentY_ = Vector3::UnitY();
    Vector3 normal_ = Vector3::UnitZ();
    double alphaX_ = 1.0;
    double alphaY_ = 1.0;
};

} // namespace mf
