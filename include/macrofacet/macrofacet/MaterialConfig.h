#pragma once

#include "macrofacet/fields/ScalarField.h"
#include "macrofacet/gpss/GPSSField.h"

namespace mf {

enum class NdfFamily { GeneralizedGaussian, BeckmannLimit, GGXBaseline };

struct ConductorParameters {
    Spectrum eta = Spectrum(0.2, 0.9, 1.1);
    Spectrum k = Spectrum(3.9, 2.5, 2.2);
    bool forceUnitFresnel = false;
};

struct MaterialConfig {
    ConductorParameters conductor;
    // Classic's local NDF choice for extinction and scattering.
    NdfFamily ndfFamily = NdfFamily::GeneralizedGaussian;
    Vector2 ggxAlpha = Vector2(0.5, 0.5);
    // Spatial Beckmann alpha. For the Gaussian material NDF each gradient
    // component has variance alpha(x)^2 / 2; GGX reads alpha(x) directly.
    ScalarFieldPtr alphaField;

    // Local tangent-plane material prior. Its mean at the origin is zero with
    // unit normal gradient; the field kernel supplies covariance unless alpha
    // is overridden per point by alphaField.
    PointPrior materialNdf(const GPSSField& field, const Point3& x) const;
    Vector2 ggxAlphaAt(const Point3& x) const;
    void validate(const Bounds3& domain) const;
};

} // namespace mf
