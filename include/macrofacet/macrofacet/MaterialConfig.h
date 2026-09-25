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
    // Classic's NDF choice. Correlated GP flights use the field's Gaussian prior.
    NdfFamily ndfFamily = NdfFamily::GeneralizedGaussian;
    Vector2 ggxAlpha = Vector2(0.5, 0.5);
    ScalarFieldPtr alphaField;

    PointPrior materialNdf(const GPSSField& field, const Point3& x) const;
    Vector2 ggxAlphaAt(const Point3& x) const;
    void validate(const Bounds3& domain) const;
};

} // namespace mf
