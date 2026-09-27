#pragma once

#include "macrofacet/fields/ScalarField.h"
#include "macrofacet/gpss/GPSSField.h"

namespace mf {

enum class NdfFamily { GeneralizedGaussian, BeckmannLimit, GGXBaseline };
enum class GpModel { LocalTangent, GlobalPointwise };

struct ConductorParameters {
    Spectrum eta = Spectrum(0.2, 0.9, 1.1);
    Spectrum k = Spectrum(3.9, 2.5, 2.2);
    bool forceUnitFresnel = false;
};

struct MaterialConfig {
    ConductorParameters conductor;
    // The same point prior drives extinction, collision normals and scattering.
    GpModel gpModel = GpModel::LocalTangent;
    NdfFamily ndfFamily = NdfFamily::GeneralizedGaussian;
    Vector2 ggxAlpha = Vector2(0.5, 0.5);
    // Spatial Beckmann alpha. For the Gaussian material NDF each gradient
    // component has variance alpha(x)^2 / 2; GGX reads alpha(x) directly.
    ScalarFieldPtr alphaField;

    // Local mode centres a tangent-plane GP at x and may replace its gradient
    // covariance with alpha(x)^2 / 2 I. Global mode uses field.pointPrior(x)
    // unchanged; the render still evaluates only pointwise marginals.
    PointPrior materialNdf(const GPSSField& field, const Point3& x) const;
    Vector2 ggxAlphaAt(const Point3& x) const;
    void validate(const Bounds3& domain) const;
};

} // namespace mf
