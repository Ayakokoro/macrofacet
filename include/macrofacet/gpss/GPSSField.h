#pragma once

#include "macrofacet/gpss/MeanField.h"
#include "macrofacet/gpss/SquaredExponentialKernel.h"
#include <memory>

namespace mf {

enum class NdfFamily { GeneralizedGaussian, BeckmannLimit, GGXBaseline };

struct ConductorParameters {
    Spectrum eta = Spectrum(0.2, 0.9, 1.1);
    Spectrum k = Spectrum(3.9, 2.5, 2.2);
    bool forceUnitFresnel = false;
};

struct PointPrior {
    double meanF = 0.0;
    double varianceF = 0.0;
    Vector3 meanG = Vector3::Zero();
    Matrix3 covarianceG = Matrix3::Zero();
    Vector3 covarianceFG = Vector3::Zero();
};

struct GPSSField {
    MeanFieldPtr mean;
    SquaredExponentialKernel kernel;
    Bounds3 activeDomain;
    ConductorParameters conductor;
    NdfFamily ndfFamily = NdfFamily::GeneralizedGaussian;
    Vector2 ggxAlpha = Vector2(0.5, 0.5);

    PointPrior pointPrior(const Point3& x) const;
    void validate() const;
};

} // namespace mf

