#pragma once

#include "macrofacet/gpss/MeanField.h"
#include "macrofacet/gpss/SquaredExponentialKernel.h"
#include <memory>

namespace mf {

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
    PointPrior pointPrior(const Point3& x) const;
    void validate() const;
};

} // namespace mf
