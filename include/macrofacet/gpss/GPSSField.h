#pragma once

#include "macrofacet/gpss/MeanField.h"
#include "macrofacet/gpss/CovarianceKernel.h"
#include <memory>

namespace mf {

// value-gradient joint distribution at a single point, with optional prior information
struct PointPrior {
    double meanF = 0.0;
    double varianceF = 0.0;
    Vector3 meanG = Vector3::Zero();
    Matrix3 covarianceG = Matrix3::Zero();
    Vector3 covarianceFG = Vector3::Zero();
};

struct GPSSField {
    MeanFieldPtr mean;
    CovarianceKernel kernel;
    Bounds3 activeDomain;
    PointPrior pointPrior(const Point3& x) const;
    void validate() const;
};

} // namespace mf
