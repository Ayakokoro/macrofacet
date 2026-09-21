#pragma once

#include "macrofacet/core/Types.h"
#include "macrofacet/mathutility/NumericPolicy.h"

namespace mf {

class GaussianNdf {
public:
    GaussianNdf(Vector3 meanGradient, Matrix3 covarianceGradient,
                const NumericPolicy& policy = defaultNumericPolicy());

    PositiveResult evaluateD(const Vector3& unitNormal) const;
    PositiveResult projectedArea(const Vector3& travelDirection) const;
    double visibleNormalPdf(const Vector3& unitNormal, const Vector3& travelDirection) const;
    bool isBeckmannLimit() const { return beckmannLimit_; }

private:
    Vector3 mean_;
    Matrix3 covariance_;
    Eigen::LLT<Matrix3> cholesky_;
    bool beckmannLimit_ = false;
    double alphaX_ = 0.0;
    double alphaY_ = 0.0;
    NumericPolicy policy_;
};

} // namespace mf

