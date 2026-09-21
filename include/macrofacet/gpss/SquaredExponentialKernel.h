#pragma once

#include "macrofacet/core/Types.h"

namespace mf {

struct KernelJet {
    double valueValue = 0.0;
    Vector3 gradientXValueY = Vector3::Zero();
    Vector3 valueXGradientY = Vector3::Zero();
    Matrix3 gradientXGradientY = Matrix3::Zero();
};

class SquaredExponentialKernel {
public:
    SquaredExponentialKernel() : sigma_(1.0), precision_(Matrix3::Identity()) {}
    SquaredExponentialKernel(double sigma, Matrix3 precision);
    static SquaredExponentialKernel fromCorrelationLengths(
        double sigma, const Vector3& lengths, const Matrix3& rotation = Matrix3::Identity());

    KernelJet evaluate(const Point3& x, const Point3& y) const;
    double sigma() const { return sigma_; }
    const Matrix3& precision() const { return precision_; }

private:
    double sigma_;
    Matrix3 precision_;
};

} // namespace mf
