#include "macrofacet/gpss/SquaredExponentialKernel.h"
#include "macrofacet/mathutility/NumericPolicy.h"
#include <Eigen/Eigenvalues>
#include <cmath>
#include <stdexcept>

namespace mf {

SquaredExponentialKernel::SquaredExponentialKernel(double sigma, Matrix3 precision)
    : sigma_(sigma), precision_(0.5 * (precision + precision.transpose())) {
    if (!(sigma > 0.0) || !std::isfinite(sigma) || !precision_.allFinite()) {
        throw std::invalid_argument("invalid squared-exponential kernel parameters");
    }
    Eigen::SelfAdjointEigenSolver<Matrix3> eig(precision_);
    if (eig.info() != Eigen::Success || eig.eigenvalues().minCoeff() <
        -covarianceTolerance(std::max(1.0, precision_.norm()))) {
        throw NumericError(NumericStatus::InvalidCovariance, "kernel precision must be PSD");
    }
    precision_ = eig.eigenvectors() * eig.eigenvalues().cwiseMax(0.0).asDiagonal() *
                 eig.eigenvectors().transpose();
}

SquaredExponentialKernel SquaredExponentialKernel::fromCorrelationLengths(
    double sigma, const Vector3& lengths, const Matrix3& rotation) {
    if (!rotation.allFinite() ||
        (rotation.transpose() * rotation - Matrix3::Identity()).norm() > 1e-10 ||
        std::abs(rotation.determinant() - 1.0) > 1e-10) {
        throw std::invalid_argument("kernel rotation must be orthonormal and right-handed");
    }
    Vector3 inverseSquares;
    for (int i = 0; i < 3; ++i) {
        if (std::isinf(lengths[i]) && lengths[i] > 0.0) inverseSquares[i] = 0.0;
        else if (lengths[i] > 0.0 && std::isfinite(lengths[i])) {
            inverseSquares[i] = 1.0 / (lengths[i] * lengths[i]);
        } else {
            throw std::invalid_argument("correlation lengths must be positive or +infinity");
        }
    }
    return {sigma, rotation * inverseSquares.asDiagonal() * rotation.transpose()};
}

KernelJet SquaredExponentialKernel::evaluate(const Point3& x, const Point3& y) const {
    const Vector3 delta = x - y;
    const Vector3 v = precision_ * delta;
    const double k = sigma_ * sigma_ * std::exp(-0.5 * delta.dot(v));
    return {k, -v * k, v * k, (precision_ - v * v.transpose()) * k};
}

} // namespace mf

