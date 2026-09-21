#include "macrofacet/macrofacet/GaussianNdf.h"
#include "macrofacet/mathutility/GaussianMoments1D.h"
#include <Eigen/Eigenvalues>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace mf {

GaussianNdf::GaussianNdf(Vector3 meanGradient, Matrix3 covarianceGradient,
                         const NumericPolicy& policy)
    : mean_(std::move(meanGradient)),
      covariance_(0.5 * (covarianceGradient + covarianceGradient.transpose())), policy_(policy) {
    if (!mean_.allFinite() || !covariance_.allFinite()) {
        throw NumericError(NumericStatus::InvalidInput, "invalid Gaussian NDF parameters");
    }
    const double tolerance = covarianceTolerance(std::max(1.0, covariance_.norm()), policy_);
    const bool alignedMean = (mean_ - Vector3::UnitZ()).norm() <= tolerance;
    const bool alignedCovariance = std::abs(covariance_(0, 1)) <= tolerance &&
                                   std::abs(covariance_(0, 2)) <= tolerance &&
                                   std::abs(covariance_(1, 2)) <= tolerance &&
                                   std::abs(covariance_(2, 2)) <= tolerance;
    if (alignedMean && alignedCovariance && covariance_(0, 0) > 0.0 &&
        covariance_(1, 1) > 0.0) {
        beckmannLimit_ = true;
        alphaX_ = std::sqrt(2.0 * covariance_(0, 0));
        alphaY_ = std::sqrt(2.0 * covariance_(1, 1));
        return;
    }
    cholesky_.compute(covariance_);
    if (cholesky_.info() != Eigen::Success) {
        throw NumericError(NumericStatus::UnsupportedDegenerateNdf,
                           "Gaussian NDF requires SPD covariance or the aligned Beckmann limit");
    }
}

PositiveResult GaussianNdf::evaluateD(const Vector3& unitNormal) const {
    if (!unitNormal.allFinite() || std::abs(unitNormal.norm() - 1.0) > 1e-9) {
        throw NumericError(NumericStatus::InvalidInput, "NDF direction must be a unit vector");
    }
    if (beckmannLimit_) {
        if (!(unitNormal.z() > 0.0)) return exactZero();
        const double z2 = unitNormal.z() * unitNormal.z();
        const double exponent = -std::pow(unitNormal.x() / alphaX_, 2) / z2 -
                                std::pow(unitNormal.y() / alphaY_, 2) / z2;
        const double logD = exponent - std::log(kPi * alphaX_ * alphaY_) - 2.0 * std::log(z2);
        return positiveFromLog(logD);
    }
    const Matrix3 lower = cholesky_.matrixL();
    const Vector3 u = lower.triangularView<Eigen::Lower>().solve(mean_);
    const Vector3 v = lower.triangularView<Eigen::Lower>().solve(unitNormal);
    const double a = v.squaredNorm();
    const double m = u.dot(v) / a;
    const double s = 1.0 / std::sqrt(a);
    const Vector3 residual = u - m * v;
    const PositiveResult moment = positiveRawMoment(3, m, s, policy_);
    if (moment.status == NumericStatus::ExactZero) return moment;
    double logDeterminant = 0.0;
    for (int i = 0; i < 3; ++i) logDeterminant += 2.0 * std::log(lower(i, i));
    const double logD = -std::log(2.0 * kPi) - 0.5 * logDeterminant -
                        0.5 * std::log(a) - 0.5 * residual.squaredNorm() + moment.logValue;
    return positiveFromLog(logD, moment.absError, moment.status);
}

PositiveResult GaussianNdf::projectedArea(const Vector3& travelDirection) const {
    const Vector3 w = normalizedOrThrow(travelDirection);
    const double meanK = w.dot(mean_);
    const double varianceK = validateNonnegative(w.dot(covariance_ * w), covariance_.norm(), policy_);
    return negativePartMean(meanK, std::sqrt(varianceK), policy_);
}

double GaussianNdf::visibleNormalPdf(const Vector3& unitNormal,
                                     const Vector3& travelDirection) const {
    const double cosine = -normalizedOrThrow(travelDirection).dot(unitNormal);
    if (!(cosine > 0.0)) return 0.0;
    const PositiveResult area = projectedArea(travelDirection);
    if (area.status == NumericStatus::ExactZero) return 0.0;
    return cosine * evaluateD(unitNormal).value / area.value;
}

} // namespace mf

