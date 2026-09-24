#include "macrofacet/gpss/ConditionedRay.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace mf {
namespace {

// z = (t / ell)^2. Avoid cancellation of the O(z^2) value variance
// and O(z^4) Schur-complement numerator near a known surface point.
double valueVarianceFraction(double z) {
    if (z < 0.1) {
        double powerOverFactorial = z * z / 2.0;
        double sum = powerOverFactorial;
        for (int n = 3; n <= 18; ++n) {
            powerOverFactorial *= -z / n;
            sum += (n - 1) * powerOverFactorial;
        }
        return sum;
    }
    return -std::expm1(-z) - z * std::exp(-z);
}

double zeroSlopeVarianceFraction(double z) {
    if (z == 0.0) return 0.0;
    const double e = std::exp(-z);
    double numerator;
    if (z < 0.5) {
        const double z2 = z * z;
        numerator = e * z2 * z2 * (1.0 / 12.0 + z2 * (1.0 / 360.0 +
            z2 * (1.0 / 20160.0 + z2 * (1.0 / 1814400.0 +
            z2 * (1.0 / 239500800.0 + z2 / 43589145600.0)))));
    } else {
        const double oneMinusE = -std::expm1(-z);
        numerator = oneMinusE * oneMinusE - z * z * e;
    }
    return numerator / valueVarianceFraction(z);
}

} // namespace

ConditionedRay::ConditionedRay(const GPSSField& field, Point3 x0, Vector3 g0, Vector3 w,
                               const NumericPolicy& policy)
    : ConditionedRay(field, std::move(x0), 0.0, std::move(g0), std::move(w), policy) {}

ConditionedRay::ConditionedRay(const GPSSField& field, Point3 x0, double f0,
                               Vector3 g0, Vector3 w, const NumericPolicy& policy)
    : field_(field), x0_(std::move(x0)), g0_(std::move(g0)), w_(normalizedOrThrow(w)),
      rayPrecision_(w_.dot(field.kernel.precision() * w_)), policy_(policy), f0_(f0) {
    field_.validate();
    if (!x0_.allFinite() || !g0_.allFinite() || !std::isfinite(f0_) || f0_ < 0.0 ||
        (f0_ == 0.0 && !(w_.dot(g0_) > 0.0))) {
        throw std::invalid_argument("conditioned ray requires an outward surface or positive exterior observation");
    }
    originMean_ = field_.mean->evaluate(x0_);
    const PointPrior originPrior = field_.pointPrior(x0_);
    observation_.mean << originPrior.meanF, originPrior.meanG.x(), originPrior.meanG.y(),
                         originPrior.meanG.z();
    observation_.covariance.setZero();
    observation_.covariance(0, 0) = originPrior.varianceF;
    observation_.covariance.template block<3, 3>(1, 1) = originPrior.covarianceG;
    observed_ << f0_, g0_.x(), g0_.y(), g0_.z();

    // This support check catches impossible observations for height-field limits.
    Gaussian<1> dummyTarget;
    dummyTarget.mean[0] = observation_.mean[0];
    dummyTarget.covariance(0, 0) = observation_.covariance(0, 0);
    Eigen::Matrix<double, 1, 4> cross = observation_.covariance.row(0);
    (void)conditionGaussian(dummyTarget, observation_, cross, observed_, policy_);
}

double ConditionedRay::priorCovariance(const Descriptor& a, const Descriptor& b) const {
    const KernelJet jet = field_.kernel.evaluate(a.point, b.point);
    if (a.component < 0 && b.component < 0) return jet.valueValue;
    if (a.component >= 0 && b.component < 0) return jet.gradientXValueY[a.component];
    if (a.component < 0 && b.component >= 0) return jet.valueXGradientY[b.component];
    return jet.gradientXGradientY(a.component, b.component);
}

DynamicGaussian ConditionedRay::conditionDescriptors(
    const std::vector<Descriptor>& descriptors) const {
    const int n = static_cast<int>(descriptors.size());
    DynamicGaussian target{Eigen::VectorXd(n), Eigen::MatrixXd(n, n)};
    Eigen::MatrixXd cross(n, 4);
    const Descriptor originValue{x0_, -1};
    std::array<Descriptor, 4> observations{{originValue, {x0_, 0}, {x0_, 1}, {x0_, 2}}};
    for (int i = 0; i < n; ++i) {
        const MeanJet jet = field_.mean->evaluate(descriptors[static_cast<std::size_t>(i)].point);
        target.mean[i] = descriptors[static_cast<std::size_t>(i)].component < 0
                             ? jet.value : jet.gradient[descriptors[static_cast<std::size_t>(i)].component];
        for (int j = 0; j < n; ++j) {
            target.covariance(i, j) = priorCovariance(
                descriptors[static_cast<std::size_t>(i)], descriptors[static_cast<std::size_t>(j)]);
        }
        for (int j = 0; j < 4; ++j) {
            cross(i, j) = priorCovariance(descriptors[static_cast<std::size_t>(i)],
                                          observations[static_cast<std::size_t>(j)]);
        }
    }
    DynamicGaussian observation{observation_.mean, observation_.covariance};
    return conditionGaussianDynamic(target, observation, cross, observed_, policy_);
}

double ConditionedRay::stableValueCovariance(double s, double t) const {
    const double sigma2 = field_.kernel.sigma() * field_.kernel.sigma();
    const double a = rayPrecision_;
    if (a == 0.0) return 0.0;
    const double r = a * s * t;
    if (r < 0.5) {
        double difference;
        if (std::abs(r) < 1e-3) {
            double term = 0.5 * r * r;
            difference = term;
            for (int n = 3; n <= 12; ++n) {
                term *= r / static_cast<double>(n);
                difference += term;
                if (std::abs(term) < 1e-18 * std::max(1.0, std::abs(difference))) break;
            }
        } else {
            difference = std::expm1(r) - r;
        }
        return sigma2 * std::exp(-0.5 * a * (s * s + t * t)) * difference;
    }
    return sigma2 * (std::exp(-0.5 * a * (s - t) * (s - t)) -
                     std::exp(-0.5 * a * (s * s + t * t)) * (1.0 + r));
}

double ConditionedRay::stableValueSlopeCovariance(double s, double t) const {
    const double sigma2 = field_.kernel.sigma() * field_.kernel.sigma();
    const double a = rayPrecision_;
    if (a == 0.0) return 0.0;
    const double first = std::exp(-0.5 * a * (s - t) * (s - t));
    const double product = std::exp(-0.5 * a * (s * s + t * t));
    return sigma2 * (a * (s - t) * first +
                     product * (a * t * (1.0 + a * s * t) - a * s));
}

double ConditionedRay::stableSlopeVariance(double t) const {
    const double sigma2 = field_.kernel.sigma() * field_.kernel.sigma();
    const double a = rayPrecision_;
    const double q = a * t * t;
    if (a == 0.0) return 0.0;
    if (q == 0.0) return 0.0;
    const double bracket = zeroSlopeVarianceFraction(q) +
        std::exp(-2.0 * q) * q * q * q / valueVarianceFraction(q);
    return sigma2 * a * validateNonnegative(bracket, 1.0, policy_);
}

std::pair<double, double> ConditionedRay::conditionedValueSlopeMean(double t) const {
    if (t == 0.0) return {f0_, w_.dot(g0_)};
    const MeanJet endpoint = field_.mean->evaluate(x0_ + t * w_);
    const double deltaF = f0_ - originMean_.value;
    const double deltaK = w_.dot(g0_ - originMean_.gradient);
    const double exponent = -0.5 * rayPrecision_ * t * t;
    const double r = std::exp(exponent);
    const double meanRemainder = field_.mean->affineGradient() ? 0.0 :
        field_.mean->valueDifference(x0_, t * w_) - t * w_.dot(originMean_.gradient);
    return {f0_ + meanRemainder + t * w_.dot(g0_) +
                std::expm1(exponent) * (deltaF + t * deltaK),
            w_.dot(endpoint.gradient) + r * (deltaK - rayPrecision_ * t * (deltaF + t * deltaK))};
}

Gaussian<2> ConditionedRay::endpointValueSlope(double t) const {
    if (!(t >= 0.0) || !std::isfinite(t)) throw std::invalid_argument("invalid ray age");
    Gaussian<2> result;
    const auto mean = conditionedValueSlopeMean(t);
    result.mean << mean.first, mean.second;
    if (t == 0.0) return result;
    const double sigma2 = field_.kernel.sigma() * field_.kernel.sigma();
    const double q = rayPrecision_ * t * t;
    const double varianceF = sigma2 * valueVarianceFraction(q);
    const double covariance = sigma2 * rayPrecision_ * rayPrecision_ * t * t * t * std::exp(-q);
    result.covariance << validateNonnegative(varianceF, sigma2, policy_), covariance,
                         covariance, stableSlopeVariance(t);
    return result;
}

Gaussian<3> ConditionedRay::midpointValueSlope(double t) const {
    Gaussian<3> result;
    const double half = 0.5 * t;
    const auto midpointMean = conditionedValueSlopeMean(half);
    const auto endpointMean = conditionedValueSlopeMean(t);
    result.mean << midpointMean.first, endpointMean.first, endpointMean.second;
    result.covariance.setZero();
    result.covariance(0, 0) = stableValueCovariance(half, half);
    result.covariance(0, 1) = result.covariance(1, 0) = stableValueCovariance(half, t);
    result.covariance(0, 2) = result.covariance(2, 0) = stableValueSlopeCovariance(half, t);
    const Gaussian<2> endpoint = endpointValueSlope(t);
    result.covariance.template block<2, 2>(1, 1) = endpoint.covariance;
    return result;
}

Gaussian<4> ConditionedRay::endpointValueGradient(double t) const {
    if (!(t >= 0.0) || !std::isfinite(t)) throw std::invalid_argument("invalid ray age");
    Gaussian<4> result;
    if (t == 0.0) {
        result.mean << f0_, g0_.x(), g0_.y(), g0_.z();
        return result;
    }
    const double z = rayPrecision_ * t * t;
    const double r = std::exp(-0.5 * z);
    const double sigma2 = field_.kernel.sigma() * field_.kernel.sigma();
    const Matrix3& a = field_.kernel.precision();
    const Vector3 aw = a * w_;
    const double deltaF = f0_ - originMean_.value;
    const Vector3 deltaG = g0_ - originMean_.gradient;
    result.mean.template tail<3>() = field_.mean->evaluate(x0_ + t * w_).gradient +
        r * (deltaG - t * aw * (deltaF + t * w_.dot(deltaG)));
    const Gaussian<2> stable = endpointValueSlope(t);
    result.mean[0] = stable.mean[0];
    result.covariance.template block<3, 1>(1, 0) = sigma2 * std::exp(-z) * t * z * aw;
    result.covariance.template block<1, 3>(0, 1) =
        result.covariance.template block<3, 1>(1, 0).transpose();
    result.covariance(0, 0) = stable.covariance(0, 0);
    result.covariance.template block<3, 3>(1, 1) = sigma2 *
        (-std::expm1(-z) * a + std::exp(-z) * t * t * (1.0 - z) * aw * aw.transpose());
    return result;
}

Gaussian<1> ConditionedRay::slopeGivenEndpointZero(double t) const {
    const Gaussian<2> fk = endpointValueSlope(t);
    if (!(fk.covariance(0, 0) > 0.0)) {
        throw NumericError(NumericStatus::UnsupportedSingularFlight, "zero-value conditioning has no continuous density");
    }
    const double z = rayPrecision_ * t * t;
    Gaussian<1> result;
    result.mean[0] = fk.mean[1] - fk.covariance(0, 1) / fk.covariance(0, 0) * fk.mean[0];
    result.covariance(0, 0) = field_.kernel.sigma() * field_.kernel.sigma() *
                             rayPrecision_ * zeroSlopeVarianceFraction(z);
    return result;
}

Gaussian<3> ConditionedRay::gradientGivenEndpointZero(double t) const {
    const Gaussian<4> fg = endpointValueGradient(t);
    if (!(fg.covariance(0, 0) > 0.0)) {
        throw NumericError(NumericStatus::UnsupportedSingularFlight, "zero-value conditioning has no continuous density");
    }
    Gaussian<3> result;
    result.mean = fg.mean.template tail<3>() -
        fg.covariance.template block<3, 1>(1, 0) * (fg.mean[0] / fg.covariance(0, 0));
    const double z = rayPrecision_ * t * t;
    // Construct the Schur complement from nonnegative whitened components.
    Eigen::SelfAdjointEigenSolver<Matrix3> eig(field_.kernel.precision());
    const Matrix3 squareRoot = eig.eigenvectors() * eig.eigenvalues().cwiseMax(0.0).cwiseSqrt().asDiagonal() *
                              eig.eigenvectors().transpose();
    const Vector3 p = normalizedOrThrow(squareRoot * w_);
    Vector3 u, v;
    orthonormalComplement(p, u, v);
    const Vector3 su = squareRoot * u, sv = squareRoot * v, sp = squareRoot * p;
    result.covariance = field_.kernel.sigma() * field_.kernel.sigma() *
        (-std::expm1(-z) * (su * su.transpose() + sv * sv.transpose()) +
         zeroSlopeVarianceFraction(z) * sp * sp.transpose());
    return result;
}

Gaussian<5> ConditionedRay::midpointValueGradient(double t) const {
    const Point3 midpoint = x0_ + 0.5 * t * w_;
    const Point3 endpoint = x0_ + t * w_;
    DynamicGaussian dynamic = conditionDescriptors(
        {{midpoint, -1}, {endpoint, -1}, {endpoint, 0}, {endpoint, 1}, {endpoint, 2}});
    Gaussian<5> result{dynamic.mean, dynamic.covariance};
    const Gaussian<3> stable = midpointValueSlope(t);
    result.mean[0] = stable.mean[0];
    result.mean[1] = stable.mean[1];
    result.covariance(0, 0) = stable.covariance(0, 0);
    result.covariance(0, 1) = result.covariance(1, 0) = stable.covariance(0, 1);
    result.covariance(1, 1) = stable.covariance(1, 1);
    return result;
}

DynamicGaussian ConditionedRay::valuesAt(const std::vector<double>& ages) const {
    const int n = static_cast<int>(ages.size());
    DynamicGaussian result{Eigen::VectorXd(n), Eigen::MatrixXd(n, n)};
    for (int i = 0; i < n; ++i) {
        if (!(ages[static_cast<std::size_t>(i)] > 0.0) ||
            (i > 0 && !(ages[static_cast<std::size_t>(i)] > ages[static_cast<std::size_t>(i - 1)]))) {
            throw std::invalid_argument("conditioned-ray ages must be strictly increasing and positive");
        }
        result.mean[i] = conditionedValueSlopeMean(ages[static_cast<std::size_t>(i)]).first;
        for (int j = 0; j < n; ++j) {
            result.covariance(i, j) = stableValueCovariance(
                ages[static_cast<std::size_t>(i)], ages[static_cast<std::size_t>(j)]);
        }
    }
    return result;
}

DynamicGaussian ConditionedRay::checkpointValuesAndEndpointSlope(
    const std::vector<double>& interiorAges, double t) const {
    for (std::size_t i = 0; i < interiorAges.size(); ++i) {
        if (!(interiorAges[i] > 0.0 && interiorAges[i] < t) ||
            (i > 0 && !(interiorAges[i] > interiorAges[i - 1]))) {
            throw std::invalid_argument("interior ages must be sorted strictly inside (0,t)");
        }
    }
    const int n = static_cast<int>(interiorAges.size());
    DynamicGaussian result{Eigen::VectorXd(n + 2), Eigen::MatrixXd::Zero(n + 2, n + 2)};
    for (int i = 0; i < n; ++i) {
        result.mean[i] = conditionedValueSlopeMean(interiorAges[static_cast<std::size_t>(i)]).first;
    }
    const auto endpointMean = conditionedValueSlopeMean(t);
    result.mean[n] = endpointMean.first;
    result.mean[n + 1] = endpointMean.second;
    for (int i = 0; i <= n; ++i) {
        const double si = i < n ? interiorAges[static_cast<std::size_t>(i)] : t;
        for (int j = 0; j <= n; ++j) {
            const double sj = j < n ? interiorAges[static_cast<std::size_t>(j)] : t;
            result.covariance(i, j) = stableValueCovariance(si, sj);
        }
        result.covariance(i, n + 1) = result.covariance(n + 1, i) =
            stableValueSlopeCovariance(si, t);
    }
    result.covariance(n + 1, n + 1) = stableSlopeVariance(t);
    return result;
}

} // namespace mf
