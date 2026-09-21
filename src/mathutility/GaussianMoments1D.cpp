#include "macrofacet/mathutility/GaussianMoments1D.h"
#include "macrofacet/mathutility/Gaussian1D.h"
#include "macrofacet/mathutility/Quadrature.h"
#include "macrofacet/mathutility/RootFinding.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace mf {

PositiveResult positiveRawMoment(int order, double mean, double stddev,
                                 const NumericPolicy& policy) {
    if (order < 0 || order > 3 || stddev < 0.0 || !std::isfinite(mean) ||
        !std::isfinite(stddev)) {
        throw NumericError(NumericStatus::InvalidInput, "invalid truncated Gaussian moment");
    }
    if (stddev == 0.0) {
        if (!(mean > 0.0)) return exactZero();
        const double value = order == 0 ? 1.0 : std::pow(mean, order);
        return {value, std::log(value), 0.0, NumericStatus::Degenerate};
    }
    const double a = mean / stddev;
    if (order == 0) return positiveFromLog(normalLogCdf(a));

    // Direct expressions are excellent except in the cancelling negative tail.
    if (a > -5.0) {
        const double p = normalCdf(a);
        const double d = normalPdf(a);
        double value = 0.0;
        if (order == 1) value = mean * p + stddev * d;
        if (order == 2) value = (mean * mean + stddev * stddev) * p + mean * stddev * d;
        if (order == 3) {
            value = (mean * mean * mean + 3.0 * mean * stddev * stddev) * p +
                    (mean * mean * stddev + 2.0 * stddev * stddev * stddev) * d;
        }
        value = validateNonnegative(value, std::pow(std::abs(mean) + stddev, order), policy);
        if (value > 0.0) return {value, std::log(value), 0.0, NumericStatus::Ok};
    }

    const double r = static_cast<double>(order);
    const double root = std::hypot(a, 2.0 * std::sqrt(r));
    const double mode = a >= 0.0 ? 0.5 * (a + root) : 2.0 * r / (root - a);
    auto logIntegrand = [=](double u) {
        if (!(u > 0.0)) return -std::numeric_limits<double>::infinity();
        return r * std::log(u) + normalLogPdf(u - a);
    };
    PositiveResult dimensionless = integratePositiveLog(logIntegrand, 0.0, mode, policy);
    if (dimensionless.status == NumericStatus::ExactZero) return dimensionless;
    const double logValue = r * std::log(stddev) + dimensionless.logValue;
    return positiveFromLog(logValue, dimensionless.absError * std::pow(stddev, order),
                           dimensionless.status);
}

PositiveResult negativePartMean(double mean, double stddev, const NumericPolicy& policy) {
    return positiveRawMoment(1, -mean, stddev, policy);
}

double negativeFluxNormalLogPdf(double k, double mean, double stddev,
                                const NumericPolicy& policy) {
    if (!(k < 0.0)) return -std::numeric_limits<double>::infinity();
    const PositiveResult normalizer = negativePartMean(mean, stddev, policy);
    if (normalizer.status == NumericStatus::ExactZero) {
        throw NumericError(NumericStatus::InvalidInput, "negative-flux distribution has no mass");
    }
    if (stddev == 0.0) {
        return k == mean ? std::numeric_limits<double>::infinity()
                         : -std::numeric_limits<double>::infinity();
    }
    const double z = (k - mean) / stddev;
    return std::log(-k) + normalLogPdf(z) - std::log(stddev) - normalizer.logValue;
}

PositiveResult negativeFluxNormalCdf(double k, double mean, double stddev,
                                     const NumericPolicy& policy) {
    if (stddev < 0.0 || !std::isfinite(mean) || !std::isfinite(stddev)) {
        throw NumericError(NumericStatus::InvalidInput, "invalid negative-flux Gaussian");
    }
    if (k >= 0.0) return {1.0, 0.0, 0.0, NumericStatus::Ok};
    if (stddev == 0.0) {
        if (!(mean < 0.0)) return exactZero();
        return k < mean ? exactZero() : PositiveResult{1.0, 0.0, 0.0, NumericStatus::Degenerate};
    }
    const PositiveResult denominator = negativePartMean(mean, stddev, policy);
    if (denominator.status == NumericStatus::ExactZero) return exactZero();
    const double z = (k - mean) / stddev;
    const double direct = stddev * normalPdf(z) - mean * normalCdf(z);
    if (direct > 0.0 && std::isfinite(direct)) {
        const double value = std::clamp(direct / denominator.value, 0.0, 1.0);
        return {value, value > 0.0 ? std::log(value) : -std::numeric_limits<double>::infinity(),
                0.0, value > 0.0 ? NumericStatus::Ok : NumericStatus::ExactZero};
    }

    auto logIntegrand = [=](double u) {
        const double factor = -k + stddev * u;
        return std::log(factor) + normalLogPdf(z - u);
    };
    const double mode = std::max(0.0, z + std::sqrt(z * z + 4.0));
    const PositiveResult numerator = integratePositiveLog(logIntegrand, 0.0, mode, policy);
    if (numerator.status == NumericStatus::ExactZero) return numerator;
    const double logValue = numerator.logValue - denominator.logValue;
    PositiveResult result = positiveFromLog(logValue, numerator.absError / denominator.value,
                                            numerator.status);
    if (result.value > 1.0 && result.value - 1.0 <= 64.0 * std::numeric_limits<double>::epsilon()) {
        result.value = 1.0;
        result.logValue = 0.0;
    }
    return result;
}

double sampleNegativeFluxNormal(double mean, double stddev, Random& rng,
                                const NumericPolicy& policy) {
    if (stddev == 0.0) {
        if (mean < 0.0) return mean;
        throw NumericError(NumericStatus::InvalidInput, "negative-flux distribution has no mass");
    }
    const double u = rng.openUniform01();
    auto cdf = [&](double k) { return negativeFluxNormalCdf(k, mean, stddev, policy).value; };
    double lower = std::min(-stddev, mean - 8.0 * stddev);
    for (int i = 0; i < 128 && cdf(lower) >= u; ++i) lower = lower * 2.0 - stddev;
    const RootResult root = solveMonotoneIncreasing(cdf, u, lower, 0.0, policy);
    if (root.status != NumericStatus::Ok) {
        throw NumericError(root.status, "failed to sample negative-flux Gaussian");
    }
    return std::min(root.value, -std::numeric_limits<double>::min());
}

} // namespace mf

