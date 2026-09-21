#include "macrofacet/mathutility/Gaussian1D.h"
#include "macrofacet/mathutility/RootFinding.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace mf {

double normalLogPdf(double z) {
    return -0.5 * z * z - 0.5 * std::log(2.0 * kPi);
}

double normalPdf(double z) { return std::exp(normalLogPdf(z)); }

double normalCdf(double z) { return 0.5 * std::erfc(-z / kSqrtTwo); }

double normalLogCdf(double z) {
    if (z > 0.0) return std::log1p(-0.5 * std::erfc(z / kSqrtTwo));
    const double ordinary = 0.5 * std::erfc(-z / kSqrtTwo);
    if (ordinary > 0.0) return std::log(ordinary);

    const long double x = -static_cast<long double>(z);
    long double term = 1.0L;
    long double sum = 1.0L;
    long double previous = std::numeric_limits<long double>::infinity();
    for (int n = 1; n < 256; ++n) {
        term *= -static_cast<long double>(2 * n - 1) / (x * x);
        if (std::abs(term) >= previous) break;
        sum += term;
        previous = std::abs(term);
        if (std::abs(term) < 1e-30L * std::abs(sum)) break;
    }
    return static_cast<double>(-0.5L * x * x - 0.5L * std::log(2.0L * kPi) -
                               std::log(x) + std::log(sum));
}

double normalSurvival(double z) { return normalCdf(-z); }
double normalLogSurvival(double z) { return normalLogCdf(-z); }

double normalPdfOverCdf(double z) {
    if (z > -10.0) return std::exp(normalLogPdf(z) - normalLogCdf(z));
    const double x = -z;
    const double inverse = 1.0 / x;
    const double inverse2 = inverse * inverse;
    return x + inverse * (1.0 - 2.0 * inverse2 + 10.0 * inverse2 * inverse2);
}

double normalQuantile(double u) {
    if (!(u > 0.0 && u < 1.0)) throw std::invalid_argument("normal quantile requires 0<u<1");
    if (u > 0.5) return -normalQuantile(1.0 - u);
    return normalQuantileFromLogCdf(std::log(u));
}

double normalQuantileFromLogCdf(double logU) {
    if (!(logU < 0.0) || std::isnan(logU)) {
        throw std::invalid_argument("log CDF quantile requires a finite negative value");
    }
    if (logU > std::log(0.5)) return -normalQuantile(-std::expm1(logU));
    double lo = -2.0;
    while (normalLogCdf(lo) > logU) lo *= 2.0;
    NumericPolicy inversePolicy = defaultNumericPolicy();
    inversePolicy.relativeTolerance = 1e-13;
    inversePolicy.absoluteTolerance = 1e-14;
    inversePolicy.maxRootIterations = std::max(inversePolicy.maxRootIterations, 256);
    const RootResult result = solveMonotoneIncreasing(normalLogCdf, logU, lo, 0.0,
                                                      inversePolicy);
    if (result.status != NumericStatus::Ok) {
        throw NumericError(result.status, "failed to invert normal log CDF");
    }
    return result.value;
}

double sampleStandardNormal(Random& rng) { return normalQuantile(rng.openUniform01()); }

} // namespace mf
