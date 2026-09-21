#include "macrofacet/mathutility/GaussianScreenIntegral.h"
#include "macrofacet/mathutility/Gaussian1D.h"
#include "macrofacet/mathutility/GaussianMoments1D.h"
#include "macrofacet/mathutility/Quadrature.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace mf {

PositiveResult negativeFluxTimesCdf(double meanK, double stddevK, double a, double b,
                                    const NumericPolicy& policy) {
    if (stddevK < 0.0 || !std::isfinite(meanK) || !std::isfinite(stddevK) ||
        !std::isfinite(a) || !std::isfinite(b)) {
        throw NumericError(NumericStatus::InvalidInput, "invalid screened flux parameters");
    }
    if (stddevK == 0.0) {
        const double value = std::max(-meanK, 0.0) * normalCdf(a + b * meanK);
        return value > 0.0 ? PositiveResult{value, std::log(value), 0.0, NumericStatus::Degenerate}
                           : exactZero();
    }
    if (b == 0.0) {
        PositiveResult moment = negativePartMean(meanK, stddevK, policy);
        if (moment.status == NumericStatus::ExactZero) return moment;
        return positiveFromLog(moment.logValue + normalLogCdf(a), moment.absError, moment.status);
    }
    auto logIntegrand = [=](double u) {
        if (!(u > 0.0)) return -std::numeric_limits<double>::infinity();
        return std::log(stddevK) + std::log(u) + normalLogPdf(u + meanK / stddevK) +
               normalLogCdf(a - b * stddevK * u);
    };
    const double normalizedMean = meanK / stddevK;
    auto derivative = [=](double u) {
        const double screenArgument = a - b * stddevK * u;
        return 1.0 / u - (u + normalizedMean) -
               b * stddevK * normalPdfOverCdf(screenArgument);
    };
    double lo = 1e-14;
    double hi = std::max(1.0, std::abs(normalizedMean) + 1.0);
    while (derivative(hi) > 0.0 && hi < 1e150) hi *= 2.0;
    for (int i = 0; i < 128; ++i) {
        const double middle = lo + 0.5 * (hi - lo);
        if (derivative(middle) > 0.0) lo = middle;
        else hi = middle;
    }
    const double mode = lo + 0.5 * (hi - lo);
    PositiveResult result = integratePositiveLog(logIntegrand, 0.0, mode, policy);
    const PositiveResult maximum = negativePartMean(meanK, stddevK, policy);
    if (result.value > maximum.value && result.value - maximum.value <=
        256.0 * std::numeric_limits<double>::epsilon() * std::max(1.0, maximum.value)) {
        result = maximum;
    }
    return result;
}

PositiveResult negativeFluxTimesAffineIndicator(double meanK, double stddevK,
                                                double intercept, double slope,
                                                const NumericPolicy& policy) {
    if (stddevK < 0.0 || !std::isfinite(meanK) || !std::isfinite(stddevK) ||
        !std::isfinite(intercept) || !std::isfinite(slope)) {
        throw NumericError(NumericStatus::InvalidInput, "invalid affine-screen parameters");
    }
    if (stddevK == 0.0) {
        const double value = meanK < 0.0 && intercept + slope * meanK > 0.0 ? -meanK : 0.0;
        return value > 0.0 ? PositiveResult{value, std::log(value), 0.0, NumericStatus::Degenerate}
                           : exactZero();
    }
    if (slope == 0.0) return intercept > 0.0 ? negativePartMean(meanK, stddevK, policy)
                                             : exactZero();

    double lower = -std::numeric_limits<double>::infinity();
    double upper = 0.0;
    const double threshold = -intercept / slope;
    if (slope > 0.0) lower = threshold;
    else upper = std::min(upper, threshold);
    if (!(lower < upper)) return exactZero();

    auto integrand = [=](double k) {
        const double z = (k - meanK) / stddevK;
        return (-k) * normalPdf(z) / stddevK;
    };
    IntegralResult integral;
    if (std::isfinite(lower)) {
        integral = integrateFinite(integrand, lower, upper, policy);
    } else {
        auto reversed = [=](double u) { return integrand(upper - u); };
        integral = integrateSemiInfinite(reversed, 0.0, policy);
    }
    if (!(integral.value > 0.0)) return exactZero();
    return {integral.value, std::log(integral.value), integral.absError, integral.status};
}

} // namespace mf
