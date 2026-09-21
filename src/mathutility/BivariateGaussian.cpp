#include "macrofacet/mathutility/BivariateGaussian.h"
#include "macrofacet/mathutility/Gaussian1D.h"
#include "macrofacet/mathutility/Quadrature.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace mf {

PositiveResult standardBivariateNormalCdf(double a, double b, double correlation,
                                          const NumericPolicy& policy) {
    const double rhoTolerance = 128.0 * std::numeric_limits<double>::epsilon();
    if (!std::isfinite(a) || !std::isfinite(b) ||
        correlation < -1.0 - rhoTolerance || correlation > 1.0 + rhoTolerance) {
        throw NumericError(NumericStatus::InvalidInput, "invalid bivariate normal parameters");
    }
    const double rho = std::clamp(correlation, -1.0, 1.0);
    if (rho == 0.0) return positiveFromLog(normalLogCdf(a) + normalLogCdf(b));
    if (rho == 1.0) return positiveFromLog(normalLogCdf(std::min(a, b)));
    if (rho == -1.0) {
        const double value = normalCdf(a) - normalCdf(-b);
        if (value <= 0.0) return exactZero();
        return {value, std::log(value), 0.0, NumericStatus::Degenerate};
    }

    const double residual = std::sqrt((1.0 - rho) * (1.0 + rho));
    auto logIntegrand = [=](double u) {
        const double x = a - u;
        return normalLogPdf(x) + normalLogCdf((b - rho * x) / residual);
    };
    auto derivative = [=](double u) {
        const double x = a - u;
        const double z = (b - rho * x) / residual;
        return x + (rho / residual) * normalPdfOverCdf(z);
    };
    double mode = 0.0;
    if (derivative(1e-14) > 0.0) {
        double lo = 1e-14;
        double hi = std::max(1.0, std::abs(a) + 1.0);
        while (derivative(hi) > 0.0 && hi < 1e150) hi *= 2.0;
        for (int i = 0; i < 128; ++i) {
            const double middle = lo + 0.5 * (hi - lo);
            if (derivative(middle) > 0.0) lo = middle;
            else hi = middle;
        }
        mode = lo + 0.5 * (hi - lo);
    }
    return integratePositiveLog(logIntegrand, 0.0, mode, policy);
}

PositiveResult positiveOrthant2(const Vector2& mean, const Matrix2& covariance,
                                const NumericPolicy& policy) {
    Matrix2 cov = 0.5 * (covariance + covariance.transpose());
    const double tolerance = covarianceTolerance(std::max(1.0, cov.norm()), policy);
    if (cov(0, 0) < -tolerance || cov(1, 1) < -tolerance) {
        throw NumericError(NumericStatus::InvalidCovariance, "negative bivariate variance");
    }
    const bool deterministic0 = cov(0, 0) <= tolerance;
    const bool deterministic1 = cov(1, 1) <= tolerance;
    if (deterministic0 && mean[0] <= 0.0) return exactZero();
    if (deterministic1 && mean[1] <= 0.0) return exactZero();
    if (deterministic0 && deterministic1) {
        return {1.0, 0.0, 0.0, NumericStatus::Degenerate};
    }
    if (deterministic0) return positiveFromLog(normalLogCdf(mean[1] / std::sqrt(cov(1, 1))));
    if (deterministic1) return positiveFromLog(normalLogCdf(mean[0] / std::sqrt(cov(0, 0))));
    const double s0 = std::sqrt(cov(0, 0));
    const double s1 = std::sqrt(cov(1, 1));
    return standardBivariateNormalCdf(mean[0] / s0, mean[1] / s1,
                                      cov(0, 1) / (s0 * s1), policy);
}

} // namespace mf
